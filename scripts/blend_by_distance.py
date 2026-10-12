"""Distance-dependent radar velocity blend for the LiDAR + camera system.

Fits, on training-split boxes, one blend weight per distance bucket and box
origin (v = (1-a) v_detector + a v_radar), applies it to the validation
boxes, and writes a new fused result file for the devkit.

Usage:
  python scripts/blend_by_distance.py --dataroot data/nuscenes \
      --lidar detections/centerpoint.json --camera detections/fcos3d.json \
      --base results/fused/L+C.json --out results/fused_lc/L+C+Ra.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import available_samples, ego_pose_of_sample, load_nusc, load_results, save_results
from fuse import build_rows, nms_center, clean, MAX_BOXES

BUCKETS = [(0, 20), (20, 30), (30, 40), (40, 1e9)]
WEIGHTS = np.linspace(0, 1, 21)


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return len(BUCKETS) - 1


def dist_of(nusc, b):
    ego = np.array(ego_pose_of_sample(nusc, b["_token"])["translation"][:2])
    return float(np.linalg.norm(np.array(b["translation"][:2]) - ego))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--base", required=True, help="L+C result file (defines which boxes survive the merge)")
    ap.add_argument("--out", default="results/fused_lc/L+C+Ra.json")
    ap.add_argument("--summary", default="results/blend_by_distance.json")
    args = ap.parse_args()

    nusc = load_nusc(args.dataroot, args.version)
    dets = {"lidar": load_results(args.lidar)[0], "camera": load_results(args.camera)[0]}
    train_tokens = [t for t in available_samples(nusc, "train") if t in dets["lidar"]]
    val_tokens = [t for t in available_samples(nusc, "val") if t in dets["lidar"]]
    tr_boxes, _, _ = build_rows(nusc, train_tokens, dets, with_gt=True)
    va_boxes, _, _ = build_rows(nusc, val_tokens, dets, with_gt=True)

    # fit a per (origin, bucket) on training true positives that have radar returns
    weights = {}
    report = {"fit": [], "val": []}
    for origin in ("lidar", "camera"):
        for bi in range(len(BUCKETS)):
            sel = [b for b in tr_boxes if b["_modality"] == origin and b["_radar_n"] > 0 and "_gt_velocity" in b
                   and bucket_of(dist_of(nusc, b)) == bi]
            if len(sel) < 30:
                weights[(origin, bi)] = 0.1 if origin == "lidar" else 1.0
                report["fit"].append({"origin": origin, "bucket": BUCKETS[bi], "n": len(sel), "a": weights[(origin, bi)], "note": "too few, default"})
                continue
            errs = []
            for a in WEIGHTS:
                errs.append(np.mean([np.linalg.norm((1 - a) * np.array(b["velocity"]) + a * b["_radar_v"] - b["_gt_velocity"]) for b in sel]))
            a_best = float(WEIGHTS[int(np.argmin(errs))])
            weights[(origin, bi)] = a_best
            report["fit"].append({"origin": origin, "bucket": BUCKETS[bi], "n": len(sel), "a": a_best,
                                  "err_a0": float(errs[0]), "err_best": float(min(errs))})
            print(f"fit {origin:6s} {BUCKETS[bi]}: n={len(sel):5d} a={a_best:.2f} err {errs[0]:.3f} -> {min(errs):.3f}")

    # apply to validation boxes and compare against fixed 0.1 on matched true positives
    base, _ = load_results(args.base)
    keep = {t: {(round(b["translation"][0], 3), round(b["translation"][1], 3), b["detection_name"]) for b in base.get(t, [])} for t in val_tokens}
    results = {t: [] for t in val_tokens}
    for b in va_boxes:
        k = (round(b["translation"][0], 3), round(b["translation"][1], 3), b["detection_name"])
        if k not in keep[b["_token"]]:
            continue
        b = dict(b)
        if b["_radar_n"] > 0:
            a = weights[(b["_modality"], bucket_of(dist_of(nusc, b)))]
            v = (1 - a) * np.array(b["velocity"]) + a * b["_radar_v"]
            b["velocity"] = [float(v[0]), float(v[1])]
        results[b["_token"]].append(b)
    for t in val_tokens:
        results[t] = [clean(x) for x in sorted(results[t], key=lambda x: -x["detection_score"])[:MAX_BOXES]]
    save_results(args.out, results, True, True, True)
    print("wrote", args.out)

    # summary on validation true positives, per bucket
    for bi in range(len(BUCKETS)):
        sel = [b for b in va_boxes if "_gt_velocity" in b and b["_radar_n"] > 0 and bucket_of(dist_of(nusc, b)) == bi]
        if not sel:
            continue
        e0 = np.mean([np.linalg.norm(np.array(b["velocity"]) - b["_gt_velocity"]) for b in sel])
        e1 = np.mean([np.linalg.norm(0.9 * np.array(b["velocity"]) + 0.1 * b["_radar_v"] - b["_gt_velocity"]) for b in sel])
        ea = np.mean([np.linalg.norm((1 - weights[(b["_modality"], bi)]) * np.array(b["velocity"]) + weights[(b["_modality"], bi)] * b["_radar_v"] - b["_gt_velocity"]) for b in sel])
        report["val"].append({"bucket": BUCKETS[bi], "n": len(sel), "err_no_radar": float(e0), "err_fixed_0.1": float(e1), "err_adaptive": float(ea)})
        print(f"val {BUCKETS[bi]}: n={len(sel):5d} no-radar {e0:.3f} fixed0.1 {e1:.3f} adaptive {ea:.3f}")
    json.dump(report, open(args.summary, "w"), indent=2)


if __name__ == "__main__":
    main()
