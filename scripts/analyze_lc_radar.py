"""Where does radar help the LiDAR + camera system? Velocity error of matched
true positives broken down by distance, object speed, class and box origin.

Compares two fused result files that contain the same boxes with different
velocities (L+C vs L+C+Rv). Matching: greedy by score, same class, BEV centre
distance <= 2 m, like the devkit's TP definition.

Usage:
  python scripts/analyze_lc_radar.py --dataroot data/nuscenes \
      --base results/fused/L+C.json --radar results/fused/L+C+Rv.json \
      --lidar results/fused/L.json --out results/analysis_lc_radar.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (available_samples, ego_pose_of_sample, gt_boxes_of_sample, load_nusc,
                    load_results, radar_points_global, points_in_box_bev, box_yaw)

TP_DIST = 2.0
DIST_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
SPEED_BUCKETS = [("parked, under 0.5 m/s", 0, 0.5), ("slow, 0.5 to 3 m/s", 0.5, 3), ("moving, over 3 m/s", 3, 1e9)]


def match(preds, gts):
    """Return list of (pred, gt) pairs, greedy by score, same class, <= 2 m."""
    used = np.zeros(len(gts), dtype=bool)
    pairs = []
    for p in sorted(preds, key=lambda b: -b["detection_score"]):
        c = np.array(p["translation"][:2])
        for i, g in enumerate(gts):
            if used[i] or g["detection_name"] != p["detection_name"]:
                continue
            if np.linalg.norm(np.array(g["translation"][:2]) - c) <= TP_DIST:
                used[i] = True
                pairs.append((p, g))
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--base", required=True)
    ap.add_argument("--radar", required=True)
    ap.add_argument("--lidar", required=True, help="LiDAR-only result file, to tag box origin")
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--out", default="results/analysis_lc_radar.json")
    args = ap.parse_args()

    nusc = load_nusc(args.dataroot, args.version)
    base, _ = load_results(args.base)
    radar, _ = load_results(args.radar)
    lidar_only, _ = load_results(args.lidar)
    tokens = [t for t in available_samples(nusc, "val") if t in base]

    rows = []  # one per matched true positive
    n_boxes = n_with_radar = 0
    for tok in tokens:
        ego = np.array(ego_pose_of_sample(nusc, tok)["translation"][:2])
        gts = gt_boxes_of_sample(nusc, tok)
        rp = radar_points_global(nusc, tok)
        lid_keys = {(round(b["translation"][0], 3), round(b["translation"][1], 3)) for b in lidar_only.get(tok, [])}
        preds_b = [b for b in base.get(tok, []) if b["detection_score"] >= args.score_min]
        # radar file has identical boxes in identical order (same scores), so index-align by position
        preds_r = [b for b in radar.get(tok, []) if b["detection_score"] >= args.score_min]
        key = lambda b: (round(b["translation"][0], 3), round(b["translation"][1], 3), b["detection_name"])
        rmap = {key(b): b for b in preds_r}
        for p, g in match(preds_b, gts):
            pr = rmap.get(key(p))
            if pr is None:
                continue
            n_boxes += 1
            m = points_in_box_bev(rp[:, :2], p["translation"][:2], p["size"][:2], box_yaw(p["rotation"]), 1.0) if len(rp) else np.zeros(0, bool)
            has_radar = bool(m.sum() > 0)
            n_with_radar += has_radar
            gv = np.array(g["velocity"])
            e_b = float(np.linalg.norm(np.array(p["velocity"]) - gv))
            e_r = float(np.linalg.norm(np.array(pr["velocity"]) - gv))
            rows.append({
                "dist": float(np.linalg.norm(np.array(g["translation"][:2]) - ego)),
                "speed": float(np.linalg.norm(gv)), "cls": g["detection_name"],
                "origin": "lidar" if (round(p["translation"][0], 3), round(p["translation"][1], 3)) in lid_keys else "camera",
                "has_radar": has_radar, "err_base": e_b, "err_radar": e_r,
            })

    def summarize(sel, name):
        r = [x for x in rows if sel(x)]
        if not r:
            return {"name": name, "n": 0}
        eb = np.mean([x["err_base"] for x in r]); er = np.mean([x["err_radar"] for x in r])
        return {"name": name, "n": len(r), "err_base": float(eb), "err_radar": float(er),
                "change_pct": float(100 * (er - eb) / eb) if eb > 0 else 0.0,
                "frac_with_radar": float(np.mean([x["has_radar"] for x in r]))}

    out = {
        "n_frames": len(tokens), "n_true_positives": n_boxes, "frac_tp_with_radar": n_with_radar / max(n_boxes, 1),
        "overall": summarize(lambda x: True, "all true positives"),
        "by_distance": [summarize(lambda x, lo=lo, hi=hi: lo <= x["dist"] < hi, f"{lo} to {hi} m") for lo, hi in DIST_BUCKETS],
        "by_speed": [summarize(lambda x, lo=lo, hi=hi: lo <= x["speed"] < hi, nm) for nm, lo, hi in SPEED_BUCKETS],
        "by_origin": [summarize(lambda x, o=o: x["origin"] == o, f"{o}-origin boxes") for o in ("lidar", "camera")],
        "by_class": [summarize(lambda x, c=c: x["cls"] == c, c) for c in
                     ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle"]],
        "moving_far": summarize(lambda x: x["speed"] >= 3 and x["dist"] >= 30, "moving, beyond 30 m"),
        "moving_with_radar": summarize(lambda x: x["speed"] >= 3 and x["has_radar"], "moving, with radar returns"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k in ("n_frames", "n_true_positives", "frac_tp_with_radar", "overall", "moving_far", "moving_with_radar")}, indent=2))
    for key in ("by_distance", "by_speed", "by_origin", "by_class"):
        print(f"\n{key}")
        for s in out[key]:
            if s["n"]:
                print(f"  {s['name']:28s} n={s['n']:6d}  base {s['err_base']:.3f}  radar {s['err_radar']:.3f}  change {s['change_pct']:+.1f}%  with-radar {s['frac_with_radar']:.2f}")


if __name__ == "__main__":
    main()
