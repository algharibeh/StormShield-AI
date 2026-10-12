"""Present-class metrics for a devkit evaluation slice.

The nuScenes devkit scores a class with no ground truth in the evaluated
frames as AP 0 and every true-positive error 1.0. On a small slice (the 15
night scenes of blob 10 have no bus, trailer or construction vehicle and only
a handful of bicycles and traffic cones) that padding dominates mAP and mAVE.
This script re-averages the saved per-class values over the classes that have
at least --min-gt ground-truth boxes in the slice, so that systems can be
compared on what the slice actually contains.

Usage:
  python scripts/slice_metrics.py --summary results/eval_b1b10/summary_min0_max0_night.json \
      --dataroot data/nuscenes --scene-keyword night [--min-range 30] [--min-gt 20]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

NAMES = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
         "motorcycle", "bicycle", "traffic_cone", "barrier"]
VEL_CLASSES = [n for n in NAMES if n not in ("traffic_cone", "barrier")]


def gt_counts(dataroot, scene_keyword, min_range):
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    meta = os.path.join(dataroot, "v1.0-trainval")
    load = lambda n: json.load(open(os.path.join(meta, n + ".json")))
    val = set(create_splits_scenes()["val"])
    scenes = {s["token"]: s for s in load("scene")}
    keep = {}
    for s in load("sample"):
        sc = scenes[s["scene_token"]]
        if sc["name"] not in val:
            continue
        if scene_keyword and scene_keyword.lower() not in sc["description"].lower():
            continue
        keep[s["token"]] = s
    cat = {c["token"]: c["name"] for c in load("category")}
    inst = {i["token"]: i["category_token"] for i in load("instance")}
    # raw sample.json carries no "data" map; find each keyframe's LIDAR_TOP record by hand,
    # keep only frames whose LiDAR file is on disk (the devkit evaluation used exactly those)
    sensors = {s["token"]: s["channel"] for s in load("sensor")}
    cs_chan = {c["token"]: sensors[c["sensor_token"]] for c in load("calibrated_sensor")}
    ego_tok = {}
    for d in load("sample_data"):
        if d["is_key_frame"] and d["sample_token"] in keep and cs_chan[d["calibrated_sensor_token"]] == "LIDAR_TOP":
            if os.path.exists(os.path.join(dataroot, d["filename"])):
                ego_tok[d["sample_token"]] = d["ego_pose_token"]
    keep = {t: s for t, s in keep.items() if t in ego_tok}
    print("frames on disk in this slice:", len(keep))
    ego = None
    if min_range > 0:
        poses = {e["token"]: e for e in load("ego_pose")}
        ego = {tok: poses[ep] for tok, ep in ego_tok.items()}
    counts = Counter()
    for a in load("sample_annotation"):
        s = keep.get(a["sample_token"])
        if s is None:
            continue
        name = category_to_detection_name(cat[inst[a["instance_token"]]])
        if name is None:
            continue
        if min_range > 0:
            ep = ego[s["token"]]
            d = np.linalg.norm(np.array(a["translation"][:2]) - np.array(ep["translation"][:2]))
            if d < min_range:
                continue
        counts[name] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--scene-keyword", default=None)
    ap.add_argument("--min-range", type=float, default=0.0)
    ap.add_argument("--min-gt", type=int, default=20)
    ap.add_argument("--out", default=None, help="JSON to write (default: next to the summary)")
    args = ap.parse_args()

    counts = gt_counts(args.dataroot, args.scene_keyword, args.min_range)
    present = [n for n in NAMES if counts.get(n, 0) >= args.min_gt]
    present_vel = [n for n in present if n in VEL_CLASSES]
    print("ground truth per class:", dict(counts))
    print(f"classes with at least {args.min_gt} boxes: {present}")

    summary = json.load(open(args.summary))
    rows = {}
    print(f"{'config':8} {'mAP*':>7} {'mAVE*':>7} {'mATE*':>7} {'carAP':>7} {'carAVE':>7}")
    for cfg, r in summary.items():
        if "per_class_tp" not in r:
            print(f"{cfg}: summary lacks per_class_tp (rerun evaluate.py)")
            continue
        aps = r["per_class_ap"]
        tp = r["per_class_tp"]
        row = {
            "mAP_present": float(np.mean([aps[n] for n in present])),
            "mAVE_present": float(np.mean([tp[n]["vel_err"] for n in present_vel])),
            "mATE_present": float(np.mean([tp[n]["trans_err"] for n in present])),
            "car_AP": aps["car"], "car_AVE": tp["car"]["vel_err"],
            "classes": present, "vel_classes": present_vel,
        }
        rows[cfg] = row
        print(f"{cfg:8} {row['mAP_present']:7.3f} {row['mAVE_present']:7.3f} {row['mATE_present']:7.3f} "
              f"{row['car_AP']:7.3f} {row['car_AVE']:7.3f}")
    out = args.out or args.summary.replace("summary_", "present_")
    json.dump({"gt_counts": dict(counts), "min_gt": args.min_gt, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
