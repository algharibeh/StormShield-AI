"""Export per-frame data for the radar-enhancement animation.

Writes one compact JSON per scene into the visualization folder, holding for
every keyframe: ego pose, ground-truth boxes with velocity, the LiDAR+camera
boxes with their velocity before and after radar, and the radar returns.
No raw sensor data is copied, so the output is small enough for git.

Usage:
  python scripts/export_viz_data.py --dataroot data/nuscenes \
      --base results/fused/L+C.json --radar results/fused/L+C+Rv.json \
      --lidar results/fused/L.json --out "visualizing data/data" --n-scenes 2
"""
import argparse
import json
import os
import sys

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(__file__))
from common import (available_samples, box_yaw, gt_boxes_of_sample, load_nusc, load_results,
                    points_in_box_bev, radar_points_global)

TP_DIST = 2.0


def ego_pose_xyyaw(nusc, tok):
    s = nusc.get("sample", tok)
    sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    yaw = Quaternion(ep["rotation"]).yaw_pitch_roll[0]
    return [float(ep["translation"][0]), float(ep["translation"][1]), float(yaw)], s["timestamp"]


def match(preds, gts):
    used = np.zeros(len(gts), dtype=bool)
    out = {}
    for i, p in sorted(enumerate(preds), key=lambda ip: -ip[1]["detection_score"]):
        c = np.array(p["translation"][:2])
        for j, g in enumerate(gts):
            if used[j] or g["detection_name"] != p["detection_name"]:
                continue
            if np.linalg.norm(np.array(g["translation"][:2]) - c) <= TP_DIST:
                used[j] = True
                out[i] = j
                break
    return out


def scene_interest(nusc, tokens):
    """Moving ground-truth objects beyond 20 m, summed over the scene's keyframes."""
    n = 0
    for t in tokens:
        ego, _ = ego_pose_xyyaw(nusc, t)
        for g in gt_boxes_of_sample(nusc, t):
            d = np.hypot(g["translation"][0] - ego[0], g["translation"][1] - ego[1])
            if np.hypot(*g["velocity"]) > 3 and d > 20:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--base", required=True)
    ap.add_argument("--radar", required=True)
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--out", default="visualizing data/data")
    ap.add_argument("--n-scenes", type=int, default=2)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names to export instead of auto-pick")
    ap.add_argument("--score-min", type=float, default=0.3)
    args = ap.parse_args()

    nusc = load_nusc(args.dataroot, args.version)
    base, _ = load_results(args.base)
    radar, _ = load_results(args.radar)
    lidar_only, _ = load_results(args.lidar)
    val = [t for t in available_samples(nusc, "val") if t in base]
    by_scene = {}
    for t in val:
        by_scene.setdefault(nusc.get("sample", t)["scene_token"], []).append(t)
    for st in by_scene:
        by_scene[st].sort(key=lambda t: nusc.get("sample", t)["timestamp"])

    if args.scenes:
        chosen = [st for st in by_scene if nusc.get("scene", st)["name"] in args.scenes]
    else:
        ranked = sorted(by_scene, key=lambda st: -scene_interest(nusc, by_scene[st]))
        chosen = ranked[: args.n_scenes]
    os.makedirs(args.out, exist_ok=True)

    key = lambda b: (round(b["translation"][0], 3), round(b["translation"][1], 3), b["detection_name"])
    for st in chosen:
        scene = nusc.get("scene", st)
        frames = []
        for tok in by_scene[st]:
            ego, ts = ego_pose_xyyaw(nusc, tok)
            gts = gt_boxes_of_sample(nusc, tok)
            rp = radar_points_global(nusc, tok)
            preds = [b for b in base.get(tok, []) if b["detection_score"] >= args.score_min]
            rmap = {key(b): b for b in radar.get(tok, [])}
            lid = {(round(b["translation"][0], 3), round(b["translation"][1], 3)) for b in lidar_only.get(tok, [])}
            m = match(preds, gts)
            boxes = []
            for i, p in enumerate(preds):
                pr = rmap.get(key(p), p)
                inside = points_in_box_bev(rp[:, :2], p["translation"][:2], p["size"][:2], box_yaw(p["rotation"]), 1.0) if len(rp) else np.zeros(0, bool)
                gj = m.get(i)
                gv = np.array(gts[gj]["velocity"]) if gj is not None else None
                boxes.append({
                    "x": round(p["translation"][0], 2), "y": round(p["translation"][1], 2),
                    "yaw": round(box_yaw(p["rotation"]), 3),
                    "w": round(p["size"][0], 2), "l": round(p["size"][1], 2),
                    "cls": p["detection_name"], "score": round(p["detection_score"], 3),
                    "origin": "lidar" if (round(p["translation"][0], 3), round(p["translation"][1], 3)) in lid else "camera",
                    "v_base": [round(p["velocity"][0], 2), round(p["velocity"][1], 2)],
                    "v_radar": [round(pr["velocity"][0], 2), round(pr["velocity"][1], 2)],
                    "n_radar": int(inside.sum()),
                    "gt": gj,
                    "err_base": round(float(np.linalg.norm(np.array(p["velocity"]) - gv)), 3) if gv is not None else None,
                    "err_radar": round(float(np.linalg.norm(np.array(pr["velocity"]) - gv)), 3) if gv is not None else None,
                })
            frames.append({
                "token": tok, "timestamp": ts, "ego": ego,
                "gt": [{"x": round(g["translation"][0], 2), "y": round(g["translation"][1], 2),
                        "yaw": round(box_yaw(g["rotation"]), 3), "w": round(g["size"][0], 2), "l": round(g["size"][1], 2),
                        "cls": g["detection_name"], "v": [round(g["velocity"][0], 2), round(g["velocity"][1], 2)]} for g in gts],
                "boxes": boxes,
                "radar": [[round(float(v), 2) for v in row] for row in rp.tolist()],
            })
        out = {
            "scene": scene["name"], "description": scene["description"], "location": nusc.get("log", scene["log_token"])["location"],
            "source": "nuScenes v1.0-trainval (Motional), derived data for non-commercial research, CC BY-NC-SA 4.0",
            "systems": {"base": "LiDAR (CenterPoint) + camera (FCOS3D), late fusion",
                        "radar": "same boxes with radar-corrected velocity"},
            "frames": frames,
        }
        path = os.path.join(args.out, f"{scene['name']}.json")
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"wrote {path}: {len(frames)} frames, {os.path.getsize(path)/1e6:.1f} MB, {scene['description']}")


if __name__ == "__main__":
    main()
