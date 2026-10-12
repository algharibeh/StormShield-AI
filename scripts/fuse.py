"""Late fusion of camera and LiDAR detections with radar cues.

Inputs: nuScenes-format detection JSON files produced by pretrained
detectors (see colab/ notebook). Outputs: one fused JSON per configuration,
ready for evaluate.py.

Configurations written:
  C        camera detections as they came
  C+R      camera re-scored with radar features, radar velocity
  L        LiDAR detections as they came
  L+R      LiDAR re-scored with radar features, blended velocity
  L+C      LiDAR and camera merged and re-scored, no radar
  L+C+R    LiDAR and camera merged and re-scored with radar, radar velocity

The re-scorer is a logistic regression fitted on the training-split samples
present locally. Nothing here needs a GPU.

Usage:
  python scripts/fuse.py --dataroot data/nuscenes --lidar detections/lidar_centerpoint.json \
      --camera detections/camera_fcos3d.json --out results/fused
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (DETECTION_NAMES, available_samples, box_yaw, ego_pose_of_sample,
                    gt_boxes_of_sample, load_nusc, load_results, points_in_box_bev,
                    radar_points_global, save_results)

MAX_BOXES = 500
TP_DIST = 2.0
N_BASE = 3 + len(DETECTION_NAMES)


def radar_features(box, radar):
    """Radar cues for one box: n points, max rcs, mean compensated velocity (vx, vy)."""
    if radar.shape[0] == 0:
        return 0, 0.0, np.zeros(2)
    m = points_in_box_bev(radar[:, :2], box["translation"][:2], box["size"][:2],
                          box_yaw(box["rotation"]), margin=1.0)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, np.zeros(2)
    sel = radar[m]
    return n, float(sel[:, 4].max()), sel[:, 2:4].mean(axis=0)


def match_gt(boxes, gts):
    """Greedy centre-distance matching, boxes sorted by score desc. Returns labels and matched GT index."""
    used = np.zeros(len(gts), dtype=bool)
    labels, matched = [], []
    for b in boxes:
        c = np.array(b["translation"][:2])
        hit = -1
        for i, g in enumerate(gts):
            if used[i] or g["detection_name"] != b["detection_name"]:
                continue
            if np.linalg.norm(np.array(g["translation"][:2]) - c) <= TP_DIST:
                used[i] = True
                hit = i
                break
        labels.append(1 if hit >= 0 else 0)
        matched.append(hit)
    return labels, matched


def build_rows(nusc, tokens, dets_by_modality, with_gt):
    """Flatten detections into feature rows. Returns (boxes, X, y)."""
    boxes, feats, labels = [], [], []
    for tok in tokens:
        radar = radar_points_global(nusc, tok)
        ego = np.array(ego_pose_of_sample(nusc, tok)["translation"][:2])
        cands = []
        for modality, dets in dets_by_modality.items():
            for b in dets.get(tok, []):
                b = dict(b)
                b["_modality"] = modality
                b["_token"] = tok
                cands.append(b)
        cands.sort(key=lambda b: -b["detection_score"])
        for b in cands:
            n, rcs, v_r = radar_features(b, radar)
            v_b = np.array(b.get("velocity", [0.0, 0.0]), dtype=float)
            dist = float(np.linalg.norm(np.array(b["translation"][:2]) - ego))
            cls = np.zeros(len(DETECTION_NAMES))
            if b["detection_name"] in DETECTION_NAMES:
                cls[DETECTION_NAMES.index(b["detection_name"])] = 1.0
            f = np.concatenate([
                [b["detection_score"], dist / 50.0, 1.0 if b["_modality"] == "lidar" else 0.0],
                cls,
                [np.log1p(n), rcs / 30.0, float(np.linalg.norm(v_r)),
                 float(np.linalg.norm(v_b - v_r)) if n else 0.0],
            ])
            b["_radar_n"] = n
            b["_radar_v"] = v_r
            feats.append(f)
            boxes.append(b)
        if with_gt:
            # label each modality independently: a camera box for an object that LiDAR
            # also found is still a true positive of the camera detector
            gts = gt_boxes_of_sample(nusc, tok)
            lab_of = {}
            for modality in dets_by_modality:
                sub = [b for b in cands if b["_modality"] == modality]
                labs, matched = match_gt(sub, gts)
                for b, lab, m in zip(sub, labs, matched):
                    lab_of[id(b)] = lab
                    if m >= 0:
                        b["_gt_velocity"] = np.array(gts[m]["velocity"])
            labels.extend(lab_of[id(b)] for b in cands)
    X = np.array(feats) if feats else np.zeros((0, N_BASE + 4))
    y = np.array(labels) if with_gt else None
    return boxes, X, y


def fit_rescorer(X, y, use_radar):
    cols = slice(0, None) if use_radar else slice(0, N_BASE)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[:, cols], y)
    return clf, cols


def cv_scores(nusc, boxes, X, y, use_radar, n_folds=5):
    """Scene-level cross-validated re-scoring: each box is scored by a model
    fitted on the other folds' scenes, so no box is scored by a model that saw it."""
    scenes = np.array([nusc.get("sample", b["_token"])["scene_token"] for b in boxes])
    uniq = sorted(set(scenes))
    folds = {s: i % n_folds for i, s in enumerate(uniq)}
    fold_of = np.array([folds[s] for s in scenes])
    out = np.zeros(len(boxes))
    for f in range(n_folds):
        tr, te = fold_of != f, fold_of == f
        if te.sum() == 0:
            continue
        clf, cols = fit_rescorer(X[tr], y[tr], use_radar)
        out[te] = clf.predict_proba(X[te][:, cols])[:, 1]
    return out


def fit_velocity_blend(boxes):
    """Weight a in v = (1-a) v_lidar + a v_radar minimising velocity error on LiDAR true positives."""
    idx = [b for b in boxes if b["_modality"] == "lidar" and b["_radar_n"] > 0 and "_gt_velocity" in b]
    if not idx:
        return 0.0
    best_a, best_err = 0.0, float("inf")
    for a in np.linspace(0, 1, 11):
        errs = [np.linalg.norm((1 - a) * np.array(b["velocity"]) + a * b["_radar_v"] - b["_gt_velocity"])
                for b in idx]
        if np.mean(errs) < best_err:
            best_err, best_a = float(np.mean(errs)), float(a)
    print(f"velocity blend fitted on {len(idx)} LiDAR true positives, a = {best_a:.1f}, error {best_err:.3f} m/s")
    return best_a


def nms_center(boxes, thresh=1.0, lidar_primary_radius=2.5):
    """LiDAR-primary merge. LiDAR boxes are kept as they are (the detector already
    applied NMS). A camera box is dropped if a same-class LiDAR box lies within
    lidar_primary_radius metres, since monocular depth error makes it a duplicate,
    and camera boxes suppress each other within thresh metres."""
    lidar = [b for b in boxes if b["_modality"] == "lidar"]
    keep = list(lidar)
    for b in boxes:
        if b["_modality"] == "lidar":
            continue
        c = np.array(b["translation"][:2])
        ok = True
        for k in keep:
            if k["detection_name"] != b["detection_name"]:
                continue
            r = lidar_primary_radius if k["_modality"] == "lidar" else thresh
            if np.linalg.norm(np.array(k["translation"][:2]) - c) < r:
                ok = False
                break
        if ok:
            keep.append(b)
    return sorted(keep, key=lambda b: -b["detection_score"])


def clean(b):
    return {k: v for k, v in b.items() if not k.startswith("_")}


def write_config(path, boxes, scores, tokens, velocity_mode, blend_a, use_camera, use_lidar, use_radar, merge):
    results = {t: [] for t in tokens}
    for b, s in zip(boxes, scores):
        b = dict(b)
        b["detection_score"] = float(s)
        if velocity_mode == "radar" and b["_radar_n"] > 0:
            if b["_modality"] == "camera":
                b["velocity"] = [float(b["_radar_v"][0]), float(b["_radar_v"][1])]
            else:
                v = (1 - blend_a) * np.array(b["velocity"]) + blend_a * b["_radar_v"]
                b["velocity"] = [float(v[0]), float(v[1])]
        results[b["_token"]].append(b)
    for t in tokens:
        bs = sorted(results[t], key=lambda b: -b["detection_score"])
        if merge:
            bs = nms_center(bs)
        results[t] = [clean(b) for b in bs[:MAX_BOXES]]
    save_results(path, results, use_camera, use_lidar, use_radar)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--lidar", required=True, help="LiDAR detections JSON (nuScenes format)")
    ap.add_argument("--camera", required=True, help="camera detections JSON (nuScenes format)")
    ap.add_argument("--out", default="results/fused")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--fit-on", choices=["cv", "train"], default="cv",
                    help="cv: scene-level 5-fold cross-validation over the validation scenes (default); "
                         "train: fit on training-split scenes (biased, the pretrained detectors saw them)")
    args = ap.parse_args()

    nusc = load_nusc(args.dataroot, args.version)
    if args.version.endswith("mini"):
        args.train_split, args.val_split = "mini_train", "mini_val"
    lidar, _ = load_results(args.lidar)
    camera, _ = load_results(args.camera)
    have = set(lidar) & set(camera)
    train_tokens = [t for t in available_samples(nusc, args.train_split) if t in have]
    val_tokens = [t for t in available_samples(nusc, args.val_split) if t in have]
    print(f"train samples with detections: {len(train_tokens)}, val samples: {len(val_tokens)}")
    if not train_tokens or not val_tokens:
        sys.exit("need detections for both train and val samples; check the JSON files")

    dets = {"lidar": lidar, "camera": camera}
    tr_boxes, Xtr, ytr = build_rows(nusc, train_tokens, dets, with_gt=True)
    va_boxes, Xva, yva = build_rows(nusc, val_tokens, dets, with_gt=True)
    mod_tr = np.array([b["_modality"] for b in tr_boxes])
    mod_va = np.array([b["_modality"] for b in va_boxes])
    blend_a = fit_velocity_blend(tr_boxes)
    print(f"re-scorer fitted by: {args.fit_on}")

    # raw single-modality configs, and velocity-only radar variants (scores untouched)
    for name, mod, uc, ul in [("C", "camera", True, False), ("L", "lidar", False, True)]:
        sel = mod_va == mod
        chosen = [b for b, s in zip(va_boxes, sel) if s]
        raw_scores = [b["detection_score"] for b in chosen]
        write_config(os.path.join(args.out, f"{name}.json"), chosen, raw_scores, val_tokens,
                     "raw", 0.0, uc, ul, False, merge=False)
        write_config(os.path.join(args.out, f"{name}+Rv.json"), chosen, raw_scores, val_tokens,
                     "radar", blend_a, uc, ul, True, merge=False)

    # merged LiDAR + camera, raw scores, with and without radar velocity
    both = np.isin(mod_va, ["lidar", "camera"])
    chosen = [b for b, s in zip(va_boxes, both) if s]
    raw_scores = [b["detection_score"] for b in chosen]
    write_config(os.path.join(args.out, "L+C.json"), chosen, raw_scores, val_tokens,
                 "raw", 0.0, True, True, False, merge=True)
    write_config(os.path.join(args.out, "L+C+Rv.json"), chosen, raw_scores, val_tokens,
                 "radar", blend_a, True, True, True, merge=True)

    # re-scored configs (radar features in the score) plus radar velocity
    plan = [
        ("C+R", ["camera"], True, True, False, True),
        ("L+R", ["lidar"], True, False, True, True),
        ("L+C+R", ["lidar", "camera"], True, True, True, True),
    ]
    for name, mods, use_radar, uc, ul, ur in plan:
        va_sel = np.isin(mod_va, mods)
        sub = [b for b, s in zip(va_boxes, va_sel) if s]
        if args.fit_on == "cv":
            scores = cv_scores(nusc, sub, Xva[va_sel], yva[va_sel], use_radar)
        else:
            tr_sel = np.isin(mod_tr, mods)
            clf, cols = fit_rescorer(Xtr[tr_sel], ytr[tr_sel], use_radar)
            scores = clf.predict_proba(Xva[va_sel][:, cols])[:, 1]
        write_config(os.path.join(args.out, f"{name}.json"), sub, scores, val_tokens,
                     "radar" if use_radar else "raw", blend_a, uc, ul, ur, merge=len(mods) > 1)


if __name__ == "__main__":
    main()
