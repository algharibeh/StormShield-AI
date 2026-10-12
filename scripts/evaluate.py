"""Official nuScenes detection evaluation restricted to the samples we have.

Evaluates every JSON in a folder (or one file) and prints a table of
NDS, mAP, mATE, mAVE, mAOE. Optional slices: --min-range (far objects only),
--max-range (near only), --scene-keyword rain|night (scene description).

Usage:
  python scripts/evaluate.py --dataroot data/nuscenes --results results/fused
  python scripts/evaluate.py --dataroot data/nuscenes --results results/fused --min-range 30
  python scripts/evaluate.py --dataroot data/nuscenes --results results/fused --scene-keyword rain
"""
import argparse
import glob
import json
import os
import sys

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval

sys.path.insert(0, os.path.dirname(__file__))
from common import load_nusc, scene_description


class SubsetEval(DetectionEval):
    """DetectionEval over only the sample tokens present in the prediction file."""

    def __init__(self, nusc, config, result_path, eval_set, output_dir, min_range=0.0,
                 scene_keyword=None, verbose=False):
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        pred, self.meta = load_prediction(result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
        gt_all = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        gt_tokens = set(gt_all.sample_tokens)
        tokens = [t for t in pred.sample_tokens if t in gt_tokens]
        if scene_keyword:
            tokens = [t for t in tokens if scene_keyword.lower() in scene_description(nusc, t)]
        self.sample_tokens = tokens
        gt = EvalBoxes()
        pr = EvalBoxes()
        for t in tokens:
            gt.add_boxes(t, gt_all[t])
            pr.add_boxes(t, pred[t])
        pr = add_center_dist(nusc, pr)
        gt = add_center_dist(nusc, gt)
        pr = filter_eval_boxes(nusc, pr, self.cfg.class_range, verbose=verbose)
        gt = filter_eval_boxes(nusc, gt, self.cfg.class_range, verbose=verbose)
        if min_range > 0:
            for boxes in (pr, gt):
                for t in boxes.sample_tokens:
                    boxes.boxes[t] = [b for b in boxes[t] if b.ego_dist >= min_range]
        self.pred_boxes, self.gt_boxes = pr, gt


def run_one(nusc, path, out_dir, min_range, max_range, keyword, eval_set):
    cfg = config_factory("detection_cvpr_2019")
    if max_range:
        cfg.class_range = {k: min(v, max_range) for k, v in cfg.class_range.items()}
    ev = SubsetEval(nusc, cfg, path, eval_set, out_dir, min_range=min_range, scene_keyword=keyword)
    metrics, _ = ev.evaluate()
    s = metrics.serialize()
    return {
        "samples": len(ev.sample_tokens), "NDS": s["nd_score"], "mAP": s["mean_ap"],
        "mATE": s["tp_errors"]["trans_err"], "mAVE": s["tp_errors"]["vel_err"],
        "mAOE": s["tp_errors"]["orient_err"], "per_class_ap": s["mean_dist_aps"],
        "per_class_tp": s["label_tp_errors"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--results", required=True, help="folder of JSONs or one JSON")
    ap.add_argument("--min-range", type=float, default=0.0)
    ap.add_argument("--max-range", type=float, default=0.0)
    ap.add_argument("--scene-keyword", default=None)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    nusc = load_nusc(args.dataroot, args.version)
    eval_set = "mini_val" if args.version.endswith("mini") else "val"
    if os.path.isdir(args.results):
        files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    else:
        files = [args.results]
    tag = f"min{int(args.min_range)}_max{int(args.max_range)}_{args.scene_keyword or 'all'}"
    rows = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        rows[name] = run_one(nusc, f, os.path.join(args.out, tag, name),
                             args.min_range, args.max_range, args.scene_keyword, eval_set)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nslice: {tag}")
    header = ("config", "samples", "NDS", "mAP", "mATE", "mAVE", "mAOE")
    print("{:8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(*header))
    for name, r in rows.items():
        print("{:8} {:7d} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(
            name, r["samples"], r["NDS"], r["mAP"], r["mATE"], r["mAVE"], r["mAOE"]))


if __name__ == "__main__":
    main()
