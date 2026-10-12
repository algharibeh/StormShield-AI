"""Tables for the velocity-degradation report: clean LiDAR against a single-sweep LiDAR
(sw1) and a single-sweep LiDAR with range-dependent dropout (sw1_far), under the
LiDAR-primary and the camera-heavier (weak-lidar) merge, with and without radar.

Reads the devkit summaries written by evaluate.py and prints Markdown tables for the
full range and beyond 30 m, the per-class velocity errors beyond 30 m, and a JSON copy.

Usage:
  python scripts/velocity_degradation_tables.py --out results/velocity_degradation.json
"""
import argparse
import json
import os

CLASSES = ["car", "truck", "bus", "pedestrian", "motorcycle", "bicycle"]
SLICES = [("min0_max0_all", "Full range"), ("min30_max0_all", "Beyond 30 m")]
# condition label -> (lidar-primary eval folder, distance-aware eval folder, weak-lidar eval folder)
CONDITIONS = [
    ("Clean LiDAR, 10 sweeps", "results/eval", "results/eval_la", "results/eval_weak"),
    ("Single sweep", "results/eval_sw1", "results/eval_sw1_la", "results/eval_sw1_weak"),
    ("Single sweep + range dropout", "results/eval_sw1_far", "results/eval_sw1_far_la", "results/eval_sw1_far_weak"),
]


def load(folder, tag):
    path = os.path.join(folder, f"summary_{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def systems(base, la, weak):
    out = []
    if "L" in base:
        out.append(("LiDAR", base["L"]))
    if "L+C" in base:
        out.append(("LiDAR + camera", base["L+C"]))
    if "L+C+Rv" in base:
        out.append(("+ radar velocity, single weight", base["L+C+Rv"]))
    if "L+C+Ra" in la:
        out.append(("+ radar velocity, distance-aware", la["L+C+Ra"]))
    if "L+C" in weak:
        out.append(("Camera-heavier merge", weak["L+C"]))
    if "L+C+Rv" in weak:
        out.append(("Camera-heavier merge + radar velocity", weak["L+C+Rv"]))
    if "L+C+R" in base:
        out.append(("+ radar re-scoring", base["L+C+R"]))
    return out


def pct(a, b):
    return f"{100 * (b - a) / a:+.0f} %" if a else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/velocity_degradation.json")
    args = ap.parse_args()
    report = {}
    for tag, label in SLICES:
        print(f"\n### {label}\n")
        print("| Condition | System | NDS | mAP | mATE, m | mAVE, m/s | mAVE change |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        report[tag] = []
        for cond, base_dir, la_dir, weak_dir in CONDITIONS:
            rows = systems(load(base_dir, tag), load(la_dir, tag), load(weak_dir, tag))
            ref = dict(rows).get("LiDAR + camera", {}).get("mAVE")
            ref_weak = dict(rows).get("Camera-heavier merge", {}).get("mAVE")
            for name, r in rows:
                against = ref_weak if name.startswith("Camera-heavier merge + radar") else ref
                change = pct(against, r["mAVE"]) if name.startswith(("+ radar", "Camera-heavier merge + radar")) else ""
                print(f"| {cond} | {name} | {r['NDS']:.3f} | {r['mAP']:.3f} | {r['mATE']:.3f} | {r['mAVE']:.3f} | {change} |")
                report[tag].append({"condition": cond, "system": name, **{m: r[m] for m in ("NDS", "mAP", "mATE", "mAVE")}})
        if tag == "min30_max0_all":
            print(f"\n### Velocity error beyond 30 m by class, m/s\n")
            print("| Condition | System | " + " | ".join(CLASSES) + " |")
            print("| --- | --- |" + " --- |" * len(CLASSES))
            for cond, base_dir, la_dir, weak_dir in CONDITIONS:
                for name, r in systems(load(base_dir, tag), load(la_dir, tag), load(weak_dir, tag)):
                    tp = r.get("per_class_tp")
                    if not tp or name in ("LiDAR", "+ radar re-scoring"):
                        continue
                    print(f"| {cond} | {name} | " + " | ".join(f"{tp[c]['vel_err']:.3f}" for c in CLASSES) + " |")
                    report.setdefault("per_class_vel_err_min30", []).append(
                        {"condition": cond, "system": name, **{c: tp[c]["vel_err"] for c in CLASSES}})
    json.dump(report, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
