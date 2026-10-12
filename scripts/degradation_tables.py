"""Tables for the LiDAR-degradation row: clean LiDAR against 25 percent of the points.

Reads the devkit summaries written by evaluate.py for the clean run and the
degraded run and prints Markdown tables (full range and beyond 30 m) with the
change from clean, plus per-class velocity error beyond 30 m where the
summaries carry per-class true-positive errors. Also writes a JSON copy.

Usage:
  python scripts/degradation_tables.py --out results/degradation_keep25.json
"""
import argparse
import json
import os

CLASSES = ["car", "truck", "pedestrian", "motorcycle"]
METRICS = ["NDS", "mAP", "mATE", "mAVE"]
SLICES = [("min0_max0_all", "Full range"), ("min30_max0_all", "Beyond 30 m")]


def load(folder, tag):
    path = os.path.join(folder, f"summary_{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def rows_for(eval_dir, la_dir, a01_dir, tag):
    """Ordered (label, metrics) pairs for one slice from the three evaluation folders."""
    base, la, a01 = load(eval_dir, tag), load(la_dir, tag), load(a01_dir, tag)
    out = []
    if "L" in base:
        out.append(("LiDAR", base["L"]))
    if "L+C" in base:
        out.append(("LiDAR + camera", base["L+C"]))
    if "L+C+Rv" in a01:
        out.append(("+ radar velocity, fixed 0.1", a01["L+C+Rv"]))
    if "L+C+Rv" in base:
        out.append(("+ radar velocity, refitted", base["L+C+Rv"]))
    if "L+C+Ra" in la:
        out.append(("+ radar velocity, distance-aware", la["L+C+Ra"]))
    if "L+C+R" in base:
        out.append(("+ radar re-scoring", base["L+C+R"]))
    return out


def fmt(v, digits=3):
    return f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/eval")
    ap.add_argument("--clean-la", default="results/eval_la")
    ap.add_argument("--degraded", default="results/eval_keep25")
    ap.add_argument("--degraded-la", default="results/eval_keep25_la")
    ap.add_argument("--degraded-a01", default="results/eval_keep25_a01")
    ap.add_argument("--out", default="results/degradation_keep25.json")
    args = ap.parse_args()

    report = {}
    for tag, label in SLICES:
        clean = dict(rows_for(args.clean, args.clean_la, args.clean, tag))
        degraded = rows_for(args.degraded, args.degraded_la, args.degraded_a01, tag)
        print(f"\n### {label}\n")
        print("| System | NDS | mAP | mATE, m | mAVE, m/s | NDS clean | mAP clean | mAVE clean |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        report[tag] = []
        for name, r in degraded:
            # the clean run has one radar-velocity row (its fitted weight was 0.1)
            cname = "+ radar velocity, refitted" if name == "+ radar velocity, fixed 0.1" else name
            c = clean.get(cname, {})
            cells = [name] + [fmt(r[m]) for m in METRICS]
            cells += [fmt(c[m]) if m in c else "" for m in ("NDS", "mAP", "mAVE")]
            print("| " + " | ".join(cells) + " |")
            report[tag].append({"system": name, **{m: r[m] for m in METRICS},
                                "clean": {m: c[m] for m in METRICS if m in c}})
        # per-class velocity error on the far slice
        if tag == "min30_max0_all":
            print(f"\n### Velocity error beyond 30 m by class, m/s (degraded LiDAR)\n")
            print("| System | " + " | ".join(CLASSES) + " |")
            print("| --- |" + " --- |" * len(CLASSES))
            for name, r in degraded:
                tp = r.get("per_class_tp")
                if not tp:
                    continue
                print("| " + name + " | " + " | ".join(fmt(tp[c]["vel_err"]) for c in CLASSES) + " |")
                report.setdefault("per_class_vel_err_min30", []).append(
                    {"system": name, **{c: tp[c]["vel_err"] for c in CLASSES}})
            clean_tp = {name: r.get("per_class_tp") for name, r in clean.items() if r.get("per_class_tp")}
            if clean_tp:
                print(f"\n### Velocity error beyond 30 m by class, m/s (clean LiDAR)\n")
                print("| System | " + " | ".join(CLASSES) + " |")
                print("| --- |" + " --- |" * len(CLASSES))
                for name, tp in clean_tp.items():
                    print("| " + name + " | " + " | ".join(fmt(tp[c]["vel_err"]) for c in CLASSES) + " |")
                    report.setdefault("per_class_vel_err_min30_clean", []).append(
                        {"system": name, **{c: tp[c]["vel_err"] for c in CLASSES}})
    json.dump(report, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
