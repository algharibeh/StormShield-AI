"""Figure for REPORT_degradation.md: velocity error of the LiDAR + camera system with clean
LiDAR and with 25 percent of the LiDAR points, without radar, with the fixed radar weight
and with the distance-aware weight, at full range and beyond 30 m.

Usage: python scripts/make_degradation_figure.py --out "visualizing data/figures/velocity_error_degradation.png"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY = "#bdbdbd"
BLUE = "#2f6fdb"
NAVY = "#1b3f8a"
INK = "#333333"
MUTED = "#777777"


def load():
    rows = []
    for label, folder, la in (("Clean LiDAR", "results/eval", "results/eval_la"),
                              ("25 % of LiDAR points", "results/eval_keep25", "results/eval_keep25_la")):
        for tag, rng in (("min0_max0_all", "full range"), ("min30_max0_all", "beyond 30 m")):
            s = json.load(open(os.path.join(folder, f"summary_{tag}.json")))
            a = json.load(open(os.path.join(la, f"summary_{tag}.json")))
            # rounded to the three decimals of the tables so the labels match the report text
            rows.append((f"{label},\n{rng}", round(s["L+C"]["mAVE"], 3), round(s["L+C+Rv"]["mAVE"], 3),
                         round(a["L+C+Ra"]["mAVE"], 3)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizing data/figures/velocity_error_degradation.png")
    args = ap.parse_args()
    rows = load()
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    series = [("LiDAR + camera", GREY, [r[1] for r in rows]),
              ("with radar velocity, weight 0.1", BLUE, [r[2] for r in rows]),
              ("with radar velocity, distance-aware", NAVY, [r[3] for r in rows])]
    top = max(v for _, _, vals in series for v in vals)
    for i, (label, color, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals, w, color=color, label=label, linewidth=0)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.012,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    for i, r in enumerate(rows):
        ch_fixed = 100 * (r[2] - r[1]) / r[1]
        ch_la = 100 * (r[3] - r[1]) / r[1]
        ax.text(x[i], top * 1.12, f"{ch_fixed:+.1f} % / {ch_la:+.1f} %", ha="center", va="bottom",
                fontsize=8.5, color=INK, fontweight="bold" if min(ch_fixed, ch_la) <= -4 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylim(0, top * 1.26)
    ax.set_ylabel("mean velocity error, m/s", fontsize=9, color=MUTED)
    ax.set_title("LiDAR + camera system, 914 daytime frames: change in velocity error with radar "
                 "(fixed weight / distance-aware)", fontsize=10, color=INK, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              handlelength=1.2, columnspacing=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Official devkit values over all ten classes, rounded to three decimals as in the tables.\n"
             "The full-range error is lower with degraded LiDAR because the matched set shrinks (17,354 against "
             "18,403 true positives), not because velocities improve.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
