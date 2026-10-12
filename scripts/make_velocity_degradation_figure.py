"""Figure for REPORT_velocity_degradation.md: velocity error of the LiDAR + camera system
beyond 30 m and at full range, for clean LiDAR, a single-sweep LiDAR and a single-sweep
LiDAR with range-dependent dropout, without radar, with the single refitted radar weight
and with the distance-aware weights.

Usage: python scripts/make_velocity_degradation_figure.py --out "visualizing data/figures/velocity_error_velocity_degradation.png"
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
CONDITIONS = [("Clean LiDAR\n10 sweeps", "results/eval", "results/eval_la"),
              ("Single sweep", "results/eval_sw1", "results/eval_sw1_la"),
              ("Single sweep\n+ range dropout", "results/eval_sw1_far", "results/eval_sw1_far_la")]


def load(tag):
    rows = []
    for label, base, la in CONDITIONS:
        b, l = os.path.join(base, f"summary_{tag}.json"), os.path.join(la, f"summary_{tag}.json")
        if not (os.path.exists(b) and os.path.exists(l)):
            continue
        s, a = json.load(open(b)), json.load(open(l))
        rows.append((label, round(s["L+C"]["mAVE"], 3), round(s["L+C+Rv"]["mAVE"], 3), round(a["L+C+Ra"]["mAVE"], 3)))
    return rows


def panel(ax, rows, title):
    x = np.arange(len(rows))
    w = 0.26
    series = [("LiDAR + camera", GREY, [r[1] for r in rows]),
              ("with radar velocity, single weight", BLUE, [r[2] for r in rows]),
              ("with radar velocity, distance-aware", NAVY, [r[3] for r in rows])]
    top = max(v for _, _, vals in series for v in vals)
    for i, (label, color, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals, w, color=color, label=label, linewidth=0)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.012,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    for i, r in enumerate(rows):
        c1, c2 = 100 * (r[2] - r[1]) / r[1], 100 * (r[3] - r[1]) / r[1]
        ax.text(x[i], top * 1.12, f"{c1:+.0f} % / {c2:+.0f} %", ha="center", va="bottom", fontsize=9, color=INK,
                fontweight="bold" if min(c1, c2) <= -10 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylim(0, top * 1.26)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=26)
    ax.set_ylabel("mean velocity error, m/s", fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizing data/figures/velocity_error_velocity_degradation.png")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=130)
    panel(axes[0], load("min30_max0_all"), "LiDAR + camera system, beyond 30 m")
    panel(axes[1], load("min0_max0_all"), "LiDAR + camera system, full range")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                   handlelength=1.2, columnspacing=1.5)
    fig.text(0.01, 0.01, "914 daytime validation frames, official devkit values over all ten classes, rounded to three "
             "decimals as in the tables. Labels: change with the single weight / with the distance-aware weights.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
