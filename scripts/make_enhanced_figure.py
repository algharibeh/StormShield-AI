"""Figure for REPORT_enhanced.md: velocity error by condition, with and without radar.

Two panels (fused LiDAR + camera system; camera alone) because the two systems
live on different scales and a shared axis would flatten the fused panel.
Night values are the present-class averages (results/eval_b1b10/present_*.json);
day values are the official devkit numbers from the first report.

Usage: python scripts/make_enhanced_figure.py --out "visualizing data/figures/velocity_error_by_condition.png"
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
INK = "#333333"
MUTED = "#777777"


def load_rows():
    night = json.load(open("results/eval_b1b10/present_min0_max0_night.json"))["rows"]
    night30 = json.load(open("results/eval_b1b10/present_min30_max0_night.json"))["rows"]
    rain = json.load(open("results/eval_b1b10/summary_min0_max0_rain.json"))
    day = json.load(open("results/eval/summary_min0_max0_all.json"))
    day30 = json.load(open("results/eval/summary_min30_max0_all.json"))
    conds = ["Day, full", "Day, beyond 30 m", "Night, full", "Night, beyond 30 m", "Rain at night, full"]

    def pair(src, base, radar, key):
        return src[base][key], src[radar][key]

    fused = [pair(day, "L+C", "L+C+Rv", "mAVE"), pair(day30, "L+C", "L+C+Rv", "mAVE"),
             pair(night, "L+C", "L+C+Rv", "mAVE_present"), pair(night30, "L+C", "L+C+Rv", "mAVE_present"),
             pair(rain, "L+C", "L+C+Rv", "mAVE")]
    cam = [pair(day, "C", "C+Rv", "mAVE"), pair(day30, "C", "C+Rv", "mAVE"),
           pair(night, "C", "C+Rv", "mAVE_present"), pair(night30, "C", "C+Rv", "mAVE_present"),
           pair(rain, "C", "C+Rv", "mAVE")]
    return conds, fused, cam


def panel(ax, conds, rows, title, base_label):
    x = np.arange(len(conds))
    w = 0.36
    base = [r[0] for r in rows]
    radar = [r[1] for r in rows]
    b1 = ax.bar(x - w / 2 - 0.01, base, w, color=GREY, label=base_label, linewidth=0)
    b2 = ax.bar(x + w / 2 + 0.01, radar, w, color=BLUE, label="with radar velocity", linewidth=0)
    top = max(base + radar)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    for i, (b, r) in enumerate(rows):
        ch = 100 * (r - b) / b
        label = f"{ch:+.0f} %" if abs(ch) >= 0.5 else "0 %"
        ax.text(x[i], top * 1.14, label, ha="center", va="bottom", fontsize=9, color=INK,
                fontweight="bold" if ch <= -4 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(", ", "\n") for c in conds], fontsize=8.5, color=INK)
    ax.set_ylim(0, top * 1.28)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.5)
    ax.set_ylabel("mean velocity error, m/s", fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizing data/figures/velocity_error_by_condition.png")
    args = ap.parse_args()
    conds, fused, cam = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=130)
    panel(axes[0], conds, fused, "LiDAR + camera system", "LiDAR + camera")
    panel(axes[1], conds, cam, "Camera alone", "camera")
    fig.text(0.01, 0.01, "Night bars: mean over the classes present in the night scenes. Day and rain bars: official devkit "
             "values over all ten classes. Rain at night: 120 frames.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
