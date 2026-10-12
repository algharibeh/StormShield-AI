"""Render the report figures as PNG for the Markdown report.

Usage: python scripts/make_report_figures.py --analysis results/analysis_lc_radar.json --out "visualizing data/figures"
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


def by_distance(analysis, out):
    rows = [r for r in analysis["by_distance"] if r["n"]]
    labels = [r["name"] for r in rows]
    base = [r["err_base"] for r in rows]
    radar = [r["err_radar"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    b1 = ax.bar(x - w / 2, base, w, color=GREY, label="LiDAR + camera")
    b2 = ax.bar(x + w / 2, radar, w, color=BLUE, label="with radar")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8.5)
    for i, r in enumerate(rows):
        ch = round(r["change_pct"])
        ax.text(i, -0.075, f"{ch:+d}%, n = {r['n']:,}", ha="center", va="top", fontsize=8.5,
                color=BLUE if ch < 0 else "#666666", transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean velocity error of matched true positives, m/s")
    ax.set_ylim(0, max(base + radar) * 1.18)
    ax.set_title("Radar lowers velocity error only beyond 30 m: 3 to 4 percent there, nothing under 20 m", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.2)
    path = os.path.join(out, "velocity_error_by_distance.png")
    fig.savefig(path); plt.close(fig)
    return path


def by_speed(analysis, out):
    rows = [r for r in analysis["by_speed"] if r["n"]] + [analysis["moving_far"], analysis["by_origin"][1]]
    labels = ["parked\n< 0.5 m/s", "slow\n0.5 to 3 m/s", "moving\n> 3 m/s", "moving\nbeyond 30 m", "camera-origin\nboxes"]
    base = [r["err_base"] for r in rows]; radar = [r["err_radar"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    b1 = ax.bar(x - w / 2, base, w, color=GREY, label="LiDAR + camera")
    b2 = ax.bar(x + w / 2, radar, w, color=BLUE, label="with radar")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)
    for i, r in enumerate(rows):
        ch = round(r["change_pct"])
        ax.text(i, -0.16, f"{ch:+d}%, n = {r['n']:,}", ha="center", va="top", fontsize=8.5,
                color=BLUE if ch < 0 else "#666666", transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean velocity error, m/s")
    ax.set_ylim(0, max(base + radar) * 1.18)
    ax.set_title("The gain sits on moving objects and on the boxes the LiDAR did not see", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.26)
    path = os.path.join(out, "velocity_error_by_speed.png")
    fig.savefig(path); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis_lc_radar.json")
    ap.add_argument("--out", default="visualizing data/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    a = json.load(open(args.analysis))
    print(by_distance(a, args.out)); print(by_speed(a, args.out))


if __name__ == "__main__":
    main()
