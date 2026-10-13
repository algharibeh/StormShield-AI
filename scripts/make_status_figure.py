"""StormShield-AI status figure: the roadmap with what exists so far, and the measured
results of the four fusion experiments in one picture, for the repository README.

Reads the devkit summaries so the result bars are the evaluated numbers, not typed in.

Usage: python scripts/make_status_figure.py --out "visualizing data/figures/stormshield_status.png"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#cde2fb"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984", "#dddcd7"
SURFACE = "#fcfcfb"


def load(path, key, metric="mAVE"):
    return json.load(open(path))[key][metric]


def results():
    """(group, label, before, after) in m/s from the evaluation summaries."""
    day = "results/eval/summary_min0_max0_all.json"
    day30 = "results/eval/summary_min30_max0_all.json"
    night = json.load(open("results/eval_b1b10/present_min0_max0_night.json"))["rows"]
    night30 = json.load(open("results/eval_b1b10/present_min30_max0_night.json"))["rows"]
    rain = "results/eval_b1b10/summary_min0_max0_rain.json"
    rows = [
        ("Camera alone", "Day, full range", load(day, "C"), load(day, "C+Rv")),
        ("Camera alone", "Night, full range", night["C"]["mAVE_present"], night["C+Rv"]["mAVE_present"]),
        ("Camera alone", "Rain at night, full range", load(rain, "C"), load(rain, "C+Rv")),
        ("Camera alone", "Night, beyond 30 m", night30["C"]["mAVE_present"], night30["C+Rv"]["mAVE_present"]),
        ("LiDAR + camera", "Day, beyond 30 m", load(day30, "L+C"), load(day30, "L+C+Rv")),
        ("LiDAR + camera", "Night, beyond 30 m", night30["L+C"]["mAVE_present"], night30["L+C+Rv"]["mAVE_present"]),
        ("LiDAR + camera", "25 % of LiDAR points, beyond 30 m",
         load("results/eval_keep25/summary_min30_max0_all.json", "L+C"),
         load("results/eval_keep25/summary_min30_max0_all.json", "L+C+Rv")),
        ("LiDAR + camera", "Single-sweep LiDAR, beyond 30 m",
         load("results/eval_sw1/summary_min30_max0_all.json", "L+C"),
         load("results/eval_sw1_la/summary_min30_max0_all.json", "L+C+Ra")),
        ("LiDAR + camera", "Single sweep + range dropout, beyond 30 m",
         load("results/eval_sw1_far/summary_min30_max0_all.json", "L+C"),
         load("results/eval_sw1_far_la/summary_min30_max0_all.json", "L+C+Ra")),
    ]
    return rows


PHASES = [
    ("Phase 1", "Data collection and\nenvironmental modeling", "groundwork on public data;\ncollection not started",
     ["nuScenes day, night and rain slices evaluated",
      "three LiDAR degradation models built and run",
      "evaluation pipeline: official devkit, per-range,\nper-class, per-box-origin"]),
    ("Phase 2", "AI model development and\nedge optimization", "first step",
     ["sensor trust weights fitted from data:\n0.1 with a healthy LiDAR, 0.8 to 1.0 when its\nvelocity degrades",
      "historical-frame fusion and edge deployment\nnot started"]),
    ("Phase 3", "Integration with control\nand security systems", "not started", []),
    ("Phase 4", "Field validation and\ndeployment pathways", "not started", []),
]
STATUS_FILL = {"groundwork on public data;\ncollection not started": 0.2, "first step": 0.2, "not started": 0.0}


def draw_roadmap(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (name, title, status, items) in enumerate(PHASES):
        x0, w = i + 0.04, 0.92
        # phase card
        ax.add_patch(FancyBboxPatch((x0, 0.52), w, 0.42, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.2, edgecolor=LINE, facecolor="white"))
        # progress strip inside the card
        ax.add_patch(FancyBboxPatch((x0 + 0.05, 0.555), w - 0.10, 0.045, boxstyle="round,pad=0,rounding_size=0.01",
                                    linewidth=0, facecolor=BLUE_LIGHT))
        frac = STATUS_FILL[status]
        if frac > 0:
            ax.add_patch(FancyBboxPatch((x0 + 0.05, 0.555), (w - 0.10) * frac, 0.045,
                                        boxstyle="round,pad=0,rounding_size=0.01", linewidth=0, facecolor=BLUE))
        ax.text(x0 + 0.05, 0.90, name, fontsize=11, fontweight="bold", color=INK, va="top")
        ax.text(x0 + 0.05, 0.84, title, fontsize=9.5, color=INK2, va="top", linespacing=1.25)
        ax.text(x0 + 0.05, 0.705, status, fontsize=8.5, color=BLUE if frac else MUTED, va="top", ha="left",
                fontweight="bold" if frac else "normal", linespacing=1.2)
        # what exists, under the card
        y = 0.46
        for it in items:
            ax.text(x0 + 0.05, y, "●", fontsize=7, color=BLUE, va="top")
            ax.text(x0 + 0.11, y, it, fontsize=8.3, color=INK2, va="top", linespacing=1.2)
            y -= 0.10 * (1 + it.count("\n")) + 0.03
        if not items:
            ax.text(x0 + 0.05, y, "no work yet", fontsize=8.3, color=MUTED, va="top", style="italic")
        # connector arrow to the next phase
        if i < 3:
            ax.annotate("", xy=(x0 + w + 0.06, 0.75), xytext=(x0 + w + 0.005, 0.75),
                        arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2))


def draw_results(ax, rows):
    groups = ["LiDAR + camera", "Camera alone"]
    colors = {"LiDAR + camera": BLUE, "Camera alone": ORANGE}
    ordered = [r for g in groups for r in rows if r[0] == g]
    ys, labels = [], []
    y = 0
    for g in groups:
        ax.text(-0.01, y + 0.55, g, fontsize=10, fontweight="bold", color=INK, ha="right", va="center",
                transform=ax.get_yaxis_transform())
        y += 1
        for r in ordered:
            if r[0] != g:
                continue
            ys.append(y); labels.append(r[1]); y += 1
        y += 0.4
    top = max(r[2] for r in ordered)
    for yy, r in zip(ys, ordered):
        before, after = r[2], r[3]
        pct = 100 * (after - before) / before
        pct_txt = f"{pct:+.1f} %" if abs(pct) < 10 else f"{pct:+.0f} %"
        ax.barh(yy, before, height=0.34, color=LINE, linewidth=0, zorder=2)
        ax.barh(yy, after, height=0.34, color=colors[r[0]], linewidth=0, zorder=3)
        ax.text(before + top * 0.012, yy, f"{before:.2f} → {after:.2f} m/s   {pct_txt}",
                fontsize=8.5, color=INK, va="center", fontweight="bold" if pct <= -20 else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.55)
    ax.set_xlabel("mean velocity error of true positives, m/s (grey: without radar; colour: with radar velocity)",
                  fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizing data/figures/stormshield_status.png")
    args = ap.parse_args()
    rows = results()
    fig = plt.figure(figsize=(14, 10), dpi=150, facecolor=SURFACE)
    fig.text(0.04, 0.965, "StormShield-AI: where the project stands", fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.933, "Self-healing perception for harsh and adversarial environments. Four fusion experiments on nuScenes are complete;\n"
             "they form the evaluation groundwork for Phase 1, whose real-data collection has not started, and the first sensor-trust result of Phase 2.",
             fontsize=10, color=INK2, va="top", linespacing=1.3)
    # stat tiles
    tiles = [("4", "experiments, each with\nits own report"), ("1,516", "validation keyframes scored\nwith the official devkit"),
             ("3", "LiDAR degradation models:\nthinning, single sweep, range dropout"), ("0.1 → 0.8", "radar velocity weight the fusion\nlearns as LiDAR velocity degrades")]
    for i, (big, small) in enumerate(tiles):
        x = 0.04 + i * 0.235
        fig.text(x, 0.868, big, fontsize=20, fontweight="bold", color=BLUE, va="top")
        fig.text(x, 0.828, small, fontsize=8.5, color=INK2, va="top", linespacing=1.25)
    fig.text(0.04, 0.768, "Roadmap and what exists today", fontsize=12, fontweight="bold", color=INK, va="top")
    ax1 = fig.add_axes([0.03, 0.495, 0.94, 0.26])
    draw_roadmap(ax1)
    fig.text(0.04, 0.47, "Measured so far: velocity error removed by radar velocity fusion", fontsize=12,
             fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.447, "Detection accuracy (mAP) is unchanged in every row, since radar velocity fusion only edits "
             "velocities. The gain grows as the other sensor's own velocity estimate gets worse.",
             fontsize=9, color=INK2, va="top")
    ax2 = fig.add_axes([0.30, 0.065, 0.66, 0.36])
    draw_results(ax2, rows)
    fig.text(0.04, 0.015, "Night rows are averaged over the classes present in the night scenes; other rows are official "
             "devkit values over all ten classes. Single-sweep rows use the distance-aware radar weights.",
             fontsize=7.5, color=MUTED, va="bottom")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
