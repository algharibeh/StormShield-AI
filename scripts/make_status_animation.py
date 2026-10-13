"""Animated results chart for the README: the velocity error of each measured condition starts at
its value without radar and drops to its value with radar velocity, group by group, then holds.
Written as a looping GIF, which GitHub plays inline. Numbers come from the evaluation summaries.

Usage: python scripts/make_status_animation.py --out "visualizing data/figures/stormshield_results.gif"
"""
import argparse
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from make_status_figure import results, BLUE, ORANGE, INK, INK2, MUTED, LINE, SURFACE

FPS = 20
HOLD_START, GROW, HOLD_END = 1.0, 1.6, 3.0   # seconds: grey bars, the drop, the final frame


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def render(rows, progress, size=(1100, 640)):
    """One frame. progress: 0 = all bars at 'without radar', 1 = all at 'with radar'."""
    groups = ["LiDAR + camera", "Camera alone"]
    colors = {"LiDAR + camera": BLUE, "Camera alone": ORANGE}
    ordered = [r for g in groups for r in rows if r[0] == g]
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0.33, 0.12, 0.64, 0.76])
    top = max(r[2] for r in ordered)
    y, ys = 0, []
    n = len(ordered)
    for g in groups:
        ax.text(-0.01, y + 0.55, g, fontsize=11, fontweight="bold", color=INK, ha="right", va="center",
                transform=ax.get_yaxis_transform())
        y += 1
        for i, r in enumerate(ordered):
            if r[0] != g:
                continue
            ys.append(y)
            # staggered start: each bar begins a little after the previous one
            k = len(ys) - 1
            local = ease((progress * (n + 2) - k) / 3.0)
            before, after = r[2], r[3]
            cur = before + (after - before) * local
            pct = 100 * (cur - before) / before
            ax.barh(y, before, height=0.36, color=LINE, linewidth=0, zorder=2)
            ax.barh(y, cur, height=0.36, color=colors[g], linewidth=0, zorder=3)
            label = f"{before:.2f} m/s" if local <= 0 else f"{before:.2f} → {cur:.2f} m/s   {pct:+.0f} %"
            ax.text(before + top * 0.012, y, label, fontsize=9.5, color=INK, va="center",
                    fontweight="bold" if local >= 1 and pct <= -20 else "normal")
            y += 1
        y += 0.5
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in ordered], fontsize=9.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.55)
    ax.set_xlabel("mean velocity error of true positives, m/s   (grey: without radar; colour: with radar velocity)",
                  fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.03, 0.955, "StormShield-AI: velocity error removed by radar velocity fusion, per measured condition",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.905, "Detection accuracy (mAP) is unchanged in every row. The gain grows as the other sensor's "
             "own velocity estimate gets worse.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.03, 0.02, "Night rows: classes present in the night scenes. Other rows: official devkit values over all "
             "ten classes. Single-sweep rows use the distance-aware radar weights.", fontsize=7.5, color=MUTED, va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizing data/figures/stormshield_results.gif")
    args = ap.parse_args()
    rows = results()
    frames, durations = [], []
    frames.append(render(rows, 0.0)); durations.append(int(HOLD_START * 1000))
    n_grow = int(GROW * FPS)
    for i in range(1, n_grow + 1):
        frames.append(render(rows, i / n_grow)); durations.append(int(1000 / FPS))
    durations[-1] = int(HOLD_END * 1000)
    pal = [f.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save(os.path.splitext(args.out)[0] + "_last.png")
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e6:.2f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
