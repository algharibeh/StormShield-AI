"""Replay a nuScenes scene and watch what radar adds to LiDAR + camera detection.

Left panel: LiDAR + camera boxes with their velocity arrows.
Right panel: the same boxes after radar velocity fusion, with the radar returns
and their Doppler velocity drawn. Ground truth is shown in grey on both sides.
The bottom strip tracks the velocity error of both systems frame by frame.

Needs only numpy and matplotlib. Data files are produced by
scripts/export_viz_data.py and live in ./data.

Usage:
  python play_enhancement.py                      # first scene in ./data, live window
  python play_enhancement.py --scene scene-0103   # a specific scene
  python play_enhancement.py --save demo.mp4      # write a video (ffmpeg) or .gif (pillow)
  python play_enhancement.py --fps 3 --range 60   # slower, wider view
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
COL_GT = "#9a9a9a"
COL_BASE = "#2f6fdb"
COL_RADAR = "#e8762b"
COL_RADAR_PT = "#f0a35c"
COL_EGO = "#222222"


def load_scene(name):
    files = sorted(glob.glob(os.path.join(HERE, "data", "*.json")))
    if not files:
        raise SystemExit("no data files in ./data; run scripts/export_viz_data.py first")
    if name:
        cand = [f for f in files if name in os.path.basename(f)]
        if not cand:
            raise SystemExit(f"scene {name} not found; available: {[os.path.basename(f) for f in files]}")
        files = cand
    with open(files[0]) as f:
        return json.load(f)


def to_ego(xy, ego):
    """Global (N,2) -> ego frame with forward = +y on screen."""
    x0, y0, yaw = ego
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = np.asarray(xy, dtype=float).reshape(-1, 2) - np.array([x0, y0])
    local = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)
    # ego x is forward in nuScenes; put forward on the screen's vertical axis
    return np.stack([-local[:, 1], local[:, 0]], axis=1)


def rot_vec(v, ego):
    yaw = ego[2]
    c, s = np.cos(-yaw), np.sin(-yaw)
    v = np.asarray(v, dtype=float).reshape(-1, 2)
    local = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)
    return np.stack([-local[:, 1], local[:, 0]], axis=1)


def box_corners(b, ego):
    """Footprint corners in screen coordinates."""
    hw, hl = b["w"] / 2, b["l"] / 2
    yaw = b["yaw"]
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]], axis=1) + np.array([b["x"], b["y"]])
    return to_ego(world, ego)


class Player:
    def __init__(self, scene, view_range=50.0, arrow_scale=1.0):
        self.scene = scene
        self.frames = scene["frames"]
        self.R = view_range
        self.k = arrow_scale
        self.err_base, self.err_radar = [], []
        self.err_base_far, self.err_radar_far = [], []

        self.fig = plt.figure(figsize=(14, 8.2), facecolor="white")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[4, 1.25], hspace=0.28, wspace=0.08)
        self.ax_b = self.fig.add_subplot(gs[0, 0])
        self.ax_r = self.fig.add_subplot(gs[0, 1])
        self.ax_e = self.fig.add_subplot(gs[1, :])
        self.fig.suptitle(f"What radar adds to LiDAR + camera   |   nuScenes {scene['scene']}, {scene['location']}",
                          fontsize=14, y=0.995)
        for ax, title in ((self.ax_b, "LiDAR + camera"), (self.ax_r, "LiDAR + camera + radar")):
            ax.set_title(title, fontsize=13, pad=4)
            ax.set_aspect("equal")
            ax.set_xlim(-self.R, self.R); ax.set_ylim(-self.R * 0.6, self.R * 1.4)
            ax.set_xticks([]); ax.set_yticks([])
            for r in (10, 20, 30, 40, 50):
                ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#e4e4e4", lw=0.8, zorder=0))
                ax.text(0.3, r + 0.4, f"{r} m", fontsize=7, color="#b0b0b0")
        self.ax_e.set_xlabel("keyframe"); self.ax_e.set_ylabel("velocity error, m/s")
        self.ax_e.set_xlim(0, len(self.frames) - 1); self.ax_e.grid(alpha=0.25)
        self.line_b, = self.ax_e.plot([], [], color=COL_BASE, lw=2, label="LiDAR + camera")
        self.line_r, = self.ax_e.plot([], [], color=COL_RADAR, lw=2, label="LiDAR + camera + radar")
        self.ax_e.legend(loc="upper right", frameon=False, fontsize=9)
        self.text = self.fig.text(0.5, 0.942, "", ha="center", va="top", fontsize=10.5)
        self.legend_text = self.fig.text(0.5, 0.012,
            "grey: ground truth and true velocity   |   blue: fused boxes and velocity   |   orange: radar returns (Doppler) and radar-corrected velocity\n"
            "green / red outline: radar improved / worsened this box by more than 0.1 m/s   |   thick arrow: moving faster than 3 m/s",
            ha="center", va="bottom", fontsize=8.5, color="#555555")
        self.fig.subplots_adjust(top=0.86, bottom=0.11, left=0.06, right=0.97)
        self.artists = []

    def clear(self):
        for a in self.artists:
            a.remove()
        self.artists = []

    def draw_frame(self, i):
        self.clear()
        fr = self.frames[i]
        ego = fr["ego"]
        for ax in (self.ax_b, self.ax_r):
            ego_poly = Polygon(np.array([[-1, -2.2], [1, -2.2], [1, 2.2], [0, 3.0], [-1, 2.2]]), closed=True, fc=COL_EGO, ec="none", zorder=5)
            ax.add_patch(ego_poly); self.artists.append(ego_poly)
            for g in fr["gt"]:
                pts = box_corners(g, ego)
                p = Polygon(pts, closed=True, fc="none", ec=COL_GT, lw=1.0, zorder=2)
                ax.add_patch(p); self.artists.append(p)
                v = rot_vec(g["v"], ego)[0]
                if np.hypot(*v) > 0.3:
                    cxy = to_ego([[g["x"], g["y"]]], ego)[0]
                    a = FancyArrow(cxy[0], cxy[1], v[0] * self.k, v[1] * self.k, width=0.25, color=COL_GT, alpha=0.9, zorder=2, length_includes_head=True)
                    ax.add_patch(a); self.artists.append(a)

        # radar returns on the right panel
        rp = np.array(fr["radar"], dtype=float).reshape(-1, 5)
        if len(rp):
            pts = to_ego(rp[:, :2], ego)
            sc = self.ax_r.scatter(pts[:, 0], pts[:, 1], s=9, c=COL_RADAR_PT, zorder=3, alpha=0.9)
            self.artists.append(sc)
            vv = rot_vec(rp[:, 2:4], ego)
            moving = np.hypot(vv[:, 0], vv[:, 1]) > 0.8
            for (px, py), (vx, vy) in zip(pts[moving], vv[moving]):
                a = FancyArrow(px, py, vx * self.k * 0.6, vy * self.k * 0.6, width=0.12, color=COL_RADAR_PT, alpha=0.7, zorder=3, length_includes_head=True)
                self.ax_r.add_patch(a); self.artists.append(a)

        eb, er, ebf, erf = [], [], [], []
        n_better = n_worse = 0
        for b in fr["boxes"]:
            pts = box_corners(b, ego)
            cxy = to_ego([[b["x"], b["y"]]], ego)[0]
            dist = float(np.hypot(*cxy))
            delta = (b["err_radar"] - b["err_base"]) if b["err_base"] is not None else 0.0
            for ax, col, vkey in ((self.ax_b, COL_BASE, "v_base"), (self.ax_r, COL_RADAR if b["n_radar"] else COL_BASE, "v_radar")):
                p = Polygon(pts, closed=True, fc=col, ec=col, lw=1.2, alpha=0.18, zorder=4)
                ax.add_patch(p); self.artists.append(p)
                if ax is self.ax_r and abs(delta) > 0.1:
                    oc = "#1a9641" if delta < 0 else "#d7191c"
                    o = Polygon(pts, closed=True, fc="none", ec=oc, lw=2.2, zorder=7)
                    ax.add_patch(o); self.artists.append(o)
                    if delta < 0:
                        n_better += 1
                    else:
                        n_worse += 1
                v = rot_vec(b[vkey], ego)[0]
                speed = float(np.hypot(*v))
                if speed > 0.3:
                    a = FancyArrow(cxy[0], cxy[1], v[0] * self.k, v[1] * self.k, width=0.45 if speed > 3 else 0.25, color=col, zorder=6, length_includes_head=True)
                    ax.add_patch(a); self.artists.append(a)
            if b["err_base"] is not None:
                eb.append(b["err_base"]); er.append(b["err_radar"])
                if dist >= 30:
                    ebf.append(b["err_base"]); erf.append(b["err_radar"])

        self.err_base.append(np.mean(eb) if eb else np.nan)
        self.err_radar.append(np.mean(er) if er else np.nan)
        self.err_base_far.append(np.mean(ebf) if ebf else np.nan)
        self.err_radar_far.append(np.mean(erf) if erf else np.nan)
        xs = np.arange(len(self.err_base))
        self.line_b.set_data(xs, self.err_base); self.line_r.set_data(xs, self.err_radar)
        ymax = np.nanmax([np.nanmax(self.err_base), np.nanmax(self.err_radar), 0.5])
        self.ax_e.set_ylim(0, ymax * 1.15)

        cum_b = np.nanmean(self.err_base); cum_r = np.nanmean(self.err_radar)
        cum_bf = np.nanmean(self.err_base_far) if np.any(~np.isnan(self.err_base_far)) else np.nan
        cum_rf = np.nanmean(self.err_radar_far) if np.any(~np.isnan(self.err_radar_far)) else np.nan
        n_radar = sum(1 for b in fr["boxes"] if b["n_radar"])
        chg = (cum_r - cum_b) / cum_b * 100 if cum_b else 0
        chg_f = (cum_rf - cum_bf) / cum_bf * 100 if cum_bf else float("nan")
        far_txt = f"   |   beyond 30 m: {cum_bf:.2f} -> {cum_rf:.2f} m/s ({chg_f:+.0f}%)" if not np.isnan(cum_bf) else ""
        self.text.set_text(
            f"frame {i + 1}/{len(self.frames)}   |   {len(fr['boxes'])} fused boxes, {n_radar} with radar returns, {len(rp)} radar points   |   "
            f"this frame: {n_better} boxes improved, {n_worse} worsened\n"
            f"running velocity error: {cum_b:.2f} -> {cum_r:.2f} m/s ({chg:+.0f}%)" + far_txt)
        return self.artists + [self.line_b, self.line_r, self.text]

    def reset(self):
        self.err_base, self.err_radar = [], []
        self.err_base_far, self.err_radar_far = [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--range", type=float, default=50.0, help="half-width of the view in metres")
    ap.add_argument("--arrow-scale", type=float, default=1.0, help="metres of arrow per m/s")
    ap.add_argument("--save", default=None, help="write to .mp4 (needs ffmpeg) or .gif instead of showing")
    ap.add_argument("--loop", action="store_true", help="loop the live window")
    ap.add_argument("--dpi", type=int, default=110, help="resolution of a saved video or GIF")
    args = ap.parse_args()

    if args.save:
        matplotlib.use("Agg")
    scene = load_scene(args.scene)
    pl = Player(scene, view_range=args.range, arrow_scale=args.arrow_scale)

    def update(i):
        if i == 0:
            pl.reset()
        return pl.draw_frame(i)

    anim = FuncAnimation(pl.fig, update, frames=len(scene["frames"]), interval=1000 / args.fps, blit=False, repeat=args.loop)
    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".mp4":
            try:  # bundled ffmpeg from the imageio-ffmpeg package, if installed
                import imageio_ffmpeg
                matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass  # falls back to an ffmpeg on PATH
            writer = matplotlib.animation.FFMpegWriter(fps=args.fps, codec="libx264",
                                                       extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                                                   "-pix_fmt", "yuv420p", "-crf", "20"])  # H.264 needs even sizes
        else:
            writer = "pillow"
        anim.save(args.save, writer=writer, dpi=args.dpi)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
