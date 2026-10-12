# What radar adds to LiDAR + camera detection: an animated replay

Four write-ups live here. [REPORT.md](REPORT.md) is the daytime study of 10 October 2026 (nuScenes blob 01, scenes 0093 and 0101). [REPORT_enhanced.md](REPORT_enhanced.md) extends it with the night and rain scenes of blob 10 (11 October 2026) and adds the rain-at-night scene 1065 to the player. [REPORT_degradation.md](REPORT_degradation.md) repeats the daytime measurement with a random 25 percent of the LiDAR points kept (12 October 2026). [REPORT_velocity_degradation.md](REPORT_velocity_degradation.md) repeats it with the LiDAR detector run on a single sweep, so that its own velocity estimate degrades, and with range-dependent point dropout on top (12 October 2026). Each has an `.html` twin that plays the MP4 clips offline.

This folder is self-contained. It replays nuScenes driving scenes and shows,
frame by frame, the LiDAR + camera detections next to the same detections
after radar velocity fusion, with the radar returns drawn and a running
velocity-error comparison.

## Run it

```
pip install numpy matplotlib
python play_enhancement.py
```

Options:

| Flag | Meaning |
| --- | --- |
| `--scene scene-0103` | pick a scene from `data/` |
| `--fps 3` | playback speed (keyframes are 0.5 s apart) |
| `--range 60` | half-width of the bird's-eye view in metres |
| `--save demo.mp4` | write a video instead of opening a window (`.mp4` uses the bundled ffmpeg from `imageio-ffmpeg`, `.gif` uses pillow) |
| `--dpi 150` | resolution of the saved file (default 110) |
| `--loop` | loop the live window |

## What you see

- **Left panel:** LiDAR + camera boxes (blue) with their velocity arrows.
- **Right panel:** the same boxes after radar velocity fusion. Boxes that received radar returns turn orange; the radar returns are orange dots with Doppler arrows on moving returns.
- **Grey boxes and arrows:** ground truth and true velocity.
- **Bottom strip:** mean velocity error of matched true positives per frame, both systems.
- **Status line:** running velocity error of both systems and the same restricted to objects beyond 30 m, where the radar effect concentrates.

Thicker arrows are objects moving faster than 3 m/s. The view is ego-centric, forward is up, rings every 10 m.

## Rendered demos

Each bundled scene is rendered twice: `demo_scene-XXXX.mp4` (full resolution, about 1 MB, used by `REPORT.html`) and `demo_scene-XXXX.gif` (about 1.6 MB, used by `REPORT.md` because GitHub plays GIFs inline). Open either in any browser.

## Data

`data/*.json` is derived data, one file per scene, produced by
`scripts/export_viz_data.py` from the project root. Each frame carries the
ego pose, ground-truth boxes with velocity, the fused boxes with velocity
before and after radar, radar returns with compensated velocity, and the
per-box velocity error against ground truth. No images or point clouds are
included, so the folder stays small enough to version in git.

Source: nuScenes v1.0 (Motional), used under CC BY-NC-SA 4.0 for
non-commercial research. Detectors: CenterPoint and FCOS3D checkpoints
from MMDetection3D. Fusion and evaluation code is in the parent project.
