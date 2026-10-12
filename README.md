# Radar late fusion on nuScenes: an inference-only measurement

This repository measures what automotive radar adds to camera and LiDAR plus camera 3D object
detection on nuScenes. It is a measurement, not a trained model: two public pretrained detectors
(CenterPoint for LiDAR, FCOS3D for camera, both MMDetection3D checkpoints) are run unchanged, radar
returns are associated to their boxes by geometry, and the fused results are scored with the official
nuScenes devkit. No network is trained or fine-tuned. The only fitted quantities are a radar velocity
blend weight per distance band and a small logistic re-ranker, both fitted outside the evaluated frames.

## Reports

| Report | Scope | File |
| --- | --- | --- |
| Radar Contribution to LiDAR-Camera Fusion | daytime scenes, nuScenes blob 01, 10 October 2026 | [visualizing data/REPORT.md](visualizing%20data/REPORT.md), `REPORT.html` |
| Radar Contribution to LiDAR-Camera Fusion, Day and Night | daytime plus the night and rain scenes of blob 10, 11 October 2026 | [visualizing data/REPORT_enhanced.md](visualizing%20data/REPORT_enhanced.md), `REPORT_enhanced.html` |
| Radar Contribution to LiDAR-Camera Fusion with Degraded LiDAR | the daytime scenes with a random 25 percent of the LiDAR points kept, 12 October 2026 | [visualizing data/REPORT_degradation.md](visualizing%20data/REPORT_degradation.md), `REPORT_degradation.html` |
| Radar Contribution When the LiDAR Cannot Measure Velocity | the daytime scenes with the LiDAR detector run on a single sweep, and on a single sweep with range-dependent point dropout, 12 October 2026 | [visualizing data/REPORT_velocity_degradation.md](visualizing%20data/REPORT_velocity_degradation.md), `REPORT_velocity_degradation.html` |

The `visualizing data` folder is self-contained: the four reports, their figures, three rendered scene
animations (two daytime, one rain at night) and a player that replays them from small derived JSON
files. Raw nuScenes data, detector outputs and fused result files are not in this repository.

## Main result

Adding radar velocity to the LiDAR plus camera system leaves NDS and mAP unchanged and lowers the
mean velocity error: by nothing at full range and 4.6 percent beyond 30 m in daytime, by 4.4 percent at
full range and 4.6 to 6.7 percent beyond 30 m at night (averaged over the classes present), and by about
5 percent in the three rain-at-night scenes. The camera-only detector gains far more (25 to 46 percent
lower velocity error). Letting radar re-rank box confidence hurts in every condition with enough data.
Details, limits and per-class tables are in the reports.

## Layout

| Path | Content |
| --- | --- |
| `scripts/common.py` | nuScenes loading, radar points per keyframe, box geometry |
| `scripts/fuse.py` | late fusion: configurations C, L, L+C (raw), C+Rv, L+Rv, L+C+Rv (radar velocity), C+R, L+R, L+C+R (radar re-scoring) |
| `scripts/evaluate.py` | official devkit evaluation on the frames present locally, with `--min-range`, `--max-range` and `--scene-keyword rain|night` slices; saves per-class AP and errors |
| `scripts/slice_metrics.py` | averages a slice's metrics over the classes that have ground truth in it (the devkit scores an absent class as AP 0 and error 1.0) |
| `scripts/blend_by_distance.py` | distance-aware radar velocity weight, fitted on training frames |
| `scripts/analyze_lc_radar.py` | breakdown of the radar effect by distance, speed, box origin and class |
| `scripts/merge_detections.py` | union of detector outputs from several blobs |
| `scripts/export_viz_data.py`, `scripts/make_report_figures.py`, `scripts/make_enhanced_figure.py`, `scripts/make_report_html.py` | animation data, figures and HTML reports |
| `colab/nuscenes_inference.ipynb` | detector inference on a Google Colab GPU; writes nuScenes-format detection JSONs |
| `results/*.json`, `results/eval*/` | evaluation summaries and breakdowns (small); fused result files are excluded |
| `visualizing data/` | reports, figures, animations and the player |

## Reproducing

### 1. Data

nuScenes v1.0 trainval is available from https://www.nuscenes.org/nuscenes#download after
registration, under the nuScenes terms of use (CC BY-NC-SA 4.0, non-commercial). Needed:
`v1.0-trainval_meta.tgz` (0.4 GB) and one or more of the ten `v1.0-trainvalNN_blobs.tgz` archives
(30 to 42 GB each). Blob 01 holds daytime scenes; blob 10 holds the night and rain validation scenes used
here. Extract the metadata and the blobs into one folder, `data/nuscenes/`, which then contains
`v1.0-trainval/*.json`, `samples/` and `sweeps/`. The download links are personal to the registered
account and are not distributed with this repository.

### 2. Detector inference on Colab

Open `colab/nuscenes_inference.ipynb` in Google Colab with a GPU runtime (a T4 is enough; a session of
about 3 hours per blob). In the settings cell, paste the metadata and blob download links obtained from
the nuScenes website while logged in, or point the fields at copies on Google Drive. Running all cells
installs MMDetection3D in a Python 3.11 environment, streams and extracts the data, builds the dataset
index for the scenes present, downloads the two public checkpoints, runs both detectors over the
validation and training keyframes present, and offers `detections.zip` for download. Unzip it into
`detections/`. For several blobs, run the notebook once per blob and merge the outputs with
`scripts/merge_detections.py`.

### 3. Fusion and evaluation (CPU, Python 3.9)

```
pip install nuscenes-devkit numpy scikit-learn matplotlib imageio-ffmpeg markdown
python scripts/fuse.py --dataroot data/nuscenes --lidar detections/centerpoint.json --camera detections/fcos3d.json --out results/fused --fit-on cv
python scripts/evaluate.py --dataroot data/nuscenes --results results/fused --out results/eval
python scripts/evaluate.py --dataroot data/nuscenes --results results/fused --out results/eval --min-range 30
python scripts/evaluate.py --dataroot data/nuscenes --results results/fused --out results/eval --scene-keyword night
python scripts/slice_metrics.py --summary results/eval/summary_min0_max0_night.json --dataroot data/nuscenes --scene-keyword night
```

Each devkit evaluation loads the nuScenes metadata (2 to 3 minutes, 3 to 5 GB of RAM) and needs up to
8 GB for 1,500 frames; on a 16 GB machine run one at a time.

## Licences and attribution

nuScenes (Motional) is used under CC BY-NC-SA 4.0 for non-commercial research. Detector checkpoints
are from MMDetection3D (OpenMMLab, Apache 2.0): CenterPoint (Yin, Zhou and Krähenbühl, CVPR 2021) and
FCOS3D (Wang et al., ICCV Workshops 2021). The code in `scripts/` and `visualizing data/` is the
author's own.
