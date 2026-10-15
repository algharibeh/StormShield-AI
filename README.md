# StormShield-AI: radar late fusion on nuScenes, an inference-only measurement

StormShield-AI is intended for autonomous machines that must keep operating safely where sensors
degrade: dust, rain, fog, low visibility and, in later stages, GNSS-denied or compromised settings. It
is designed around decision-level fusion of camera, LiDAR and radar with condition-dependent sensor
trust scoring, so that the machine can judge how far each sensor can be trusted when one is degraded
and lean on the others. Fusion over past frames, memory models and edge deployment are later stages of
the plan and have not been started.

This repository holds the project's starting point: an inference-only measurement of what automotive
radar adds to camera and LiDAR plus camera 3D object detection on nuScenes. Two public pretrained
detectors (CenterPoint for LiDAR, FCOS3D for camera, both MMDetection3D checkpoints) are run
unchanged, radar returns are associated to their boxes by geometry, and the fused results are scored
with the official nuScenes devkit. The only fitted quantities are a radar velocity blend weight per
distance band and a small logistic re-ranker, both fitted outside the evaluated frames. Across four
experiments, by day, at night, in rain and with the LiDAR degraded in three ways, these measurements
form the evaluation groundwork for the project's first phase, which is the collection of real
multi-sensor data in degraded conditions and has not started, and its first sensor-trust result: the share
of radar velocity the fusion learns to take rises from 0.1 with a healthy LiDAR to 0.8 to 1.0 once the
LiDAR's own velocity estimate degrades.

## Where the project stands

![StormShield-AI status: roadmap with what exists today, and the velocity error removed by radar velocity fusion in every measured condition](visualizing%20data/figures/stormshield_status.png)

![Animated: the velocity error of each measured condition drops from its value without radar to its value with radar velocity](visualizing%20data/figures/stormshield_results.gif)

Rebuild both from the evaluation summaries with `python scripts/make_status_figure.py` and
`python scripts/make_status_animation.py`.

The written version, phase by phase and deliverable by deliverable against the project plan's own
wording, with the file that evidences each row and a list of what is not claimed, is
[docs/STATUS.md](docs/STATUS.md).

## Reports

| Report | Scope | File |
| --- | --- | --- |
| Radar Contribution to LiDAR-Camera Fusion | daytime scenes, nuScenes blob 01 | [visualizing data/REPORT.md](visualizing%20data/REPORT.md), `REPORT.html` |
| Radar Contribution to LiDAR-Camera Fusion, Day and Night | daytime plus the night and rain scenes of blob 10 | [visualizing data/REPORT_enhanced.md](visualizing%20data/REPORT_enhanced.md), `REPORT_enhanced.html` |
| Radar Contribution to LiDAR-Camera Fusion with Degraded LiDAR | the daytime scenes with a random 25 percent of the LiDAR points kept | [visualizing data/REPORT_degradation.md](visualizing%20data/REPORT_degradation.md), `REPORT_degradation.html` |
| Radar Contribution When the LiDAR Cannot Measure Velocity | the daytime scenes with the LiDAR detector run on a single sweep, and on a single sweep with range-dependent point dropout | [visualizing data/REPORT_velocity_degradation.md](visualizing%20data/REPORT_velocity_degradation.md), `REPORT_velocity_degradation.html` |
| Radar in LiDAR-Camera Late Fusion: The Sequence of Experiments | overview of the four experiments as a sequence, with the logic that led from each to the next | [visualizing data/REPORT_experiments.md](visualizing%20data/REPORT_experiments.md), `REPORT_experiments.html` |

The `visualizing data` folder is self-contained: the five reports, their figures, three rendered scene
animations (two daytime, one rain at night) and a player that replays them from small derived JSON
files. Raw nuScenes data, detector outputs and fused result files are not in this repository.

## Main result

Radar velocity fusion leaves mAP unchanged in every experiment and lowers the mean velocity error of
the LiDAR plus camera system by an amount set by how well the LiDAR itself measures velocity: little
when it accumulates ten sweeps, a third to two fifths of the far-range error when it cannot.

| Experiment | Data | What was varied | Radar's effect on LiDAR + camera velocity error beyond 30 m |
| --- | --- | --- | --- |
| 1. Daytime baseline | blob 01, 914 validation keyframes, 23 daytime scenes | nothing; camera, LiDAR and fused systems with and without radar | 0.581 to 0.554 m/s, -5 % |
| 2. Night and rain | blob 10, 602 night keyframes, 120 of them in rain | the conditions | 0.760 to 0.725 m/s on the classes present, -5 %; -7 % with distance-aware weights; in rain 0.866 to 0.800 m/s at full range, -8 % |
| 3. Sparser LiDAR | blob 01, a random 25 % of the LiDAR points | the point density | 0.626 to 0.604 m/s, -4 %; -5 % distance-aware |
| 4. LiDAR without velocity | blob 01, LiDAR detector on one sweep instead of ten; then also range-dependent dropout | the LiDAR's own velocity estimate; then its far-range coverage | 1.354 to 0.867 m/s, -36 %; with dropout 1.329 to 0.790 m/s, -41 % |

The camera-only detector gains far more in every condition, 25 to 53 percent of its velocity error
depending on condition and range. Letting radar re-rank box confidence costs accuracy wherever the
LiDAR's own confidence is informative, and helped only where the LiDAR was nearly blind at range.
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
| `scripts/degradation_tables.py`, `scripts/velocity_degradation_tables.py` | comparison tables for the degraded-LiDAR experiments, from the evaluation summaries |
| `scripts/export_viz_data.py`, `scripts/make_report_figures.py`, `scripts/make_enhanced_figure.py`, `scripts/make_degradation_figure.py`, `scripts/make_velocity_degradation_figure.py`, `scripts/make_report_html.py` | animation data, figures and HTML reports |
| `scripts/make_status_figure.py`, `scripts/make_status_animation.py` | the project status figure and the animated results chart shown above |
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
FCOS3D (Wang et al., ICCV Workshops 2021). The code in `scripts/` and `visualizing data/` was written
by the author with the help of an AI coding assistant working under the author's direction; the
experimental design, the decisions and the reports are the author's.
