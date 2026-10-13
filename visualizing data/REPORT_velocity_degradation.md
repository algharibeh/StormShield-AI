# Radar Contribution When the LiDAR Cannot Measure Velocity

*nuScenes, inference-only late fusion on the 914 daytime validation frames of blob 01, with the LiDAR detector run on a single sweep instead of ten, and on a single sweep with range-dependent point dropout. 12 October 2026. This report stands beside [REPORT.md](REPORT.md), the daytime study of 10 October, [REPORT_enhanced.md](REPORT_enhanced.md), the day-and-night study of 11 October, and [REPORT_degradation.md](REPORT_degradation.md), the uniform-thinning study of 12 October; all three are unchanged.*

## Summary

When the LiDAR detector runs on a single sweep instead of ten, its velocity estimate breaks while its detections mostly survive, and radar velocity then recovers a third of the far-range velocity error. The fused LiDAR plus camera system's velocity error rises from 0.516 to 1.266 m/s at full range and from 0.581 to 1.354 m/s beyond 30 m, against a fall in mAP from 0.486 to 0.408. Blending radar velocity into the LiDAR boxes brings the far-range error down to 0.898 m/s with a single refitted weight and 0.867 m/s with distance-aware weights, cuts of 34 and 36 percent, against 5 percent with clean LiDAR; at full range the cuts are 14 and 18 percent against 0 and 1 percent. With a range-dependent point dropout on top, which removes most far detections, the far-range cut reaches 41 percent (1.329 to 0.790 m/s), and a camera-heavier merge plus radar gives the highest far-range NDS of that condition, 0.230 against 0.196 without either. The measurement uses the same 914 daytime validation frames, ground truth, camera detections and radar association as the daytime report, with public pretrained detectors and no training.

- **What was degraded.** The LiDAR detector's velocity comes from point motion across accumulated sweeps. With one past sweep the baseline is a twentieth of a second instead of half a second, and its velocity error more than doubles on every class; radar's Doppler velocity does not depend on accumulation.
- **What radar recovers, single sweep.** Full range 1.266 to 1.084 m/s (single weight) and 1.036 m/s (distance-aware); beyond 30 m 1.354 to 0.898 and 0.867 m/s. Far-range NDS 0.292 to 0.302 and 0.305. mAP unchanged, as radar velocity never adds or removes a box.
- **Per class beyond 30 m, single sweep.** Cars 1.60 to 0.96 m/s, trucks 0.90 to 0.50, buses 3.08 to 0.97, bicycles 1.14 to 0.52 with distance-aware weights. Pedestrians get slightly worse, 0.58 to 0.65 m/s, and motorcycles barely move, 3.14 to 2.86.
- **The fusion knows it.** The LiDAR velocity weight refitted on training frames jumps from 0.1 (clean LiDAR, and 0.1 again with uniform thinning) to 0.8, and the distance-aware weights from 0.05 / 0.05 / 0.10 / 0.30 to 0.40 / 0.65 / 0.90 / 0.95 for 0 to 20, 20 to 30, 30 to 40 and beyond 40 m. With range dropout the two far weights reach 1.0: beyond 30 m the fused velocity is radar's alone.
- **Where it costs.** A single weight of 0.8 over-trusts radar close in: within 20 m and on slow objects the error rises about 10 percent while it falls 26 to 45 percent beyond 30 m. The distance-aware weights remove that cost, which is why they win at full range.
- **Camera-heavier merge.** Letting a camera box stand in for a weak LiDAR box raises the far-range NDS a little more (0.308 against 0.305 single sweep; 0.230 against 0.217 with range dropout) and far mAP under range dropout from 0.057 to 0.072, at a translation-error cost of 0.01 to 0.03 m.
- **Re-scoring.** Radar re-scoring still costs mAP with a single sweep, but with range dropout it helps for the first time: far-range NDS 0.196 to 0.228 and mAP 0.057 to 0.062, neutral at full range. Where the LiDAR is nearly blind at range, radar returns do carry evidence of presence.

![Velocity error of the LiDAR + camera system with and without radar velocity, for clean LiDAR, a single sweep, and a single sweep with range dropout](figures/velocity_error_velocity_degradation.png)

*Mean velocity error of the LiDAR + camera system beyond 30 m (left) and at full range (right), without radar, with the single refitted radar weight and with the distance-aware weights, on clean LiDAR, on a single sweep, and on a single sweep with range-dependent dropout. The labels above each group are the relative change with the single weight and with the distance-aware weights.*

## Setup

Everything except the LiDAR point cloud and one optional merge rule is the daytime report's: the same 914 validation keyframes of blob 01, the same ground truth, the same camera detections, the same radar association and evaluation. Two LiDAR conditions were produced on Colab on 12 October 2026 and fused on this PC.

| Item | Detail |
| --- | --- |
| Data | nuScenes v1.0 trainval, blob 01: 85 daytime scenes, Boston and Singapore, dry; 62 training scenes (2,462 keyframes) used to fit the velocity weights, 23 validation scenes (914 keyframes) for the numbers |
| Single sweep | The LiDAR detector's multi-sweep loader aggregates one past sweep with the keyframe instead of nine. CenterPoint infers velocity from the time offset of the points it sees; one sweep gives it a 0.05 s baseline instead of 0.5 s. This stands for a LiDAR system without sweep accumulation, as under a latency or compute budget. The detector is the published checkpoint, not retrained |
| Single sweep + range dropout | The same, plus a keep probability per point of 1.0 within 20 m, 0.5 at 20 to 30 m, 0.2 at 30 to 40 m and 0.1 beyond 40 m, one seeded draw, applied after sweep aggregation. This thins the cloud the way weather and low reflectivity do, far points first |
| LiDAR detector | CenterPoint with pillar encoder, MMDetection3D checkpoint, inference on Colab (T4, about 10 minutes per condition for 3,376 frames) |
| Camera detector | FCOS3D, MMDetection3D checkpoint, the clean detections of the daytime report |
| LiDAR-primary merge | The daytime rule: a camera box is kept only where no same-class LiDAR box lies within 2.5 m |
| Camera-heavier merge | The same, except that a LiDAR box scoring below 0.3 is replaced by a same-class camera box within 2.5 m that is at least as confident; a replacement, never a duplicate. With clean LiDAR it changes nothing (NDS 0.578 against 0.579) |
| Radar | All five radars, devkit validity filters, ego-motion-compensated radial velocity in the global frame; a return belongs to a box when it falls inside the footprint enlarged by 1 m |
| Using radar | Velocity: a camera-origin box takes the mean radar velocity of its returns; a LiDAR-origin box blends a fraction of radar velocity into its own, either a single weight fitted on training frames or one weight per distance bucket and box origin, both refitted per condition. Re-scoring: a logistic re-ranker over detector score, range, class and radar features, five-fold cross-validated across the validation scenes |
| Metrics | nuScenes Detection Score (NDS), mAP, and the mean true-positive errors for translation (mATE) and velocity (mAVE); all ten classes are present, so the official values are directly comparable with the daytime report |

The single-sweep detector keeps most of its output: 91,707 boxes above score 0.3 against 100,805 clean, car AP 0.805 against 0.864, pedestrian 0.734 against 0.832; motorcycles and bicycles, which need the accumulated points, fall from 0.167 and 0.245 to 0.012 and 0.106. The range dropout removes a further quarter of the confident boxes (74,247) and most far detections: LiDAR-only mAP beyond 30 m falls from 0.257 to 0.036.

## Results

Radar's contribution grows with the LiDAR's velocity loss: 5 percent of the far-range error with clean LiDAR, 34 to 36 percent with a single sweep, 36 to 41 percent with range dropout on top, with mAP untouched in every case. Official devkit values, lower error is better; the mAVE change is against the same condition's system without radar.

| Condition | System | NDS | mAP | mATE, m | mAVE, m/s | mAVE change |
| --- | --- | --- | --- | --- | --- | --- |
| Clean LiDAR, 10 sweeps | LiDAR + camera | 0.399 | 0.257 | 0.520 | 0.581 | |
| Clean LiDAR, 10 sweeps | + radar velocity, single weight | 0.401 | 0.257 | 0.520 | 0.554 | -5 % |
| Clean LiDAR, 10 sweeps | + radar velocity, distance-aware | 0.402 | 0.257 | 0.520 | 0.552 | -5 % |
| Clean LiDAR, 10 sweeps | Camera-heavier merge + radar velocity | 0.402 | 0.259 | 0.526 | 0.561 | -4 % |
| Single sweep | LiDAR | 0.291 | 0.183 | 0.515 | 1.377 | |
| Single sweep | LiDAR + camera | 0.292 | 0.187 | 0.542 | 1.354 | |
| Single sweep | + radar velocity, single weight | 0.302 | 0.187 | 0.542 | 0.898 | -34 % |
| Single sweep | + radar velocity, distance-aware | 0.305 | 0.187 | 0.542 | 0.867 | -36 % |
| Single sweep | Camera-heavier merge | 0.294 | 0.188 | 0.555 | 1.311 | |
| Single sweep | Camera-heavier merge + radar velocity | 0.308 | 0.188 | 0.555 | 0.864 | -34 % |
| Single sweep | + radar re-scoring | 0.294 | 0.170 | 0.572 | 0.876 | -35 % |
| Single sweep + range dropout | LiDAR | 0.166 | 0.036 | 0.636 | 1.544 | |
| Single sweep + range dropout | LiDAR + camera | 0.196 | 0.057 | 0.729 | 1.329 | |
| Single sweep + range dropout | + radar velocity, single weight | 0.211 | 0.057 | 0.729 | 0.848 | -36 % |
| Single sweep + range dropout | + radar velocity, distance-aware | 0.217 | 0.057 | 0.729 | 0.790 | -41 % |
| Single sweep + range dropout | Camera-heavier merge | 0.211 | 0.072 | 0.760 | 1.273 | |
| Single sweep + range dropout | Camera-heavier merge + radar velocity | 0.230 | 0.072 | 0.760 | 0.809 | -36 % |
| Single sweep + range dropout | + radar re-scoring | 0.228 | 0.062 | 0.779 | 0.797 | -40 % |

Beyond 30 m. The NDS moves with radar here because velocity error is one of its five terms: 1.0 to 1.3 points with a single sweep and 1.5 to 2.1 with range dropout, against 0.2 with clean LiDAR.

| Condition | System | NDS | mAP | mATE, m | mAVE, m/s | mAVE change |
| --- | --- | --- | --- | --- | --- | --- |
| Clean LiDAR, 10 sweeps | LiDAR + camera | 0.579 | 0.486 | 0.310 | 0.516 | |
| Clean LiDAR, 10 sweeps | + radar velocity, single weight | 0.579 | 0.486 | 0.310 | 0.516 | 0 % |
| Clean LiDAR, 10 sweeps | + radar velocity, distance-aware | 0.580 | 0.486 | 0.310 | 0.509 | -1 % |
| Single sweep | LiDAR | 0.475 | 0.402 | 0.292 | 1.292 | |
| Single sweep | LiDAR + camera | 0.477 | 0.408 | 0.305 | 1.266 | |
| Single sweep | + radar velocity, single weight | 0.477 | 0.408 | 0.305 | 1.084 | -14 % |
| Single sweep | + radar velocity, distance-aware | 0.477 | 0.408 | 0.305 | 1.036 | -18 % |
| Single sweep | Camera-heavier merge + radar velocity | 0.478 | 0.408 | 0.310 | 1.077 | -14 % |
| Single sweep | + radar re-scoring | 0.473 | 0.396 | 0.314 | 1.213 | -4 % |
| Single sweep + range dropout | LiDAR | 0.417 | 0.307 | 0.335 | 1.656 | |
| Single sweep + range dropout | LiDAR + camera | 0.424 | 0.325 | 0.366 | 1.584 | |
| Single sweep + range dropout | + radar velocity, single weight | 0.424 | 0.325 | 0.366 | 1.341 | -15 % |
| Single sweep + range dropout | + radar velocity, distance-aware | 0.424 | 0.325 | 0.366 | 1.298 | -18 % |
| Single sweep + range dropout | Camera-heavier merge + radar velocity | 0.427 | 0.330 | 0.375 | 1.334 | -15 % |
| Single sweep + range dropout | + radar re-scoring | 0.424 | 0.322 | 0.382 | 1.423 | -10 % |

Full range. At full range the NDS does not move with radar velocity, because the full-range error is dominated by the many near objects whose LiDAR velocity is still usable, and the translation and orientation terms are untouched. The camera-only rows are the daytime report's (2.071 to 1.469 m/s at full range, 1.275 to 0.694 beyond 30 m) and do not change, since the camera detections are the clean ones.

## Per class and by box origin

The gain sits on the vehicle classes at range and on moving objects; pedestrians, whose radar returns are few and noisy, get slightly worse, and a single weight of 0.8 costs about 10 percent close in, where the LiDAR's velocity is still usable.

| Condition | System | Car | Truck | Bus | Pedestrian | Motorcycle | Bicycle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean LiDAR | LiDAR + camera | 0.457 | 0.340 | 0.989 | 0.291 | 1.243 | 0.482 |
| Clean LiDAR | + radar velocity, single weight | 0.442 | 0.315 | 0.927 | 0.289 | 1.250 | 0.396 |
| Single sweep | LiDAR + camera | 1.596 | 0.896 | 3.082 | 0.577 | 3.137 | 1.141 |
| Single sweep | + radar velocity, single weight | 0.999 | 0.525 | 1.150 | 0.641 | 2.886 | 0.533 |
| Single sweep | + radar velocity, distance-aware | 0.959 | 0.497 | 0.970 | 0.654 | 2.855 | 0.523 |
| Single sweep | Camera-heavier merge + radar velocity | 0.990 | 0.542 | 1.146 | 0.649 | 2.577 | 0.553 |
| Single sweep + range dropout | LiDAR + camera | 2.022 | 1.079 | 3.150 | 0.906 | 1.382 | 1.467 |
| Single sweep + range dropout | + radar velocity, distance-aware | 1.205 | 0.581 | 0.953 | 0.906 | 1.267 | 0.849 |
| Single sweep + range dropout | Camera-heavier merge + radar velocity | 1.140 | 0.699 | 1.224 | 0.902 | 1.082 | 0.835 |

Velocity error in m/s by class beyond 30 m. With a single sweep, radar removes 40 percent of the far car error, 45 percent on trucks, 69 percent on buses and 54 percent on bicycles; pedestrians rise from 0.58 to 0.65 m/s, and motorcycles, which the single-sweep detector barely finds (AP 0.012), hardly move. The motorcycle numbers in the range-dropout rows rest on a handful of boxes.

The box-origin breakdown, computed on the fused system's true positives at score 0.3 and above with the single refitted weight, shows where the single weight pays and where it costs:

| Slice of the fused true positives | Clean: n, change | Single sweep: n, change | Single sweep + range dropout: n, change |
| --- | --- | --- | --- |
| All | 18,403, -1.4 % | 16,607, -12.1 % | 13,931, -10.6 % |
| Moving, beyond 30 m | 853, -5.2 % | 728, -40.2 % | 393, -41.9 % |
| Moving, over 3 m/s | 2,044, -2.6 % | 1,873, -19.2 % | 1,508, -15.8 % |
| 40 to 50 m | 1,197, -4.1 % | 925, -44.6 % | 314, -44.4 % |
| 30 to 40 m | 2,874, -3.2 % | 2,342, -25.9 % | 1,101, -34.8 % |
| 20 to 30 m | 4,870, -0.8 % | 4,309, -9.6 % | 3,534, -14.1 % |
| 10 to 20 m | 6,515, +1.3 % | 6,135, +10.3 % | 6,098, +9.2 % |
| 0 to 10 m | 2,927, 0.0 % | 2,885, +10.0 % | 2,873, +9.3 % |
| Slow, 0.5 to 3 m/s | 3,740, -1.1 % | 3,186, +10.2 % | 2,338, +4.7 % |
| Parked, under 0.5 m/s | 12,619, +1.1 % | 11,548, -6.6 % | 10,085, -5.3 % |
| LiDAR-origin boxes | 18,289, -0.8 % | 16,388, -11.9 % | 13,321, -8.7 % |
| Camera-origin boxes | 114, -34.0 % | 219, -26.2 % | 610, -31.5 % |

The close-range cost is the single weight's: 0.8 radar velocity on a box at 10 m, where the single-sweep LiDAR still measures velocity to 0.3 to 0.5 m/s, replaces a usable estimate with a radial-only one. The distance-aware weights take 0.40 within 20 m and 0.90 to 1.00 beyond 30 m, and that is why they beat the single weight at full range (1.036 against 1.084 m/s). Under the camera-heavier merge the camera-origin boxes grow from 219 to 1,356 true positives with a single sweep and from 610 to 2,297 with range dropout, and radar cuts their error by 15 and 24 percent; the fused totals barely change because these boxes are a small share of all true positives. The share of true positives with at least one radar return stays at 52 to 53 percent in every condition.

## The refitted weights

The weights the fusion learns on the training frames are a direct reading of how much the LiDAR's own velocity is worth under each condition, and they move from a tenth to nearly all radar as the LiDAR degrades.

| Condition | Single weight | LiDAR-origin weight, 0 to 20 / 20 to 30 / 30 to 40 / beyond 40 m | LiDAR-origin velocity error before the blend, by the same buckets, m/s |
| --- | --- | --- | --- |
| Clean LiDAR, 10 sweeps | 0.1 | 0.05 / 0.05 / 0.10 / 0.30 | 0.20 / 0.23 / 0.35 / 0.41 |
| 25 percent of the points, uniform (earlier report) | 0.1 | 0.05 / 0.10 / 0.20 / 0.45 | 0.19 / 0.24 / 0.40 / 0.48 |
| Single sweep | 0.8 | 0.40 / 0.65 / 0.90 / 0.95 | 0.41 / 0.51 / 0.84 / 1.07 |
| Single sweep + range dropout | 0.8 | 0.40 / 0.75 / 1.00 / 1.00 | 0.42 / 0.55 / 1.16 / 1.43 |

The camera-origin weights stay at 0.80 / 0.80 / 0.90 / 0.95 in every condition, since the camera detections are the same. On the validation frames the distance-aware blend brings the single-sweep LiDAR boxes beyond 40 m from 1.78 to 0.76 m/s and those at 30 to 40 m from 1.55 to 0.97 m/s; with range dropout, from 1.80 to 0.75 and 1.79 to 0.98 m/s. Uniform thinning, by contrast, left the LiDAR's velocity nearly intact and the weights near their clean values, which is why that earlier run found no larger radar contribution: it degraded detection, which radar velocity cannot repair, rather than velocity, which it can.

## Re-scoring

Radar re-scoring, which cost accuracy in every earlier condition, helps for the first time where the LiDAR is nearly blind at range: with range dropout it raises the far-range NDS from 0.196 to 0.228 and the far-range mAP from 0.057 to 0.062, and at full range it is neutral (NDS 0.424 either way, mAP 0.325 to 0.322). With a single sweep and no dropout it still costs 1.2 mAP points at full range and 1.7 beyond 30 m, and with clean LiDAR 1.7 and 2.1. The pattern fits the earlier explanation: the re-ranker only has something to add when the detector's own confidence has stopped being informative, and with 10 percent of the points beyond 40 m, a LiDAR box out there is as likely to be noise as not, so a radar return behind it is real evidence. Its velocity row is also the lowest of the range-dropout condition at 0.797 m/s, because the re-scored file carries the same radar velocity blend and keeps fewer far false positives. This is one condition with a few hundred far objects; it is a reason to try re-scoring on a weak-LiDAR system, not a reversal of the earlier negative result.

## Limits of this measurement

- **The degradation was chosen to hit velocity.** A single sweep is the one change that breaks the LiDAR's velocity while keeping its boxes, and that is what makes radar's gain large here. A LiDAR system that accumulates sweeps keeps its velocity, as the clean rows show. The result says what radar is worth when the LiDAR cannot measure velocity, not that radar improves a healthy LiDAR system.
- **Detector not adapted.** CenterPoint is used as published, with a loader set to one sweep it was not trained for; the mAP it loses (0.486 to 0.408) is partly that mismatch. A detector trained on single sweeps would detect better and might estimate velocity somewhat better, but it would still lack the time baseline.
- **Range dropout is one draw and not weather.** The keep probabilities 1.0 / 0.5 / 0.2 / 0.1 are a stand-in for distance-dependent loss, drawn once, without the clutter weather adds. The far-range rows of that condition rest on a few hundred objects (393 moving objects beyond 30 m), so their percentages carry several points of noise.
- **Radar hurts pedestrians and close, slow objects.** The single weight is fitted over all LiDAR boxes and over-trusts radar close in; the distance-aware weights fix the range part but not the class part. A per-class weight, or a lower weight for pedestrians, is the obvious next step and was not tried.
- **Subset size.** 914 frames from 23 daytime scenes; differences under about 0.01 NDS or 0.01 m/s are within noise.
- **Conventional radar and a late-fusion design.** nuScenes radars report no elevation and a few hundred returns per frame, and about half of the fused true positives have a radar return at all, which caps what any radar velocity fusion can do here. A 4D radar, or a tracker that integrates Doppler over time, would do more.

## Reproducing

The notebook `colab/nuscenes_inference.ipynb` is run on blob 01 with `SKIP_CAMERA = True` and `LIDAR_RUNS = [{"tag": "sw1", "sweeps": 1}, {"tag": "sw1_far", "sweeps": 1, "range_keep": "1.0,0.5,0.2,0.1"}]`; its inference cell sets the multi-sweep loader's sweep count per run and inserts a `RangeDropout` transform after it, and writes `centerpoint_sw1.json` and `centerpoint_sw1_far.json`. Then, from the project folder, for each tag:

```
.venv\Scripts\python.exe scripts\fuse.py --dataroot data\nuscenes --lidar detections\centerpoint_<tag>.json --camera detections\fcos3d.json --out results\fused_<tag> --fit-on cv
.venv\Scripts\python.exe scripts\fuse.py --dataroot data\nuscenes --lidar detections\centerpoint_<tag>.json --camera detections\fcos3d.json --out results\fused_<tag>_weak --fit-on cv --merge weak-lidar
.venv\Scripts\python.exe scripts\blend_by_distance.py --dataroot data\nuscenes --lidar detections\centerpoint_<tag>.json --camera detections\fcos3d.json --base results\fused_<tag>\L+C.json --out results\fused_<tag>_la\L+C+Ra.json --summary results\blend_by_distance_<tag>.json
.venv\Scripts\python.exe scripts\evaluate.py --dataroot data\nuscenes --results results\fused_<tag> --out results\eval_<tag>
.venv\Scripts\python.exe scripts\evaluate.py --dataroot data\nuscenes --results results\fused_<tag> --out results\eval_<tag> --min-range 30
.venv\Scripts\python.exe scripts\analyze_lc_radar.py --dataroot data\nuscenes --base results\fused_<tag>\L+C.json --radar results\fused_<tag>\L+C+Rv.json --lidar results\fused_<tag>\L.json --out results\analysis_lc_radar_<tag>.json
```

The two evaluations are repeated for `results\fused_<tag>_weak` and `results\fused_<tag>_la`, and the analysis for the weak merge. The clean camera-heavier control is `fuse.py` on `detections\centerpoint.json` with `--merge weak-lidar --out results\fused_weak`, evaluated into `results\eval_weak`. `scripts\velocity_degradation_tables.py` prints the tables (`results\velocity_degradation.json`) and `scripts\make_velocity_degradation_figure.py` draws the figure.

Files: `detections/centerpoint_sw1.json` and `detections/centerpoint_sw1_far.json` (3,376 keyframes each, Colab run of 12 October 2026); `results/fused_sw1*/` and `results/fused_sw1_far*/` (fused result files); `results/eval_sw1*/summary_*.json`, `results/eval_sw1_far*/summary_*.json` and `results/eval_weak/summary_*.json` (devkit metrics with per-class AP and errors); `results/blend_by_distance_sw1.json` and `_sw1_far.json` (refitted weights); `results/analysis_lc_radar_sw1*.json` and `_sw1_far*.json` (box-origin breakdowns); `results/velocity_degradation.json` (the tables). The nuScenes data is not redistributed.

## Sources

- nuScenes: Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020. Data used under CC BY-NC-SA 4.0 for non-commercial research.
- CenterPoint: Yin, Zhou and Krähenbühl, *Center-based 3D Object Detection and Tracking*, CVPR 2021. Checkpoint from MMDetection3D.
- FCOS3D: Wang et al., *FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection*, ICCV Workshops 2021. Checkpoint from MMDetection3D.
- MMDetection3D: OpenMMLab, version 1.4.0.
