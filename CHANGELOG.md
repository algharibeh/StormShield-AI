# Changelog

## 0.1.1 — 2026-10-25

- The attribution paragraph of the README is reduced to the authorship
  statement.
- The daytime report describes its method as inference-only with fitted
  fusion weights; the README and the status page name the companion
  dissertation repository (October 17).
- Printable stills beside the animations: the README shows the final frame of
  the results animation (`stormshield_results_last.png`, written by
  `scripts/make_status_animation.py`), and the two scene reports open with the
  last keyframe of their animation (`demo_scene-<scene>_last.png`, written by
  `play_enhancement.py --frame -1 --save`). The GIF and MP4 animations stay and
  are linked next to each still, so the reports print without animated images
  and read the same on GitHub. `scripts/make_report_html.py` swaps a still for
  its MP4 twin in the HTML copies as it did for the GIFs.

## 0.1.0 — 2026-10-15

First release of the StormShield-AI evaluation groundwork: an inference-only
measurement of radar's contribution to LiDAR plus camera 3D object detection
on nuScenes.

- Four experiments, committed October 12, 2026: daytime baseline (blob 01,
  914 keyframes); night and rain (blob 10, 602 keyframes); LiDAR thinned to a
  random 25 percent of points; LiDAR run on a single sweep without a usable
  velocity estimate, then with range-dependent point dropout.
- Scripts for radar association, velocity blending, the logistic re-ranker,
  and nuScenes devkit evaluation (`scripts/`); the Colab inference notebook.
- Five reports with figures, three rendered scene animations and a player
  (`visualizing data/`); the status figure and animation in the README.
- Design statement for StormShield-AI with what has and has not started.
- Added for the release: `LICENSE` (MIT for the author's material; nuScenes
  and MMDetection3D keep their own terms), `CITATION.cff`, this changelog;
  commit author name corrected to "Moath Algharibeh" with dates preserved.

Not started: collection of real multi-sensor data in degraded conditions,
fusion over past frames, memory models, edge deployment.
