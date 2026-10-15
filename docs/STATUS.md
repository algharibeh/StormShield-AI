# StormShield-AI — status against the project plan (October 2026)

Part of the StormShield-AI project. This page sets what exists in this
repository against the project plan as written: four phases and four key
deliverables. The plan's wording is quoted; the state is measured from the
committed files named in each row. Nothing here is a forecast.

Project plan, goal: "a self-healing perception framework that enables
autonomous agricultural machinery to maintain safe and reliable operation in
adverse environmental and adversarial conditions ... dust, rain, fog, low
visibility, and GNSS-denied environments ... builds upon doctoral research on
historical frame fusion, extending it into a deployable system that integrates
multi-sensor data, AI-based memory models, and probabilistic trust scoring."

## What exists today, in one paragraph

An inference-only measurement of what automotive radar adds to LiDAR plus
camera 3D object detection on the public nuScenes benchmark, in four
experiments: daytime; night and rain; LiDAR thinned to a quarter of its points;
LiDAR deprived of its own velocity estimate, then also of its far-range
coverage. Two public pretrained detectors run unchanged; the only fitted
quantities are a radar velocity weight per distance band and a small
re-ranker. Radar never changed detection accuracy; it lowered far-range
velocity error by about 5 percent with a healthy LiDAR and by a third to two
fifths when the LiDAR could not measure velocity, with the fitted radar weight
rising from 0.1 to 0.8–1.0 as the LiDAR degraded. That rising weight is the
first measured instance of condition-dependent sensor trust, the mechanism the
plan calls probabilistic trust scoring. Everything else in the plan has not
started.

## Phases

| Phase (plan wording) | State | Evidence in this repository |
| --- | --- | --- |
| **Phase 1, months 0–6.** "Collect and curate multi-sensor datasets under harsh environmental conditions, including dust, fog, and low-visibility scenarios." | **Collection not started.** Groundwork on public data: the evaluation pipeline exists and has been run on night, rain and three LiDAR-degradation conditions as proxies for degraded sensing. No data of the author's own, no dust or fog scenes, no agricultural scenes. nuScenes is used under its non-commercial licence and is not redistributed. | `scripts/` (association, fusion, devkit evaluation); `results/eval_*` summaries; the five reports in `visualizing data/` |
| **Phase 2, months 7–15.** "Advanced AI models incorporating historical frame fusion and sensor trust scoring ... optimized for embedded deployment." | **First step only.** Sensor trust: a radar velocity weight fitted per distance band from data, 0.1 with a healthy LiDAR and 0.8–1.0 when its velocity estimate degrades; a logistic re-ranker that helped only where the LiDAR was nearly blind at range. Historical-frame fusion, AI memory models and edge optimisation: not started. No model has been trained; no embedded target has been touched. | `README.md` main-result table; `visualizing data/REPORT_velocity_degradation.md`; `results/eval_sw1_*`, `results/eval_la`, `results/eval_b1b10_la` |
| **Phase 3, months 16–21.** "Integrated with vehicle control systems and cybersecurity mechanisms ... secure decision pathways and fallback mechanisms." | **Not started.** | — |
| **Phase 4, months 22–36.** "Field trials ... licensing strategies and dissemination pathways." | **Not started.** | — |

## Key deliverables

| Deliverable (plan wording) | State | Evidence |
| --- | --- | --- |
| "A deployable embedded perception module capable of operating in harsh and adversarial environments" | **Not started.** A design statement exists (decision-level fusion with condition-dependent sensor trust); no module, no embedded code. | `README.md`, first section |
| "Curated multi-sensor datasets for training and validation" | **Not started.** Public nuScenes slices were selected and scored (914 daytime and 602 night keyframes, 120 of them in rain) and three synthetic LiDAR degradations were defined; none of this is a dataset of the author's own and none is redistributed. | `visualizing data/REPORT_enhanced.md`, `REPORT_degradation.md`, `REPORT_velocity_degradation.md` |
| "An integrated safety- and cybersecurity-compatible perception framework" | **Not started.** The cybersecurity side is pursued separately in the AgriSecure-AI repository; nothing joins the two yet. | — |
| "Training materials and technical resources to support adoption" | **Not started as such.** The five reports, the figures, the three scene animations and the replay player are the technical record of the experiments, written for a reader, but they are not training material. | `visualizing data/` |

## Relation to the doctoral research

The plan says the project "builds upon doctoral research on historical frame
fusion." The doctoral research is the proposal "Principled Late Fusion of
Camera, LiDAR, and Radar for Robust 3D Object Detection" (Oakland University,
July 2026, revised October 2026), kept in a separate private repository; its
first phase is a literature foundation and a benchmark decision, and it was
agreed with the supervisor in October 2026 to proceed paper by paper. The
four experiments here are the pilot measurement that fed that benchmark
decision. Fusion over past frames is a later stage of both the dissertation
and this project and has not started.

## What this repository does not claim

- No agricultural data, no dust or fog, no field trial, no machine.
- No trained model; the detectors are public checkpoints run unchanged.
- No embedded deployment and no timing on any device.
- No users, no adoption, no licence executed.
- Results are measured on a public automotive benchmark because that is where
  co-registered, condition-labelled camera, LiDAR and radar data exists; they
  bear on the design, not on any claim about agricultural machinery.

Last updated October 15, 2026, from the files at this commit.
