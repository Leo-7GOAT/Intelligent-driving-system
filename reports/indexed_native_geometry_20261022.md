# Indexed native-road geometry and paired PPO development

Research acceptance remains incomplete. These are development routes and one policy seed, not independent confirmation. Geometry diagnostics, fixed-plan control and PPO outcomes are reported separately.

## Road-query defect and scope of the repair

Six saved footprint points rejected by the legacy CARLA nearest-lane query lie inside a refined union of Driving lanes. The audit minimizes centerline distance near the reported OpenDRIVE coordinate, checks actual lane half-width and height, and changes neither measured pose nor lane width. Local neighbouring-lane version v2 recovers 33/0/23/0 cells in the four shared grids, loses none, and passes 48/48 inside/outside boundary controls.

Actual v2 B2 execution then exposes a junction omission: its nearest road 516 is outside at distance 1.756802 m against half-width 1.75 m, while nearby roads 469/515 contain the same point at about 1.743176 m. Version v3 indexes one-metre Driving-lane samples across road IDs and refines nearby candidates. The index radius discovers candidates; membership still uses actual lane width. It also passes 48/48 boundary controls. Those controls are selected nonjunction cross-sections, not exhaustive map validation.

The default legacy version is unchanged. Explicit v2/v3 signatures separate incompatible road semantics. The selected geometry reaches policy observations, planning, risk and termination, and bridge logs preserve raw nearest-lane rejections alongside revised membership. A sampled grid plus a lane union is not a continuous asphalt mesh or legal-lane guarantee: opposite-direction Driving lanes may be included. Old failures remain historical results under their original version.

Read-only evidence: `20261022T144123Z_lane_union_audit`, including the original module hash, all six points, the failed v2 junction point and both grid audits. Map OpenDRIVE digest: `5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7`. No complete map is distributed.

## Actual fixed-plan CARLA control

All four routes are retained in each batch. Target speed, vehicle, legacy actuator, MPC and goal tolerances are unchanged. H/B2 here designate the planner that creates a fixed reference, not trained high-level policies.

| Batch suffix | Geometry / planner | Completed | Applied controls |
|---|---|---:|---:|
| 145159Z | v2 / Hybrid A* | 4/4 | 1,963 |
| 145407Z | v2 / 2D A* body spline | 3/4 | 1,566 |
| 150241Z | v3 / Hybrid A* | 4/4 | 1,958 |
| 150456Z | v3 / 2D A* body spline | 4/4 | 1,938 |

Full batch names are `20261022T<suffix>_native_curve_control`. All measured frame/state/IMU/timing checks pass. No collision or QP failure was observed in these batches. Severe acceleration minima near -25 m/s² and jerk RMS around 120–141 m/s³ remain. The v3 Hybrid A* last route finishes only just inside the 0.8 m position tolerance. Four-route completion does not establish robust safety, comfort, real-time operation or learning performance. Both v3 figures were visually inspected.

## Completed native PPO experiment

Batch `20261022T150606Z_native_road_learning` uses v3, common 4,447-dimensional observations, unchanged ego-arc actions, seed 11, 45-second tasks and equal nominal 10,000-control-step budgets. Rollout completion accounts for the small actual budget difference. Each method uses a 64×64 actor/critic, 32-action rollouts and two PPO epochs.

| Method | Training controls | Training world ticks | Optimizer steps | Actor L2 change | Frozen endpoint probes |
|---|---:|---:|---:|---:|---|
| H | 10,240 | 10,611 | 64 | 0.589498 | 0/4; four deadlines |
| B1 | 10,192 | 10,718 | 64 | 0.590989 | 0/4; two bounds failures, two deadlines |

Both models reload exactly. Each has one unfinished training episode. Training outcomes are H: 11 deadlines; B1: 11 bounds failures and five deadlines. Evaluation runs on the same four training roads with frozen normalization; it is not held-out performance. Evaluation adds 3,600/2,438 controls and 3,724/2,562 world ticks for H/B1. The analyzer now matches **all 26,470 training and evaluation controls** to raw bridge records and rejects duplicate frames, state changes, timestamp gaps and stale IMU frames.

H evaluation contains 268 invalid endpoint plans. Offline coverage at the four measured starts finds 24/14/26/22 ego-arc actions with analytic witnesses; thus no-path-at-start is not the explanation. At the fourth terminal state none remain. Existing goal-frame proposals do not consistently improve coverage. These eight-state probes cannot establish completeness; a missing analytic witness is not a proof of no Hybrid A* path.

## Learning stability and the next registered comparison

Lightweight v19 B1 completed 5,001,122 physical steps and 42,586 optimizer updates with exact reload. The final complete development task remains 0/10 (six deadlines, four collisions). At 5M, deterministic simple 6/12/18 m probes each pass 8/8, while 23 m passes 0/8; these repeated simple layouts are not independent conditions. Goal-entry speed and later overshoot explain the empty-task failures.

Additional frozen intermediate checkpoints change the next diagnosis: at 4,496,363 and 4,898,060 steps, 23 m probes pass 8/8 in both deterministic and stochastic modes for both splits. Complete ten-case development results are 4/10 and 3/10 respectively; static/unseen collisions persist. By 4,996,029 steps the 23 m probes fall to 0/8 in all modes. The last update's approximate KL is 0.05715 despite the configured 0.03 early-stop target; an early-stop target is not a hard KL bound. This localizes a late regression, but does not yet establish whether weights, normalization or their interaction caused it. Intermediate results are diagnostic additions, not retroactively selected final scores. Final 0/10 and all earlier failures remain.

Batch `20261022T152120Z_native_road_learning` is actually running a new H0/H/B1 comparison, seed 11, 100,000 nominal controls each and a two-hour resource window. H0 zeros the 99 feedback entries; H and B1 receive the same geometric witnesses and previous execution outcomes. All receive the full road grid (4,546 inputs). Actions, rewards, vehicle, controller and PPO settings remain fixed. There is no action masking or rule replacement. Inference checkpoints are saved around 10k/30k/100k; they are not claimed to restore CARLA physical state seamlessly. Current liveness must be checked against OS processes and heartbeat, not this report. Final results will be added only after execution and raw-frame verification.

## Reproduction and publication

CPU regression after analyzer fault-injection coverage: **117 passed, five conditional skips**. CPython 3.7 bridge/lane modules were separately compiled. Skipped optional tests are not counted as live CARLA evidence.

```powershell
python -m pytest -q
python scripts/analyze_native_control.py --batch EXTRACTED_CONTROL_BATCH
python scripts/analyze_native_road_learning.py --batch EXTRACTED_PPO_BATCH
python scripts/audit_native_candidates.py --batch EXTRACTED_PPO_BATCH --output candidate_audit.json
python scripts/native_road_learning.py --prepare SURVEY_BATCH --road-geometry-version carla_driving_indexed_union_64_v3 --controlled-budget 100000 --wall-seconds 7200 --episode-seconds 45 --evaluate --feedback-ablation
# Use the printed batch path and its frozen source, with the documented isolated CARLA client:
python BATCH/source/scripts/native_road_learning.py --batch BATCH
```

Prepared immutable assets, scanned for private paths/credentials and checked against every member hash:

- `indexed_native_geometry_diagnostics.zip`: 883 members, 6,814,464 bytes, SHA256 `0290b78f1ff4a8fafea681c93697ee10dc80bf8f9455feaefc2ee8226bef7e50`. Includes all 16 fixed-plan episodes, the v2 failure, offline audits and frozen source.
- `indexed_native_ppo_development.zip`: 240 members, 16,082,364 bytes, SHA256 `cf353f225bb81dbcf00946f702a39696f76aca7a1a98a239e9fa5344b5cc2864`. Includes both 10k models, normalization, training/evaluation traces and failures. It does not include the still-running 100k comparison or v19 weights.

Upload, independent-download verification and checkout reproduction are recorded after they occur. The full five-policy comparison, strong B3 decision contribution, static/dynamic native tasks, control improvements and fresh 120-condition confirmation remain incomplete.
