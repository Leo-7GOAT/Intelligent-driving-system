# Retention diagnosis and native planning feasibility

## Policy/normalization diagnosis

The v18 4M and5M policies were crossed with both saved normalizers in the original frozen environment: four combinations, each ten full-task development cases plus two6m empty probes. Parameters and normalization remained frozen; saved model hashes were verified. All48episodes and per-frame records are retained in`runs/v18_normalization_transfer_4M_5M_verified`. These are counterfactual development probes, not extra independent training seeds or candidate models for final selection.

| Policy | Normalizer | Full-task success |6m success|
|---|---|---|---|
|4M|4M|2/10|0/2|
|4M|5M|2/10|0/2|
|5M|4M|0/10|0/2|
|5M|5M|0/10|0/2|

The4M policy retains both full-length empty successes under5M statistics, while the5M policy does not recover with4M statistics. Thus normalization alone does not explain terminal-parking loss. Statistics still affect behavior: old normalization changes two static collisions into deadlines for the5M policy. Across obstacle episodes, the mean clipped fraction of the180 known-obstacle slots is about6.2%with4M stats versus1.66%with5M; these include padding, so this does not prove obstacle features are healthy. An initial diagnostic without explicit model-hash provenance was repeated in the verified directory; both outcomes agree and neither was overwritten.

## Actual v19 training

Fresh seed11 starts from scratch with the same v18 network, reward, PPO/gSDE settings, physical5M budget and6/12/18/23m stage boundaries. The sole config change is`route_rehearsal_probability=0.25`: after completing a distance stage,25%of training resets choose uniformly among earlier distances; other episodes retain the active stage or final mixed distribution. This is new on-policy physical experience, not replay-buffer training or expert data. Validation remains unchanged. Rehearsal also continues after4M, reducing the proportion of mixed-task episodes; actual distribution and all physical cost must be reported. All architectures share this option, but only B1 is trained in this diagnostic.

Registered batch`runs/20261022T134149Z_route_rehearsal_v19_development`, group`route_rehearsal_v19/B1_route_rehearsal_seed11`; started21:42:58, two-hour deadline about23:41. One CPU worker,12GiB disk/4GiB RAM/85°C GPU/300s no-progress guards, no automatic retry. Supervisor14444/worker49696 identities and logs are OS-verified; launcher PIDs45832/12672 are separate. At96,695physical steps the policy had1,494optimizer steps and actor updateL2 .2504;100k/300k full-task evaluations remain0/10. Before1M the schedule deliberately matches v18 and does not yet test rehearsal. Do not interpret these early updates as a successful retention intervention.

## Native geometry and B2 correction

Audit of the actual saved64x64 native grids uses all four routes, calibrated vehicle geometry, no wall-time search cutoff,10,000-node maximum, and unchanged speed target1m/s. HybridA* finds collision-free paths for all four, including routes whose original sampled centreline crossed blocked cells. Reparameterized references also pass the same full-body grid checks. Nodes are1938/2444/71/204; measured planning times4.80/7.62/.47/.83s. These are offline feasibility witnesses, not real-time performance; the first two exceed the normal1800-node setting.

B2 previously applied the new road only at final body validation, omitting it from its centre-space search grid. The search now inflates occupied road cells by vehicle half-width, with an independent no-path regression. It still fails all four native postprocessing attempts after127/192/144/121expanded grid nodes; do not claim this establishes a fundamental advantage over a well-developed2D baseline. Its curvature/clearance postprocessing needs further diagnosis. Source hashes, all failures, paths and plot are saved in`runs/native_planner_audit_20261022`.

## Actual execution of searched paths

`runs/20261022T134803Z_native_curve_control`executes those four registered HybridA* paths through time parameterization and actual MPC/CARLA under unchanged legacy actuation. The returned road-grid hash must match the audit; both raster boundaries and raw CARLA footprint probes remain active. This is a fixed-plan diagnostic, not PPO performance.

Outcomes are two successes and two sampled-footprint departures, with1462actual control frames and no observed collision or QP failure. Per-route control ticks102/512/373/475; lateralRMS .00325/.06335/.01661/.05072m. All failures are retained. Actual braking spikes remain near−25to−26m/s²and jerk130–139m/s³. Native initialization and control time are recorded separately; some control cycles exceed50ms.

The failed footprint samples exceed the nearest-lane half-width by only.000505m and.006833m, with about.042m longitudinal displacement from the returned nearest centre. The first point's lateral component is slightly inside half-width. CARLA0.9.15's[Map::GetWaypoint implementation](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/road/Map.cpp)checks planar distance to that centre against lane half-width. This supports investigating lane projection and the union of neighboring lanes; it does not justify dropping failures or silently relaxing the boundary. The five-point cell approximation also cannot guarantee agreement at every continuous body point.

Read-only geometry diagnosis is retained in`probe_gap_audit.json`; `audit_native_execution_boundary37.py`provides the reproduction entry. OwnedCARLAserverlauncher49476wasverifiedemptyandstopped. Initial startup readiness timeout was followed by a successful probe of the same server; no duplicate service was started. Latest full CPU regression:109passed/5conditional skips.

## Reproduction commands

```powershell
python scripts/audit_normalization_transfer.py --batch V18_BATCH --left 4000000 --right 5000000 --output NEW_AUDIT
python scripts/initialize_route_rehearsal_development.py
# Use the existing pointer/worker when already initialized; do not duplicate v19.
python scripts/development_watchdog.py --batch V19_BATCH
python scripts/audit_native_route_feasibility.py --source NATIVE_ROAD_LEARNING_BATCH --output NEW_PLANNER_AUDIT
python scripts/native_curve_control.py --prepare SURVEY_BATCH --planning-audit NEW_PLANNER_AUDIT
# Start/probe existing CARLA, set NAVSTACK_CARLA_PYTHON, execute the returned batch's frozen native_curve_control.py.
python scripts/analyze_native_control.py --batch CONTROL_BATCH
```

The evidence package preserves the complete control batch, all planner failures and all frozen-normalization diagnostic traces. Main acceptance remains incomplete: healthy mixed-task B1, native road semantics, strong-rule RL value, five independent policies, full ablations and fresh120-conditionCARLAconfirmation are still required.
