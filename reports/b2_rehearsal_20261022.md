# Rehearsal retention and stronger B2 postprocessing

This is development evidence, not independent confirmation or a hierarchy-superiority result. The previous turn verified an OS-live training worker; this turn implemented and tested B2 changes, completed frozen-checkpoint probes and performed new native-road experiments. Full research acceptance remains incomplete.

## B1 at the second stage endpoint

v19 checkpoint at2,000,058physical steps retains6m parking and learns12m parking. At each distance, deterministic and stochastic inference each succeed8/8 for both train and validation labels. These labels reuse the same simple geometry; they are not32independent conditions. Both audits use the batch's frozen source/model/normalization and retain all physical traces. The full ten-case development endpoint is0/10. Existing v18 at2M achieved12m8/8 but6m0/8 in both inference modes. This supports continuing the registered rehearsal hypothesis, not declaring a healthy mixed-task baseline.

Evidence: `runs/20261022T134149Z_route_rehearsal_v19_development/retention_2M_6m_frozen` and `retention_2M_12m_frozen`. At the first1M milestone the v18/v19 saved policy tensors and normalization means were not bitwise equal, even though the configured first-stage distribution is unchanged; the1M saved rollout can straddle the stage transition. Thus this is a fresh-run development comparison, not a controlled branch from an identical1M policy. `runs/v19_stage_entry_audit.json` retains the mismatch. Subsequent configuration/seed equality does not prove trajectory identity.

## Why the original B2 failed

All four saved native-road scenes were audited without selecting successful routes. The original B2 spline attempts have excessive curvature and/or body collision. Two exact target poses are free without inflation but fail the fixed0.10m postprocessing envelope, making that gate impossible for any exact endpoint fit. Clearance-cost A* plus ordinary smoothing, interpolated distance fields and sampled-footprint optimization were also tried; none passed the old full gate. Their failures are retained locally in `b2_smoothing_probe.json`, `b2_optimization_probe.json`, and `b2_optimization_exact_probe.json`. These probes do not change a deployed planner or become hidden successful samples.

The optional `PlannerConfig.grid_postprocess="body_spline_v1"` keeps the corrected2D A* search. A bounded local least-squares fit moves endpoint-tangent spline knots, penalizing oriented-body SAT overlap, curvature, stationary tangents and deformation. It uses no Hybrid A* trajectory, expert route or policy action replacement. The optimizer uses up to150function evaluations; each has finite-difference Jacobian cost, so150 is not the total residual-call count. Nearest64rectangles are guidance only. Every accepted spline is checked against all relevant actual World geometry.

The final check derives Taylor bounds for speed, minimum tangent norm, curvature and body displacement over every cubic interval. A midpoint envelope inflated by the interval's maximum translation plus rotation covers the full swept body. Subdivision tightens this bound at five fixed parameter spacings from0.02to0.00125; it never raises the curvature limit or removes obstacle cells. This replaces the unexplained fixed0.10m inflation with a conservative interval certificate. It does not account for actual tracking error or dynamic obstacles; existing trajectory/execution checks remain required.

The legacy mode remains the default. The new option changes checkpoint signatures; historical models and running v19 source are unchanged. New formal B2 comparisons require matching retraining, not silently replacing a historical low level.

## Offline native results

| Route | Legacy B2 | New B2 | New B2 search nodes | Planning seconds |
|---|---|---|---:|---:|
|0|infeasible|success|127|3.781|
|1|infeasible|success|192|1.321|
|2|infeasible|success|144|2.270|
|3|infeasible|success|121|0.798|

All reparameterized references also pass full-body grid checks. Hybrid A* finds all four paths under the larger10,000-node offline budget; its first two require1938/2444nodes, exceeding the usual1800-node limit. Neither timing is a real-time claim. New B2 still has a different search/optimization budget and exact endpoint handling; this four-scene feasibility audit cannot prove superiority over Hybrid A*.

Evidence: `runs/native_b2_certified_20261022`, including hashes, all eight H/B2 results, paths, references and plot. An earlier certificate without refinement rejected route0; `runs/native_b2_body_spline_20261022` preserves that3/4outcome. The curve geometry was unchanged by certificate refinement. Full CPU regression:111passed/5conditional skips, including thin-obstacle swept-body rejection, stationary tangents, exact endpoint constraints, default signatures and explicit new versioning.

```powershell
python scripts/audit_native_route_feasibility.py --source NATIVE_ROAD_LEARNING_BATCH --output NEW_AUDIT --grid-postprocess body_spline_v1
python scripts/native_curve_control.py --prepare SURVEY_BATCH --planning-audit NEW_AUDIT --plan-method B2
```

Native execution uses all four registered paths, unchanged legacy actuation/MPC/speed and both raw-road and raster failure gates. This is fixed-plan control diagnosis, not PPO performance. Its actual results and publication verification are recorded below after completion.


## Actual CARLA execution and third-stage update

All four B2 references were executed in `runs/20261022T142205Z_native_curve_control`: outcomes sampled-footprint departure / raster-body departure / sampled-footprint departure / success. Total1111control frames and1234world ticks before cleanup; no observed collision or QP failure. Lateral RMS0.0353/0.0259/0.0179/0.0406m, speed RMS0.415/0.475/0.415/0.382m/s. Measured acceleration minima−23.57/−23.10/−23.61/−22.77m/s² and jerk RMS121.6/122.7/133.5/132.9m/s³ remain poor. All raw-state/IMU/frame/timing records were checked. Some control cycles exceed50ms. Thus4/4offline certificates become only1/4actual task completions; the postprocessor is not a validated driving solution.

The two raw failed road samples exceed the nearest lane half-width by0.01391m and0.00977m; the first has0.102m longitudinal projection offset. This read-only query does not prove the actual drivable-road union. The separate raster failure demonstrates that tracking deviation can consume nominal path clearance. Neither failure gate was disabled. Owned server launcher19380/shipping30724 was verified empty and stopped; the first readiness timeout was followed by a successful probe of the same server.

At3,001,971physical steps, the same v19 policy completes deterministic6/12/18m probes8/8 each. Stochastic6m and18m are8/8,12m6/8. Train/validation labels duplicate geometry and are not independent conditions. The unchanged full-task development list is still0/10. Training continues into23m and then mixed tasks under the original budget; there is no claim of full-task convergence. Raw probes are `retention_3M_6m_frozen`, `retention_3M_12m_frozen`, and `retention_3M_18m_frozen` in the same batch.

The prior rehearsal/native feasibility asset was actually uploaded and downloaded: `rehearsal_native_feasibility_diagnostics.zip`,2,807,329bytes,220manifest members, SHA256`f4dc4d3316c361ec407cce5fde3f94b42813977be04c5fc2cf74e145c1165092`. The downloaded ZIP and every member hash match. It predates the B2 improvement and must not be described as containing this new control run.


## Independent source, asset and actual CARLA reproduction

Commit`53361770b667bb6802b1570286e692245882554c`is pushed and remote-SHA verified. New immutable asset`b2_native_postprocess_diagnostics.zip`has3,105,133bytes/243manifest members, SHA256`296a5614e17bdcfdd7d1c2fb094e0db289c1cd6b57e1c8a0496dd2c0b0afb76a`; actualdownload and all member hashes match. The existing tag and earlier assets remain unchanged. It includes the original1/4control batch, both offline B2 audits and frozen source/calibration; it does not include later reproduction frames or the running v19 model.

The independent`native_road_checkout`was updated only after checking it clean. With its own source and separately installed pinned environment,111tests pass/5conditional skip; all four downloaded native-road models load with frozen normalization/4447inputs. Rebuilding the downloaded B2 raw-frame summary and plot gives byte-identical files.

Actual fresh-checkout batch`20261022T143149Z_native_curve_control`executes the four downloaded B2 paths: raw-road failure/success/raw-road failure/success,1381controls/1504world ticks. This2/4differs from the initial1/4: route1now succeeds and route0fails earlier. Thus execution near the boundaries is not outcome-stable; do not pool these repeats as independent scenarios or hide the original failure. No observedcollision/QPfailure, but accelerationmin reaches−24.55m/s². All actualframesverified; ownedlauncher53244serverempty/stoppedwithreceipt. Freshcheckoutremainsclean. This is reproducibility of the entry and data flow, not bitwise CARLA replay or research acceptance.

Frozen provenance for all five v19 distance audits is verified against the batch's source hashes and each validation checkpoint's model SHA, recorded in`runs/v19_retention_provenance.json`. Continue the original5Mbudget and inspect both23m/full-task endpoints and retained short distances before choosing the next learning intervention.
