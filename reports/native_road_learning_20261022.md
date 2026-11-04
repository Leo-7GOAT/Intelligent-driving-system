# Shared native-road interface and retention evidence

The native-road integration now uses the same measured64x64 road raster for full-body HybridA*/2D-A* validity, local occupancy, an additional4096 policy inputs, road risk and failure-first termination. All H/B1/B2/B3 receive every cell used by the planner, plus the existing map-corner coordinates. Base inputs become4447; the option is disabled for existing models, with an explicit new checkpoint signature. Native PPO requires this option. Experimental PI/feedforward actuation remains rejected for training because actuator internal states are not yet integrated into observations. Legacy actuation is retained for these new experiments.

Each cell is blocked if any of its four corners or centre is outside CARLA Driving lanes or fails the height check. Geometry is hashed and saved per episode. Collision uses full oriented-body/closed-cell SAT, including holes inside the body; batched/scalar paths agree with an independent rectangle implementation. This is conservative relative to sampled points, not continuous physical-road truth. Existing native footprint samples provide an additional failure gate. The shared parking gate also checks absolute speed, preventing reverse motion from passing the stop criterion.

## Actual execution

Both new batches are actual CARLA0.9.15, fresh PPO seed11,20s diagnostic episodes, fixed32-action rollouts and a nominal640 controlled-step threshold. Four native geometries are selected before driving. Exact physical counts include partial episodes and rollout overshoot; these are not fair-performance comparisons or independent seeds.

| Batch prefix | Method | Control/world/reset ticks | Optimizer steps | Actor L2 change | Completed episodes |
|---|---|---|---|---|---|
|130938Z|H|640/701/61|4|.041862|1 deadline|
|130938Z|B1|949/1072/123|6|.055711|3 raster-boundary failures|
|131707Z|H|640/701/61|4|.041853|1 deadline|
|131707Z|B1|640/732/92|4|.046513|2 raster-boundary failures|

Names start`20261022T`and end`_native_road_learning`. Each method/batch also retains one unfinished episode. All models reload exactly with identical deterministic predictions. Both H runs execute320 solved MPC cycles,32 successful reference plans,29 invalid targets and3 wait actions. B1 uses direct controls without a hidden planner. There are no completed successes. Fresh simulation trajectories differ despite identical seeds; the second run verifies the strengthened resource checks/explicit recorder closure, not another independent training seed.

The analysis matches every recorded execution state to its raw bridge response, checks unique consecutive frames and IMU fences, validates0.05s timestamps, retains all terminations, and rebuilds trajectory overlays. Total verified controls are2869. The first server readiness probe timed out during startup; retry reached the same server, with no duplicate launch. Both owned servers were verified empty and stopped, with receipts in the first batch and`20261022T131600Z_native_grid_audit`respectively.

The first runner had per-step RAM/disk and wall checks but did not yet enforce its declared12GiB/GPU guard. The second runner adds the actual guard before work and each bounded rollout. This discrepancy is retained, not retroactively hidden. CPU workers are single-threaded; the second window was900s, no automatic retry. Latest CPU regression:107passed/5conditional skips.

All four goal footprints are free under the new raster. The original survey centreline has0/4/1/0 blocked full-body samples respectively. Thus a previously sampled native centreline is not automatically a valid reference under the stricter raster; alternate route feasibility must be audited before expanding native PPO training. No such failures were removed or repaired by snapping measured states.

## Map-query limitation

Visual inspection exposed isolated blocked cells inside the broad road corridor. Read-only audit`20261022T131600Z_native_grid_audit/map_queries.json`queried all rejected samples against the nearest Driving waypoint:5535/12763/4358/9505 failed samples; none lay more than5cm inside the nearest lane half-width, and none were height-filter failures. Some were within2cm outside lane width (71/47/42/24), so narrow lane seams can expand into whole blocked cells. No filling/dilation was applied without evidence. CARLA documents the projection distinction in its[0.9.15 Map API](https://carla.readthedocs.io/en/0.9.15/python_api/#carlamap). These are raster-boundary failures, not proof of actual asphalt departure; native-road geometry accuracy and route feasibility need further development before formal acceptance.

## Completed B1 distance curriculum

v18 completes5,002,111physical steps,513,024actions,2004optimization phases,47,170optimizer steps and12,566PPO epochs. Model reload is exact; supervisor exit0 and absent worker handles confirm completion despite the stale worker-status file saying evaluating. At2M, frozen inference succeeds8/8 on12m for both deterministic and stochastic modes but0/8 on6m; the frozen-source repeat confirms the latter. At4M,23m results are deterministic8/8 and stochastic6/8;6m retention is1/8 and5/8. Train/validation initializations duplicate simple empty geometry and are not independent scene sets. Full-task development milestones1M/2M/3M/4M/5M are0/10,1/10,0/10,2/10,0/10. Final23m inference regresses to0/8deterministic and1/8stochastic. Final mixed training's last100episodes have20successes,63collisions and17deadlines. This supports basic full-length movement/parking learning at4M but exposes forgetting and poor obstacle transfer; it does not establish a healthy final baseline.

The final ten-case physical-frame audit retains two empty deadlines and eight obstacle collisions. Empty7301stops1.034mshort with yaw.281rad, and empty7302stops.992mshort; neither enters.8mtolerance. All obstacle cases collide at3m/s, with no requested braking before collision. No further blind continuation is launched. The fixed-distance curriculum needs retention and obstacle-learning diagnosis before a new registered version.

## Published reproduction

Source`7a82fbc83f7b23ec27806b1e13ed1ee5f0dbe343`was pushed and remote-verified. A fresh independent checkout passes107tests/5conditional skips. Actual downloaded`native_road_learning_development.zip`matches SHA256`b54fdd0f142f1944878a9e095363ffde716162f2dd772dcaafd9cb9b01c34757`and396manifest hashes. Its four models load with4447inputs and frozen normalization; the second batch's plot/summary rebuild byte-identically. The initial slow direct download was interrupted only after verifying its owned process; the partial file is retained separately. The complete proxy download is the verified artifact.

The fresh checkout additionally prepares from the downloaded calibration/route data and actually runs`20261022T132946Z_native_road_learning`: H/B1each640controls,701/732worldticks,4optimizersteps and exact reload. H actor change.041714/B1.048489; H1deadline/B1two raster-boundary failures, oneunfinishedepisodeeach. All1280controls match raw CARLA responses. This is same-seed reproducibility, not additional independent performance evidence. Server startup needed a second read-only readiness probe; the original server was reused, verifiedemptyandstopped. No live training or CARLA service remains at final inspection. Main acceptance remains incomplete.

## Commands

```powershell
python scripts/native_road_learning.py --prepare runs/20261022T111210Z_carla_steering_audit
# Read runs/native_road_learning.json; use its new batch, never duplicate a live run.
scripts/start_carla.ps1 -OutputDirectory BATCH/infrastructure
# Confirm scripts/probe_carla37.py reports matching versions before executing.
$env:NAVSTACK_CARLA_PYTHON = (Resolve-Path .carla-python37/runtime/python.exe).Path
python BATCH/source/scripts/native_road_learning.py --batch BATCH
python scripts/analyze_native_road_learning.py --batch BATCH
scripts/stop_owned_carla.ps1 -Batch BATCH
```

The preparation input is a local saved survey/calibration, not an installed CARLA map replacement. Final acceptance, healthy mixed-task B1, five-seed comparisons, strong-rule RL value, native static/dynamic integration and fresh120-condition confirmation remain incomplete.
