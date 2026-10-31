# Native boundary diagnosis and explicit v4

The original longer-rollout B1score remains6/12with four bounds failures.
Cases5and7leave the registered task rectangle. Cases3and9stop solely on the
coarse occupied-cell raster;all twelve native body samples report Driving.
Finite samples alone cannot certify the full body.

The offline map has the same OpenDRIVE SHA256as the original recorded map.
Undilated.1mCARLA lane-width ribbons cover case9's complete body with.433m
clearance. Case3has a5.83e-8m²uncovered sliver in that approximate ribbon union;
it is not labeled completely covered. The twelve-point and1127-point probes
are supplementary diagnostics,not continuous-surface certificates.

The new `carla_driving_indexed_union_256_v4` samples the same indexed native
Driving-lane membership at cell corners and centres. It keeps the common64x64
policy grid and uses256x256closed occupied cells for full-body bounds,planning,
risk and B2postprocessing. No lane-width dilation,changed task rectangle or
success threshold is introduced. Recorded cell sizes shrink from roughly
.40–.46mto.10–.12m. The two raster-only endpoint bodies have.379mand.435m
clearance under v4;both task-box violations remain invalid.

The finer grid is available to every method's geometric operations. The common
policy grid is unchanged;other geometry-derived witness/risk features can
change. This is a disclosed change in map approximation,not bitwise-preserved
environment behavior or proof that PPO has improved. New reset grids are
cached only for v4by map digest,complete layout and version. Cached construction
does not tick or move the vehicle. Offline construction took3.16/4.80seconds
for the two distinct layouts.

Tests preserve holes inside a vehicle's body,actual road/task-bound violations,
scalar/batched agreement,changed fine-map invalidation,versioned hashes and
equal observation shapes across H/B1/B2/B3. Full CPU regression:671passed,
1skipped,4CARLA deselected,167.30seconds. Later preparation checks:30passed,
5.38seconds. These CPU results do not replace native evaluation.

Batch022009is a frozen v3checkpoint migrated onto v4,with zero training and
all twelve known development conditions. It completes7/12with1bounds and
4deadlines;basic motion3/4withzero unsafe. All6,542newcontrols and12initial
actions are audited;model/normalizer bytes match the original checkpoint.
The full-task target remains unmet. Batch022652starts real PPO after
the geometry change;its initial weights,empty optimizer,normalizer and RNG
match v3004737. It retains the original v3teacher dataset,which is disclosed
extra supervision. Endpoint selection stays fixed at20,480controls with all
twelve cases;neither phase counts as new final conditions or additional seeds.

The same recorded Hcase9initial static-planning query was also compared under
both geometries,with the recorded7200-node budget,vehicle,start,goal,costs and
Dubins heuristic unchanged. V3exhausts7200nodes;v4finds an analytic connection
at node1,with1625path points. This is a changed geometry result,not merely the
earlier allocation optimization. One query does not establish a full driving
success rate or a real-time bound. See `native_fine_grid_recorded_query_20261031.json`.
Frozen H/H0v4migrations023833/023839are separately registered behind the active
B1PPO phase,with zero new training and the same twelve development conditions.

Public code19da5acpasses GitHub CI657tests/16skips/4CARLA deselections in159.70s
([executed run](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41160661963)).
An independent checkout passes30targeted new-geometry/preparation tests in12.60s.

The final source2fe4af8also passes672local tests/1skip/4CARLA deselections in
217.71seconds and hosted657tests/16skips/4deselections in191.76seconds
([executed CI](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41162252424)).
Seven plot/geometry checks pass7.06seconds;44preparation/partition checks pass
7.19seconds. The corrected v4figure displays the256collision cells;the v3H0
figure still reproduces exactly.

The437-file B1migration capsule has been downloaded publicly and independently
reproduced,including6,542controls,12initial actions,frozen statistics and its
fine-grid figure. Both public inputs to the saved query also reproduce the
same statuses,nodes,costs and complete path hashes. Exact receipts and URLs:
`native_geometry_public_verification_20261031.json`. The7,864-byte fine-grid
gzip export is derived local occupancy data;the full CARLA map is not included.
After downloading/decompressing the inputs linked in that receipt:

```powershell
python scripts/compare_fine_grid_query.py --fixture downloads/native_recorded_timeout_query_20261031.json --fine-export downloads/fine_grid_export.json --output runs/fine_grid_query_reproduction.json
```

Evidence: `native_B1_rollout1280_bounds_sources_v2_20261031.json`,
`native_B1_rollout1280_lane_coverage_20261031.json`,
`native_fine_grid_diagnostic_20261031.json`,and
`native_B1_geometry_v4_training_preparation_20261031.json` in this directory.
Local offline exports remain under runs;the required CARLA map is read from
the user's installation and is not redistributed. Original runs,models,scores
and failure receipts are retained.
