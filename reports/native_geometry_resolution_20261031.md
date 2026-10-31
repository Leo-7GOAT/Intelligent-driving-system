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

Evidence: `native_B1_rollout1280_bounds_sources_v2_20261031.json`,
`native_B1_rollout1280_lane_coverage_20261031.json`,
`native_fine_grid_diagnostic_20261031.json`,and
`native_B1_geometry_v4_training_preparation_20261031.json` in this directory.
Local offline exports remain under runs;the required CARLA map is read from
the user's installation and is not redistributed. Original runs,models,scores
and failure receipts are retained.
