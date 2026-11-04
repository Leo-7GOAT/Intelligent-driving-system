# Spatial B2/Hybrid component check

`GridAStarPlanner` previously ignored `PlannerConfig.max_nodes`. The corrected
grid search stops at the exact expansion cap and returns `timeout` with
`failure_reason=search_budget`,consistent with the existing Hybrid interface.
The public `grid_search` three-value return and unlimited default remain
compatible. Tests cover exact counts,success at the last permitted node,and
skipping spline work after budget exhaustion.

The development measurement uses all eight curve and eight static-obstacle
initial states from a completed native known24 inventory. Actual reset frames,
road geometry,vehicle dimensions and registered terminal goals are retained.
Both pose-only searches receive32,000nodes and no wall-clock cutoff. B2 uses
the existing `body_spline_v1` optimizer and continuous spatial certificate.
Hybrid uses the saved base analytic/weighted-Dubins configuration. This is
different from the production measured-steering adapter.

| Input snapshot | Hybrid returned paths | B2 certified returns | Hybrid / B2 total wall seconds |
|---|---:|---:|---:|
| Newly closed seed11 H | 16/16 | 14/16 | 1.89 / 21.88 |
| Already published continued-seed H | 16/16 | 14/16 | 1.88 / 22.04 |

These are two snapshots of the same16registered development conditions.
They are not32unique conditions or independent training seeds. Cases11and15
return `body_spline_uncertified` in B2;this is not proof of geometric infeasibility.
The difference includes postprocessor convergence and certification conservatism:
B2 requires a continuous certificate,while Hybrid uses sampled primitive/analytic
collision checks. It does not isolate search kinematics or prove Hybrid superiority.
Node counts refer to different algorithm states;B2raster construction and spline
fitting are included in wall time but not expansion counts. These single-pass
shared-workstation timings are exploratory.

The first measurements retain source `fdc8825`. A final-source replay at
`8bec09b` reproduces every status,node count,cost,path bit and geometry metric.
Its timings are not substituted for the original measurements. The CLI snapshots
the executed grid source bytes and records the starting commit/worktree state.
The complete record is
`reports/native_spatial_planner_component_20261104.json`.

Reproduce using the already published `outcome_H_known24.zip` from the
[v0.4.0 development release](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.4.0-dev.20261104-outcome-shared-control).
Verify SHA256
`f930aa3811acacb497645cd971aa1697bd4afcf788d94695c5e5ab8e929982dc`
before extracting to a fresh directory. With the locked Python3.11environment
and source `8bec09b`,run from the repository root:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe scripts/benchmark_native_spatial_planners.py --batch <extracted>/20261104_outcome_frozen_H_known24 --output <fresh-output-directory>
```

The archive's522critical input files match the original bytes. One operational
cleanup record has declared path redactions;its original/export hashes match the
published transformation ledger,and its consumed status is identical. No new
download or model inference is required when the verified capsule is already local.

This adds zero native controls or optimizer updates. Native reference refinement,
steering-slew retiming,MPC,complete B2driving and frequency/safety ablations remain
pending. Historical B2batches retain their original frozen implementation. The
active H/H0and waiting B1sources are unchanged.
