# Exact road collision batching — offline development, 2026-10-25

The full-body road checker was a material search cost: `World.free_many`
already vectorized bounds and static obstacles, but called Python road SAT once
per valid pose. The new **opt-in** `road_collision_backend=batch_sat_v1` groups
poses and occupied cells for NumPy SAT. The default remains `scalar`; the live
CARLA PPO batch `20261025T085433Z_native_route_learning` retains its frozen
`b495e71` source and scalar backend throughout all H0/H/B1 jobs.

## Geometry and version boundary

Occupied road raster cells remain closed rectangles. All four body corners,
the original two unnormalized body axes, map bounds, static-obstacle checks,
vehicle dimensions and configured margins remain in use. A contact-sensitive
band selects **the unchanged scalar oracle**, rather than changing the collision
tolerance. Cached occupied cells are refreshed when the public raster changes;
replacing the road also invalidates the World cache. Spatial bins group work
without excluding candidate collision cells. Scalar single-pose collision,
clearance, risk and terminal checks remain unchanged.

The backend is recorded in active configuration and policy signatures. Disabled
serialization/signatures retain the historical format. Reset selects the same
backend for H, H0, B1, B2 and B3; native training and fresh-training preparation
preserve the covered backend. The rule diagnostic exposes
`--road-collision-backend batch_sat_v1` only with an explicit versioned catalog.
It is a shared low-level compute intervention, not PPO decision competence.
No active model, normalization state, run protocol or frozen source was changed.

## Measured results

Three interleaved repetitions per backend used the saved, previously
frame-audited development wait state in
`tests/fixtures/native_wait_state_v1.json` (SHA256
`12ebb658006ce1d0b0124872e67f2161d814bdd8df01aae80e9f1ad347d1ff48`).
The script freezes its sources and fixture beside each output before querying.
No CARLA connection, new physical control or optimizer update is involved.

| Workload | Scalar median, seconds | Batch median, seconds | Verification |
|---|---:|---:|---|
| Actual guided-search road calls: 169 batches, 5,915 poses | 0.89236 | 0.11276 | Every road collision flag equal |
| Seeded scattered poses: 2,048 poses | 0.27458 | 0.07385 | Every road collision flag equal |
| Whole Euclidean search | 10.74583 | 1.96452 | Same no-path/open-set-exhausted, 2,637 nodes |
| Whole Dubins-guided search | 0.79789 | 0.15398 | Same success, 170 nodes, cost 8.3025, 71 path points and exact path hash |

The first ungrouped prototype regressed scattered poses: 0.28052s scalar versus
0.50946s batch. That result remains in
`runs/road_batch_prototype_20261025.json`; spatial grouping repaired the measured
regression. The intermediate grouped prototype is retained in
`runs/road_batch_spatial_20261025.json`. The table above is the integrated
production-dispatch experiment `runs/road_batch_integrated_20261025.json`.
Each JSON retains individual timings and source hashes, not just medians.

These are development microbenchmarks and one saved-state full-search comparison
under a concurrently running CARLA training job. They do not establish full
decision-cycle acceleration, task success, generalization or real-time operation.
The previously measured actual B3 P95 of 4.26/17.45/4.28/6.27 seconds remains the
latest complete live-cycle evidence. A separately versioned actual CARLA
comparison must wait for the current tick owner to finish and clean up normally.

## Reproduction

From the repository root with its isolated Python environment:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_road_grid_batch.py tests/test_road_grid.py tests/test_native_route_preparation.py -q
.venv/Scripts/python.exe scripts/benchmark_road_batch.py --output runs/road_batch_local.json --repeats 3
```

Outputs must be new paths; prior evidence is never overwritten. Timings are
machine-dependent. Search status, failure reason, node count, cost and path hash
must agree across the two backends in every repetition. Random full-body,
interior-hole, near-contact, mutable-cache, mixed static/bounds and actual
lightweight planner/MPC/plant/reward comparisons have passed targeted regression.
Complete CPU regression: **267 passed, five conditional skips**, 241.99s; two
existing SB3 deprecation warnings and one expected nonfinite-pose arithmetic
warning in the invalid-input equivalence test. No CARLA conditional test is
represented as executed by these CPU results.
The current live PPO endpoint and broader research acceptance remain incomplete.
