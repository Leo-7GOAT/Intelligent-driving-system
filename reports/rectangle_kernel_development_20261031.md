# Recorded-query rectangle partition experiment

This is an offline compute experiment. Production sources and all active or
queued native training snapshots remain unchanged. It adds no vehicle controls
or optimizer updates, and does not change the research acceptance criteria.

`scripts/experimental_rectangle_kernel.py` partitions occupied cells into
nonoverlapping rectangles by merging identical horizontal runs across adjacent
rows. Their closed union is the original occupied-cell union. It inherits the
existing full-body SAT equations, tolerances and scalar near-contact fallback.
This geometric argument does not establish universal floating-point equivalence.

The two committed fixtures are the first recorded 7,200-node failure and first
successful non-reused search with at least two expanded nodes in the completed
seed22 H0 recovery evaluation. Selection precedes timing. The extractor binds
each scene to the actual episode ID and checks the audited raw-input and frozen
source hashes. The unsuccessful query remains unsuccessful.

| Recorded workload | Original cells | Rectangle partition | Result checks |
|---|---:|---:|---|
| Failed search, full-search median of 3 interleaved repetitions | 4.243370 s | 3.247504 s | Same timeout, search-budget reason, 7,200 nodes, empty path |
| Successful search, full-search median of 3 repetitions | 0.029790 s | 0.024171 s | Same success, 32 nodes, cost, all 1,085 path points |
| B1 observation median, 88 fixed-stride samples | 0.009447 s | 0.009787 s | Exact raw/normalized values and deterministic actions |
| B1 observation P95, same samples | 0.161988 s | 0.072849 s | All 12 cases, including terminal failures |

All 367,105 collision-query poses in 7,356 captured planner batches produce
identical boolean outputs. The successful path SHA256 is
`ba16f9399f563f1868f7555bae5b8786588e10f725075691c1e94c6dde706134`.
The observation experiment advances all 7,257 original measured histories,
samples every hundredth control plus each episode's first/terminal control,
and reproduces the recorded policy actions with frozen normalization.

The observation median is slightly worse. Construction costs are recorded
separately and are also higher for the partition. The P95 improvement is limited
to the sampled offline observations; 72.85 ms still exceeds B1's 50 ms nominal
period. Timing excludes the full native runner. Concurrent independent frozen
CARLA training may affect wall-clock measurements. No realtime or new native
success-rate claim follows from this experiment.

Three prototype tests cover exact occupied tiling, holes, checkerboards, free
and blocked grids, two vehicle geometries, three margins, boundary contacts,
grid mutation and empty batches. The finite checks support further evaluation;
the prototype is not yet a replacement for the frozen training kernel.

Reproduce the search experiment from the repository root with a downloaded and
verified native H/H0 v4 PPO capsule as `BASELINE`; its seven checked physical
and search dependency texts match the recovery source:

```text
python scripts/benchmark_rectangle_kernel.py --baseline BASELINE --fixture tests/fixtures/native_v4_first_budget_query_20261031.json --output budget-new.json
python scripts/benchmark_rectangle_kernel.py --baseline BASELINE --fixture tests/fixtures/native_v4_first_success_query_20261031.json --output success-new.json
python -m pytest -q tests/test_rectangle_kernel_experiment.py
```

For observation reproduction, set `NAVSTACK_AUDIT_SOURCE_ROOT` to the verified
B1 v4 PPO capsule's `source` directory before starting Python, then run
`scripts/benchmark_rectangle_observations.py --batch B1_CAPSULE --output observations-new.json`.
Outputs must be fresh paths; scripts refuse overwriting evidence.

Raw aggregate reports are `rectangle_kernel_recorded_budget_benchmark_20261031.json`,
`rectangle_kernel_recorded_success_benchmark_20261031.json` and
`rectangle_kernel_B1_observations_20261031.json` in this directory.

Separately, the matched replay-fit B1 continuation completed 20,480 new native
training controls and 51 optimizer steps. Its development endpoint is **4/12**,
with four bounds failures, one collision and three deadlines. This regresses
from its actor-fit starting endpoint (8/12) and remains an unhealthy baseline.
The complete failed endpoint is preserved in
`native_B1_v4_replay_fit_PPO_endpoint_20261031.json`. The paired approach-fit
continuation is running; no outcome is inferred before its evaluation completes.
Formal five-seed/new-condition research acceptance remains **UNMET**.
