# Exact geometry allocation optimization

The frozen H reevaluation finishes **9/12**, with three deadlines and no
collision or bounds episode. Each core family completes3/4known conditions.
This is75%overall on development data,below the80%engineering target;it does
not establish PPO value over the12/12strong rule or hierarchy advantage over
a healthy B1. The inherited model has20,674training controls/650Adam steps;
this reevaluation adds6,569controls and **zero updates**. All12initial actions
reproduce. [Endpoint](native_H_shared_mask_reevaluated_20261031.json).

The661recorded macro decisions include270selected searches exhausting their
7200-node budgets. Recorded planning time totals1117.95seconds. These are
failures of the selected bounded searches,not proof that the maps are
impassable. [Decision-level accounting](native_H_shared_mask_planning_outcomes_20261031.json).

A30-second,20Hz nonblocking local sample of the frozen worker captured580stack
samples,with13misses;205leaf samples were in per-pose`body_corners`. This is a
diagnostic sample,not an unbiased whole-training timing estimate. The isolated
profiler installation does not modify the training environment.

`ExactRoadBatch`now shares matrix allocations across poses while retaining
scalar`math.cos/sin`,the same local corner coordinates and matrix-operation
order. The closed-cell SAT,contact tolerance,scalar contact fallback,vehicle
footprint,map bounds and obstacle logic remain unchanged. Tests compare the
raw floating-point bits of60,030pose/vehicle/margin combinations,including
negative zero and large coordinates,and retain the independent scalar contact
and actual lightweight planning/MPC/physics/reward checks.

| Saved query,three interleaved repetitions | Original median | Batched-corner median | Full result |
|---|---:|---:|---|
| Guided known fixture | 76.16ms | 53.64ms | Same success,cost,170nodes,71path samples and path hash |
| Actual Hcase9initial global-goal query | 6.440s | 3.668s | Same7200-node budget exhaustion and empty path |

The hard-query improvement is about **43% less search wall time**;its collision
kernel falls5.346→2.621s. All549,151queried poses across7,626production batches
have exactly matching collision flags. The optimizer makes this failure
cheaper;it does not turn it into a successful plan or repair high-level choices.
Concurrent independently frozen CARLA work was present;these measurements are
not real-time certification or a guarantee of end-to-end training speed.
[Easy-query receipt](exact_corner_compute_benchmark_20261031.json),
[hard-query receipt](exact_corner_native_timeout_benchmark_20261031.json).

To repeat with the publicly downloaded native-controller baseline:

```powershell
python scripts/benchmark_exact_corners.py --baseline EXTRACTED/20261031T004553035320Z_native_curve_control --fixture experiments/native_recorded_timeout_query_20261031.json --output runs/exact_corner_timeout_reproduction.json --repeats 3
```

The benchmark verifies the archived kernel hash and checks that other physical
and search dependencies have identical source text,allowing Git line-ending
conversion. Original registered byte hashes are never changed. The benchmark
uses no CARLA connection,plant integration,optimizer update or new driving.

Git converts the derived JSON fixture to LF. Its decoded query values are exactly
equal to the original CRLF fixture,whose
[byte-exact release copy](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_recorded_timeout_query_20261031.json)
has SHA256`3bca0b623b320e4480964ad033aa4349a1d665bf692aa6573af31d1003d4c11a`.
Use that download as`--fixture`when checking the original byte hash.

The live B1rollout1280study and every previously prepared queued batch retain
their original frozen kernel`41fcb252…`;no running experiment was patched.
This optimization is a new source version for subsequent preparation/use.
Full local CPU regression passes668tests,1skip/4CARLA exclusions in164.55s.
Source7dfbb3dalso passes the full hosted run653tests/16skips/4CARLA exclusions
in158.15s,and independent geometry checks7/7in4.41s. The independent hard-query
repeat preserves all flags/nodes/status/path and measures6.133→3.671s.
The public Hmodel/raw/figure and661-decision diagnosis independently reproduce.
[Executed receipts](H_public_and_exact_corner_verification_20261031.json).
All original research acceptance gates remain unmet.
