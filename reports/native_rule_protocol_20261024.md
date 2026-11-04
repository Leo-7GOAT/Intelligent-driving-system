# Native catalog rule diagnostic

The current native PPO experiment uses a 31-action ego-frame catalog on four
measured roads. Earlier fixed-reference driving establishes low-level tracking
on those roads, but does not establish closed-loop competence of this catalog.
The new `scripts/evaluate_native_rule.py` provides the corresponding real CARLA
evaluation using the existing `navstack.env.rule_action` selector.

This diagnostic **actually completed in CARLA at 18:38 local on 2026-10-24**:
`runs/20261024T103653Z_native_rule`. It was prepared only after the six-arm parent
completed and its owned server was cleaned up. A fresh owned server on RPC2100
passed the empty-world probe with matching client/server0.9.15. Worker16572 and
guard25516 then completed normally, with owned server59536 stopped after cleanup.
An earlier attempt against the still-live parent was rejected before preparation.

## Measured result and failure diagnosis

| Road | Result | Actual controls | Progress (m) | Planning/MPC failures | Interventions |
|---|---|---:|---:|---|---:|
| 0 | success | 368 | 24.196245 | 0/0 | 0 |
| 1 | success | 371 | 23.586131 | 0/0 | 0 |
| 2 | success | 338 | 22.304492 | 0/0 | 0 |
| 3 | deadline | 900 | 18.755571 | 0/0 | 0 |

The complete trace contains1,977 applied controls,123 reset ticks and2,100 world
ticks, with zero training controls and zero optimizer updates. Wall time is39.94s.
Raw audit SHA256 is `c9df6f6765b204be4cfe30beaa36bd82ee7877ac261c319f8531ae7610611e46`.
All four measured starts, goals, static/dynamic task definitions and road hashes
match H_rollout32 exactly. H remains0/4. The rule shares H's frozen production
source, environment configuration and calibration. These reused training-road
probes support a development diagnosis, not independent confirmation or RL value.

Rule-selection P95 per road is0.03783/0.02118/0.03389/0.03513s, P99 is
0.03877/0.02151/0.03562/0.04280s, and maximum across roads is0.04948s.
Complete decision-cycle P95 is0.35609/0.29136/0.34687/0.39619s, P99 is
0.38207/0.33573/0.37282/0.41628s, maximum0.41680s, with no measured0.5s overruns.
These are this run's wall measurements, not a deployment latency guarantee.

Road3 executes35 reference decisions then55 waits. Before its first wait
(zero-based decision35), goal distance is1.946988m; at the last decision it is
1.968249m. Reconstructing all31 production actions at both measured states yields
one wait and30 planning failures, with no moving or parking reference. The final
yaw error is only0.022486rad and speed is approximatelyzero, but distance remains
outside the task tolerance. This is a candidate-coverage/approach-state failure
witness, not proof that the initial task is infeasible. Rule selection itself
records no planning-failure intervention because it chooses wait. Counts must
therefore be read together with progress and reference availability.

`scripts/diagnose_native_rule_endpoint.py` checks parent/config/source identity,
revalidates raw frames, pairs measured tasks and reconstructs the failed-case
first-wait and last-decision candidates without ticks or policy updates. Output:
`endpoint_diagnostic.json`, SHA256
`375c72f428bb9ddc930c07fcc756ab573bee5e88a679e61e5365a5fdd4978ccf`.
At source `6d862b1`,13 pairing, evaluator and raw-audit tests pass locally and
in the independent checkout. Raw-audit reproduction is recorded with the full
rollout receipt; the separate endpoint-diagnostic reproduction receipt is
`runs/prerelease_20261022/native_road_checkout/runs/native_rollout_complete_reproduction/rule_diagnostic_reproduction.json`.

```powershell
.venv\Scripts\python.exe scripts/audit_native_rule.py --batch runs/20261024T103653Z_native_rule --output runs/rule_raw_rebuilt.json
.venv\Scripts\python.exe scripts/diagnose_native_rule_endpoint.py --batch runs/20261024T103653Z_native_rule --parent runs/20261024T075110Z_native_rollout_pair --output runs/rule_endpoint_rebuilt.json
```

Output paths must be new; original evidence is preserved. No strengthened rule
or new learned policy has been trained as part of this diagnostic.

## Fixed comparison conditions

The evaluator copies the selected H arm's exact environment configuration, all
four route definitions, calibration and frozen source. The policy is explicitly
labelled `rule_only_B3`; the environment retains H semantics so changing the
selector does not silently change planning, time parameterization, MPC, action
set, safety, rewards or termination rules. It uses the same 7301+case-index reset
seeds as native PPO evaluation, records actual scenes and retains every outcome.
There is no PPO model, training interaction budget or independent training seed.

Each action is selected by the existing rule, then executed by the production
environment. The evaluator records full CARLA trace, raw bridge requests and
responses, actions, termination metrics, world/reset/control counts and separate
rule-selection and complete decision-cycle times (P95/P99/max/overrun fraction).
These data require a completed raw-frame audit before use as native results.

The existing ego-arc rule checks candidate endpoints and a primitive approximation;
it does not search every possible candidate reference. It therefore cannot be
assumed to satisfy the final requirement for a strong rule baseline. The first
run establishes its behavior under the unchanged catalog; any later strengthened
selector must be separately versioned, share the available planning/control
capabilities and disclose additional computation. Rule success is never PPO
success, and rule failure does not prove no feasible catalog policy exists.

## Reusable execution procedure

```powershell
.venv\Scripts\python.exe scripts/evaluate_native_rule.py --prepare runs/20261024T075110Z_native_rollout_pair --source-method H_rollout32 --wall-seconds 1800
```

The command returns a new batch with `PREPARED_NOT_STARTED`. Use the existing
owned CARLA launcher and its protocol port, passing the existing isolated
CARLA interpreter through `NAVSTACK_CARLA_PYTHON`. Run the new batch's frozen
`source/scripts/evaluate_native_rule.py --batch BATCH`, record its actual process
identity and adopt it with `watch_existing_pair.py`, including the PowerShell
executable for owned cleanup. The completed run above followed this procedure.
Its initial 30-minute resource window, disk/RAM/GPU limits,
300-second no-progress guard and zero automatic retries follow the existing
lifecycle; a failed run is preserved rather than silently repeated.

At source `bee41c4`, five runner/cleanup tests pass locally and in the independent
checkout. They verify parent gating, preservation of failed fixture episodes,
applied-control accounting and rule computation timings, plus existing cleanup
failure receipts. The fixtures use lightweight dynamics and are not CARLA
performance evidence. Receipts are `runs/native_rule_runner_regression.xml` in
each checkout. No new scientific result is claimed from these tests.

## Raw-frame verification before reporting a result

Source `362a11a` adds `scripts/audit_native_rule.py`, which preparation also
copies into the frozen diagnostic source. After actual execution completes:

```powershell
.venv\Scripts\python.exe scripts/audit_native_rule.py --batch BATCH
```

The verifier requires all registered cases and exactly one complete bridge
stream, binds every reported control to its measured state/command/frame/IMU
and episode, checks terminal metrics against the trace, and accounts for every
control and reset tick. It also verifies frozen source/calibration/configuration,
scene identities, zero training and optimizer costs, and recomputed timing
summaries from per-decision measurements. It rejects altered states, duplicate
post-terminal records, omitted cases, altered timing summaries and invented
training costs. Six synthetic verifier tests plus the two evaluator tests pass
locally and independently (8 total). These are fault-injection fixtures, not an
actual native B3 result. The subsequent measured result is reported above,
separately from these historical synthetic test receipts.
