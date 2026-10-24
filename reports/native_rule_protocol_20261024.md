# Native catalog rule diagnostic

The current native PPO experiment uses a 31-action ego-frame catalog on four
measured roads. Earlier fixed-reference driving establishes low-level tracking
on those roads, but does not establish closed-loop competence of this catalog.
The new `scripts/evaluate_native_rule.py` provides the corresponding real CARLA
evaluation using the existing `navstack.env.rule_action` selector.

This is **implemented and tested, not yet prepared or executed in CARLA**.
The active six-arm PPO rollout comparison owns the simulator. An actual attempt
to prepare against that live parent was rejected before creating any run folder.
Preparation requires normal parent completion, a successful guard exit and its
owned-empty-server shutdown receipt. The launcher additionally checks actual
CARLA processes and all three RPC ports before starting another server.

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

## Execution after the current batch finishes

```powershell
.venv\Scripts\python.exe scripts/evaluate_native_rule.py --prepare runs/20261024T075110Z_native_rollout_pair --source-method H_rollout32 --wall-seconds 1800
```

The command returns a new batch with `PREPARED_NOT_STARTED`. Use the existing
owned CARLA launcher and its protocol port, passing the existing isolated
CARLA interpreter through `NAVSTACK_CARLA_PYTHON`. Run the new batch's frozen
`source/scripts/evaluate_native_rule.py --batch BATCH`, record its actual process
identity and adopt it with `watch_existing_pair.py`, including the PowerShell
executable for owned cleanup. No launcher or worker has been started for this
diagnostic yet. Its initial 30-minute resource window, disk/RAM/GPU limits,
300-second no-progress guard and zero automatic retries follow the existing
lifecycle; a failed run is preserved rather than silently repeated.

At source `bee41c4`, five runner/cleanup tests pass locally and in the independent
checkout. They verify parent gating, preservation of failed fixture episodes,
applied-control accounting and rule computation timings, plus existing cleanup
failure receipts. The fixtures use lightweight dynamics and are not CARLA
performance evidence. Receipts are `runs/native_rule_runner_regression.xml` in
each checkout. No new scientific result is claimed from these tests.
