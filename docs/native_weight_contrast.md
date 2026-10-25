# Native actor-weight contrast

Status on 2026-10-25: implemented and locally regression-tested; **actual
CARLA contrast not yet executed**. The original H0/H/B1 training queue still
owns CARLA. Do not start another server or bridge on its RPC ports.

The completed H0 endpoint scores 3/4 on four known training roads. Its actor
probabilities remain close to uniform, so those outcomes cannot establish that
training improves on initialization. `scripts/evaluate_native_policy_weights.py`
provides the missing actual-execution control. For a selected fresh H or H0
endpoint it evaluates all four original roads twice, using the reconstructed
seeded initial weights and the trained weights. Both arms use the **same saved,
frozen training normalization statistics** and the exact parent low-level code,
calibration, task and categorical action semantics. There are no PPO updates.

This estimates a weight-training contribution conditional on the trained
normalizer. The initial arm is not a completely untrained system with fresh
statistics, a different training seed, or a held-out generalization test. It
does not measure an average treatment effect across independent policies.
The prior contribution from route-tangent candidates, Hybrid A*, MPC and
actuator adaptation remains present in both arms. H versus H0 still requires
its own matched training comparison; they have separately learned normalizers.

## Preparation and execution

Only prepare when the entire parent has `summary.json`, successful
`guard_final.json`, `infrastructure/server_stop.json` confirming an owned empty
server shutdown, and a reproducible endpoint raw audit. Preparation refuses an
active or failed parent. It freezes the parent source, model, configuration,
normalizer and input hashes. The only harness overrides are the new evaluator,
its auditor and the guard worker allowlist. No parent `navstack` source changes.
The wall window starts at preparation; launch promptly after preparation.

From the repository environment, after normal parent cleanup:

```powershell
.venv/Scripts/python.exe scripts/evaluate_native_policy_weights.py --prepare runs/PARENT_BATCH --method H --wall-seconds 7200
```

Use the printed new batch directory in the existing owned CARLA lifecycle:
`scripts/run_carla_owned_foreground.ps1` with `-RpcPort 2100
-RenderOffScreen -OutputDirectory runs/NEW_BATCH/infrastructure`; verify the
isolated 3.7 matching-version empty-server probe. Preserve
`NAVSTACK_CARLA_PYTHON` pointing to the existing isolated client. Start this
frozen worker, record its real Python PID/executable/creation identity as
`worker_process.json`, and attach the frozen `watch_existing_pair.py` guard
with the existing PowerShell executable and owned-server cleanup. The guard
supports this evaluator and observes each arm's heartbeat. It checks the same
disk/RAM/GPU/no-progress/wall limits, with zero automatic retries.

```powershell
.venv/Scripts/python.exe -u runs/NEW_BATCH/source/scripts/evaluate_native_policy_weights.py --batch runs/NEW_BATCH
.venv/Scripts/python.exe runs/NEW_BATCH/source/scripts/watch_existing_pair.py --batch runs/NEW_BATCH --identity runs/NEW_BATCH/worker_process.json --powershell POWERSHELL_EXECUTABLE
```

The worker and guard run concurrently, as in the training lifecycle. Do not
execute the worker twice or remove a live lease. A stopped/partial batch stays
in the archive; it is not silently converted into a complete paired result.

After this new batch itself finishes and the guard confirms owned-server cleanup:

```powershell
.venv/Scripts/python.exe runs/NEW_BATCH/source/scripts/audit_native_policy_weights.py --batch runs/NEW_BATCH --output runs/NEW_BATCH/raw_audit.json
```

The auditor checks all eight outcomes, every applied CARLA state/command/frame,
terminal fence, initial actuator history, reset/control/world-tick accounting,
all saved-input hashes, and every deterministic policy action reconstructed
from measured state and observable history. It reports the actual paired reset
state deltas and all task metrics. It rejects extra resets, missing cases,
truncated bridge/trace streams, unaccounted controls and incorrect arm labels.
Policy inference and full decision costs are reported separately. Failures are
retained; successful-case conditional metrics keep their existing labels.

## Completed local verification

Targeted evaluator/auditor/owned-process/previous-rule regressions: **32 passed**.
Tests include real lightweight vehicle execution plus explicitly fabricated
evidence rejection fixtures; neither is presented as a CARLA contrast result.

An additional read-only run used the actual completed H0 model and exact frozen
parent source. Initial-to-trained actor L2 is **0.0782405436038971**, matching
the real training log. Saving/reloading the reconstructed initial model is
parameter-exact. Under the common final H0 normalizer, initial selections on
recorded initial states are **22/0/30/29**, trained selections **22/22/30/30**.
These are four initial decisions, not four executed untrained outcomes.
Normalization stayed fixed and there were zero new vehicle controls or updates.
Local evidence: `runs/native_weight_initialization_20261025_v2/verification.json`,
with frozen initializer/verification source and the initial model alongside it.
