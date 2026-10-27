# Registered native CARLA execution

Use PowerShell7, the existing isolated main Python and the installed CARLA0.9.15
client/runtime. Preparation prints a new batch path; it does not execute CARLA.
Replace `runs/PREPARED_BATCH` below with that printed path. Inspect the current
[handoff](handoff.md) before preparing anything: an already registered/started
phase must be continued through its recorded recovery child.

```powershell
pwsh -NoProfile -File scripts/run_registered_native.ps1 -Batch 'runs/PREPARED_BATCH' -ValidateOnly
pwsh -NoProfile -File scripts/run_registered_native.ps1 -Batch 'runs/PREPARED_BATCH' -ActivateQueued
```

The first command checks frozen source, calibration, existing lifecycle evidence
and the prepared protocol hash. It does not probe a server or claim readiness.
The second starts an owned hidden server, requires an empty matching-version
probe, identifies the actual Python worker and starts the existing resource
guard. Its environment supplies the absolute client path to both subprocesses.
On normal exit the guard verifies empty-world owned-server cleanup. Failures
retain their logs. Cleanup refuses live bridges, nonempty worlds or unverified
ownership and never escalates to an unrelated process kill.

`-ActivateQueued` is only for a never-started prepared phase. It preserves the
original protocol/preparation bytes and records their hashes, then starts the
same registered wall-time allowance at activation. It changes only the two
resource clock fields. It does not change model, task, seed, budget, controls or
success gates. An exclusive file handle prevents simultaneous launchers.
`-ActivateQueued -ActivateOnly` activates a queue entry without starting a server;
normally use the combined command above. Repeated activation is rejected.

For a registered rule diagnostic, add `-Kind rule`. Explicit runtime overrides
are `-Python` and `-CarlaPython`. The default main runtime is `.venv`; the default
client is `.carla-python37/runtime/python.exe` or `NAVSTACK_CARLA_PYTHON`.
No reinstall, environment upgrade or old-plant restoration occurs.

After normal completion, verify actual frames separately:

```powershell
python scripts/audit_native_method_endpoint.py --batch runs/COMPLETED_BATCH --method B1
```

Use `H`/`H0` for their registered methods. Rule diagnostics use
`python scripts/audit_native_rule.py --batch runs/COMPLETED_BATCH --output runs/NEW_RULE_AUDIT.json`.
Audit outputs must be new. A successful launcher is lifecycle evidence, not a
substitute for raw frame verification, model evaluation or research acceptance.

The wrapper changes process orchestration only. Simulation action frequency is
not a wall-time guarantee; cycle P95/P99/max and missed periods remain separately
measured. The current experiments are single-seed known-road development, with
all failed and interrupted phases retained.

Frozen scene-family development probes use completed, raw-audited PPO endpoints
and a completed, raw-audited rule batch as the scene reference:

```powershell
python scripts/prepare_native_scene_transfer.py --parent runs/COMPLETED_PPO_BATCH --reference-rule runs/AUDITED_RULE_SCENE_BATCH
```

Launch its printed prepared batch with the same registered wrapper, after the
active owned lifecycle finishes. Preparation preserves all four route geometries,
copies the exact final model/normalizer/RNG, retains inherited interaction costs,
and removes auxiliary replay. Only the registered actor scene changes; training
and optimizer budgets are zero. Its actual CARLA controls are evaluation costs.
An existing registration is reused through its recorded batch, never silently
prepared or driven again. This probe uses inspected development scenes. The rule
reference supplies scene provenance, and can have an older execution version;
its success rate alone is not a matched architecture comparison.

For an explicit frozen compute probe on the original actor scene, use the same
preparer with `--failure-memo` and an audited reference rule with that same scene.
The preparer checks that execution-source changes consist only of the memo hooks
and trace metadata, and that the geometric backend files are unchanged. This
does not retrain a policy or modify an already registered batch.

Fresh `prepare_native_hazard_learning.py` jobs also accept `--failure-memo`.
The registered `exact_static_failure_v1` cache uses exact float64 start/goal
inputs, current geometry content, vehicle and planner settings. It is bounded
to 128 entries, skips dynamic and wall-clock-limited searches, and stores only
failed deterministic searches. Geometry or settings changes invalidate entries.
A cache hit records zero newly expanded nodes plus the original search node
count separately. It does not establish physical infeasibility or change actions,
rewards, trajectory generation, control or success gates.

Offline component reproduction, requiring an already audited complete endpoint:

```powershell
python scripts/benchmark_native_failure_memo.py --batch runs/COMPLETED_BATCH --method H --output runs/NEW_MEMO_BENCHMARK.json
```

This uses saved CARLA states without new driving. Lookup timings exclude the
initial full search; they are not end-to-end latency or realtime evidence.

Supervised initialization diagnostics can explicitly vary collection length:

```powershell
python scripts/prepare_native_supervised_ppo.py --parent runs/AUDITED_TEACHER_ONLY_BATCH --guarded-progress --learning-rate 3e-5 --rollout-actions 1280
```

This requires untouched PPO optimizer/interaction state and rebuilds an empty
buffer without changing policy tensors. The new length must change, remain a
multiple of the saved minibatch, and divide the registered physical budget along
with the old length. For hold1 B1, 320-to-1280 collection with minibatch320,
10epochs and20,480controls gives64-to16training phases and640Adam steps in either
setting. The scope is collection/update timing; it does not guarantee valid
bootstrap values or better driving. All original cases remain in the endpoint.

The launcher treats JSON `CONNECTION_FAILED` as still starting within its bounded
readiness loop, including when the probe process exits0. A connected incompatible
or nonempty world still fails. A verified startup-only failure can be preserved
and staged once into a fresh child:

```powershell
python scripts/retry_unstarted_native.py --failed runs/FAILED_STARTUP_BATCH
```

This is restricted to the recorded timeout classification, no worker/bridge,
matching-version empty cleanup and absent original owned processes. It copies
saved policies and runtime source unchanged. It never reuses a failed lifecycle
directory or treats a partially driven episode as zero controls. The original
failure and the one-child registration remain available for inspection.
