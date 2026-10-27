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
