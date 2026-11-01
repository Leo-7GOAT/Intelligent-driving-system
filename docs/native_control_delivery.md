# Versioned native control delivery

The previous frame audits established measured CARLA states,requests and bridge
echoes. They did not compare the engine's returned last-tick vehicle control to
the requested throttle/brake/steer. A new independent check finds51mismatches in
12,219controls from the completed exploration pilot:47TRAINand4evaluation.
Every mismatch equals the previous request. This is an acknowledgement timing
discrepancy;the data do not establish its causal contribution to policy failure.
The original physical outcomes,models,raw logs and published capsules are retained.

`recommended_adapter.control_delivery=batch_sync_v1` now sends ego control through
CARLA0.9.15`apply_batch_sync([ApplyVehicleControl(...)],False)` before the owner
issues its one physics tick. The batch response must succeed. Returned engine
pedals,gear,handbrake and reverse must match the requested control within float32
wire tolerance. An inconsistent frame is preserved by the raw reader and then
aborts the main client. The sync call never adds a physics tick. Legacy mode is
retained explicitly for historical compatibility;it is not silently rewritten.

The new frozen transfer changes only command delivery,with the same4f205708...
replay model,normalizer,environment,vehicle gains,goal gates and four original
TRAINroads. All other parent source hashes match except the three declared
delivery overlays. It performs no PPOor auxiliary fitting.

| Frozen reference | Controls | Returned-control mismatches | Successes | Bounds |
|---|---:|---:|---:|---:|
| Historical async delivery | 1,275 | 5,all matching previous request | 3/4 | 1 |
| Explicit synchronous delivery | 1,271 | 0 | 3/4 | 1 |

The same case1fails at264/256controls. The command-correspondence check passes
for all1,271new controls,but the unchanged basic driving gate still fails.
This is low-level frozen-policy transfer,not improved PPOlearning or a qualified
baseline. Future training using this interface must declare the new low-level
version. Dynamic actor delivery and real-time performance were not established.

[Legacy pilot discrepancy](../reports/native_PPO_legacy_control_delivery_20261101.json),
[checkpoint/task-matched transfer](../reports/native_sync_control_delivery_transfer_20261101.json).
The new raw transfer data remain local at this snapshot. The existing public
PPOcapsule still contains the complete original measurements,including its
engine controls,so its51mismatches are independently inspectable.

```powershell
python scripts/audit_native_control_delivery.py --batch RESTORED_PPO_BATCH --output FRESH_DELIVERY_AUDIT.json
python scripts/audit_native_control_delivery.py --compare LEGACY_REPLAY_BATCH --batch SYNC_TRANSFER_BATCH --output FRESH_TRANSFER_COMPARISON.json
python -m pytest tests/test_control_delivery.py tests/test_carla_actuation.py tests/test_actuator_observation.py -q
```

The fixed-reference TRAINinput diagnostics also preserve their limits:
[naturally changing witness bits](../reports/native_B1_witness_jump_diagnostic_20261101.json)
and [clock-only perturbations](../reports/native_B1_clock_sensitivity_20261101.json)
change no model weights or CARLAstates. They are sensitivity measurements,not
driving improvements. No observation mask or new fit was adopted from them.
