# Shared measured actuator state — 2026-10-25

Research acceptance remains incomplete. No PPO model has yet been trained with
this interface. These four known training roads are development conditions.

## Completed diagnostics

Continuation B3 `20261024T110908Z_native_rule` completes3/4:2,010controls plus
123reset ticks. Road3 progresses13.24m and times out7.48m fromgoal. Raw audit
SHA256 `114e65701e8020801416d2744e8079fa3a5ce4945aff83681c6c76a26da7851e`
reproduces independently. Full decision P95 is1.74/4.86/1.55/10.00s byroad,
exceeding the0.5s period; this is not realtime.

Paired-low-speed B3 `20261024T112429Z_native_rule` completes0/4: all four45s
deadlines,3,600controls plus123reset ticks,934.28s walltime. Progress is
20.23/1.80/19.60/7.84m. Speed RMS .390/.403/.381/.387m/s; jerk RMS
172.67/64.68/168.10/119.25m/s³. Recorded plan/MPC failures and interventions
are zero. Current and independent-checkout raw audits agree byte-for-byte,
including original rule proposals and same-pose low-speed mappings. Both runs
finish normal owned-server cleanup, produce zero policy updates and fail the
unchanged4/4 fresh-training coverage gate. Lower speed alone did not fix it.

## New interface and prospective comparison

The existing measured feedforward + PI adapter improved fixed-reference tracking
in the [earlier diagnostic](actuation_curriculum_20261022.md). Native-road PPO
previously rejected it because the policy did not observe its internal state.
New explicit `actuator_observation=ff_state_v1` appends speed setpoint, signed
pedal, filtered measured acceleration and feedback integral after the road
raster. H/H0/B1 receive the same four inputs; H0 retains them when witness/outcome
feedback is ablated. Missing/nonfinite state fails. Begin, every control and the
terminal fence export state from the CARLA owner; raw audit binds it to the
same frame and reset acknowledgement. Offline inference uses recorded state.

The new4705-input policies normalize the four extra values; candidate/raster
encodings retain fixed scales. Both H and B1 actor/critic consume the new state.
Historical4701-input models remain loadable and disabled config serialization
is unchanged. Distinct signatures require fresh models/normalizers. Route
behavior history now records the selected ego-relative target rather than the
old ego-arc entry at the same index; this correction has its own signature tag.

Before running, fix a B3 comparison against continuation: native-score speed,
same four routes,45s deadline, vehicle limits, reward, planner, MPC, route catalog
and continuation penalty. Change the adapter to previously measured
`acceleration_ff_v2`, fixed firstgear, and expose the four-state interface. Reuse
exact calibration/hashed adjacent throttle fit from
`20261022T122427Z_native_curve_control`; require physical calibration equality.
No refitting on new outcomes. Record frozen sources and adapter/fit hashes.
The new observation/history fields do not enter rule selection. This is a
low-level engineering comparison, not learned-policy evidence.

Limits:30min,12GiB disk/4GiB RAM minimum,GPU85°C,no-progress300s,zero retries,
owned cleanup. Preserve all failures/frames. Only audited4/4 coverage can stage
fresh H0/H/B1seed11, initially20,480controls each, same adapter/inputs. The gate
is not relaxed. Five training seeds, healthy full-task B1, B2, interaction and
RL-value evidence, and fresh final confirmation remain required.

```powershell
python scripts/audit_native_rule.py --batch runs/20261024T112429Z_native_rule --output runs/low_speed_audit_rebuilt.json
python -m pytest tests/test_actuator_observation.py tests/test_road_candidate_policy.py tests/test_native_policy_sequence.py
python scripts/evaluate_native_rule.py --prepare runs/20261024T075110Z_native_rollout_pair --source-method H_rollout32 --catalog route_frame_terminal --continuation-feedback --calibration runs/20261022T122427Z_native_curve_control/calibration/calibration.json
```

Preparation is not launch. Actual run identity, tests and outcome go in handoff.

Root full regression after implementation:249passed,5conditional skips,
2existing deprecation warnings in101.35s; saved XML/log under
`runs/ff_state_full_regression.*`. Python3.7 parses the bridge/controller.

Source `dd7ea9b` is pushed. Independent checkout at that commit passes35targeted
tests in106.61s, including actual parameter updates/reload and observation
history reconstruction. Actual B3 batch `20261025T070545Z_native_rule` starts
after matching0.9.15 empty-world readiness; first road succeeds341controls with
speedRMS.11327m/s and jerkRMS44.5766m/s³. Other roads are still running at15:10;
this is not yet an audited complete result. No new PPO has been launched.

At15:19, roads0/1/2 succeed in341/350/317controls, speedRMS
.11327/.08790/.10131 and jerkRMS44.5766/56.8427/59.8057. Road3 remains live;
partial records show waiting. Root OS identity and healthyguard are revalidated;
deadline stays15:35:45. These partial observations do not pass thecoveragegate.
Independent CPU tests overlapped some early evaluation; walltime/cycle costs
are descriptive host-load measurements, not controlled realtime benchmarks.
