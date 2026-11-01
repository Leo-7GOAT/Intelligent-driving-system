# Shared terminal parking assistance: engineering validation

The frozen B1, H and strong-rule B3 each complete all **four known TRAIN roads**
with zero collision/bounds episodes when using the same explicit physical parking
aid. This is a single-seed development transfer, not research acceptance or proof
of a PPO advantage. The policies have not yet been trained under this wrapper.

| Method | Success | Controls | Parking overrides | Override fraction |
|---|---:|---:|---:|---:|
| Direct B1 | 4/4 | 1,409 | 6 | 0.43% |
| Hierarchical H | 4/4 | 1,287 | 8 | 0.62% |
| Strong-rule B3 | 4/4 | 1,286 | 8 | 0.62% |

The audit verifies identical parking-helper bytes, numerical calibration, vehicle
limits and goal thresholds across all three methods. Every physical frame is
joined to CARLA and its returned engine control. Every successful terminal frame
satisfies the original position, heading and speed conditions and is **not** an
intervened frame. No pose, velocity, goal or termination criterion is rewritten.
[Complete engineering audit](../reports/native_shared_parking_engineering_20261101.json).

`goal_parking_assist=True` enables `inside_goal_pose_overspeed_brake_v2`.
After actual travel exceeds the existing 0.1 m requirement, the aid may request
full physical braking only if the measured ego is already inside the existing
goal position/heading gates and still exceeds the existing goal speed. Steering
is unchanged. It releases control once speed is low enough; the next ordinary
policy/controller command is physically executed before the unchanged success
gate can accept the episode. All overrides and unassisted proposals are logged.
The option is shared by H, H0, B1, B2 and B3 and defaults to disabled, preserving
historical serialized configurations and signatures.

The first implementation, v1, mistakenly kept intervening after the car was
already parked. The existing `allowed_success=not intervention` guard therefore
blocked success. That run is retained: 1/4 success, three deadlines, 3,021 physical
controls. Its prepared H counterpart was abandoned before any lifecycle and the
launcher now rejects that marker. V2 changes the assistance release condition;
the original success guard remains unchanged. A regression test executes the
intervened braking frame and then the unassisted successful terminal frame.

This engineering result must stay separate from learning results. The preceding
unassisted B1 reflection fit was 3/4 with one bounds failure. A separately learned
177-parameter acceleration head passed its offline overspeed screen, but actual
deployment was only 2/4: it fixed one failure while introducing others. Those
failed models and data remain preserved. Earlier real PPO completed 10,240 new
physical training controls and 320 verified Adam steps but also remained 2/4.
Neither the failed neural fits nor this shared assistance establish added PPO
value. H also transfers from legacy async delivery to the validated synchronous
interface; its numerical gains are unchanged.

The B1 transfer uses model `72d465ab…`, with 30,720 actual PPO controls in its
lineage and separately disclosed supervised pose/reflection fits. H uses model
`c46da0e2…`, with 20,579 controls in its independent training lineage. Their
different histories mean this small transfer is **not** a budget-matched
hierarchical-advantage comparison. B3 performs no learned-policy updates.

All 7,003 controls across the three v2 arms and failed v1 run are included in the
engineering audit; 3,982 belong to v2. The same four conditions are reused for
development, not counted as twelve new conditions. No final set is activated.
Five independently trained healthy baselines, meaningful added PPO value,
explainable collaboration gains, remaining fair ablations and 120 new final
conditions are still required. A formal comparison must train related methods
with the shared wrapper fixed throughout training.

```powershell
python -m pytest tests/test_parking_assist.py -q
python scripts/analyze_native_shared_parking.py --output FRESH_ENGINEERING_AUDIT.json
```

The audit command requires the corresponding preserved run inputs. Existing
registered lifecycles must not be launched again. Public evidence and standalone
reproduction details are recorded separately when the capsule is published.
