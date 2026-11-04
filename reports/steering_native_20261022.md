# Actual CARLA steering and native-road geometry audit

Research acceptance remains incomplete. This is low-level diagnostic evidence, not PPO evaluation, new independent training seeds or final road-condition coverage. The existing training jobs keep their frozen source and actuator mapping.

## Actual actuator measurements

Batch `runs/20261022T111210Z_carla_steering_audit` used the installed CARLA0.9.15 server and matching isolated3.7client. The fixed22phases cover raw normalized steering0,±0.1,±0.25,±0.46,±0.75,±1.0 at requested speed0.5/1.5m/s. Every phase starts a fresh owned vehicle, settles30ticks, then applies60straight,80steering and20braking control ticks. Summary measurement uses the predefined final40steering ticks. This directly applies raw CARLA actuator controls; no PPO, planner or MPC is represented as producing these commands.

All22phases completed:3,520control ticks,4,201world ticks including resets/inter-phase cleanup before the final cleanup tick. No collision events were observed. Actual world transforms, longitudinal speed, front wheel angles and equivalent bicycle steering were recorded. The IMU and world frame match on every control row; frames are consecutive within every phase and physical timestamps advance0.05s. Rebuilding all phase means/errors from raw rows passes. These are small bounded control probes, not a safety result.

| Requested raw steer magnitude | Largest absolute error against existing mapping, sampled operating domain |
|---|---:|
| Inputs through±0.46 | 0.01346rad over measured rows |
| All sampled inputs through±1.0 | 0.03250rad over measured rows |

The old mapping is `delta=-1.0819452938455352*normalized_steer`. Its sampled error within the policy's approximately±0.462command domain is small relative to the0.5rad physical steering limit. This weakens a gross steering-scale mismatch explanation; it does not establish dynamic tracking accuracy or rule out delay/model errors. The mapping remains unchanged. Values between samples and higher speeds were not validated by this sweep.

The requested0.5m/s phases actually average0.290–0.316m/s during the measurement window; requested1.5m/s phases average1.445–1.510m/s. The plot labels setpoints and measured values separately. In particular this is not evidence of precise0.5m/s velocity tracking. The probe's speed feedback is a calibration command, not a retuned production actuator or a learned policy.

The first attempt, `20261022T111101Z_carla_steering_audit`, failed before connecting or spawning because embedded Python does not include the script folder on its import path. It requested zero physics ticks. Its frozen source and failure record remain. A new frozen batch used package-qualified imports and reused the already-owned, verified-empty server without restart. There was no automatic retry loop. After the successful sweep, owned actors were destroyed, empty-world status was independently probed, and the identity-checked shipping process46056 was stopped. Receipts are in `cleanup.json` and `infrastructure/server_stop.json`.

## Native-road geometry candidates

The read-only installed `Town10HD_Opt` map survey examined824starts spaced8m along native road waypoints. Each candidate follows24one-metre waypoint steps with a deterministic branch rule. Predeclared geometry filters require15–90degrees total heading change, curvature within `tan(0.5)/3.00464556`, height range≤0.5m, and12sampled vehicle-footprint points on the driving surface at each path pose. It rejected479starts by heading/curvature and7by sampled footprint, retaining334distinct sampled paths after exact rounded-coordinate deduplication. No world tick or actor creation was requested by this survey.

The paths can overlap and are not334independent test conditions. Footprint sampling is not continuous swept-body collision proof. Road occupancy, traffic behavior, native-road PPO observations/rewards and actual controller execution remain unverified. These candidates supply measured native geometry for the next fixed-reference control/scene-interface work, rather than claiming that existing straight-road obstacle tests were native bends.

## Reproduction and evidence

`responses.jsonl` SHA256: `ced84977fb996a2e5055b8595b2f2103eb89190b2271115b10d09450e07bed93`.
`summary.json` SHA256: `00f9fa23075f61713eec33467269415d77f6246f061bc064ce12fea4cc962cb2`.
`native_routes.json` SHA256: `0be11c63db07494f126f6e01912e2a53305077e09b089cb718c1dca328155ffe`.
Source/calibration hashes, actual frame IDs, map/OpenDRIVE identity and the exact plotted geometry IDs remain with the batch. Raw files are local development outputs, not yet a published model/data asset.

```powershell
.\.venv\Scripts\python.exe scripts/plot_steering_audit.py --batch runs/20261022T111210Z_carla_steering_audit
```

This rebuilds and verifies the physical-frame summaries before rendering `steering_native_audit.png`. The figure was visually inspected. To acquire a new diagnostic, use `carla_steering_audit.py --prepare <calibration-batch>`, start the existing installation with `start_carla.ps1`, then run the generated frozen `source/scripts/carla_steering_audit.py --batch <batch>` with the isolated3.7client. Preparation alone does not run CARLA. The15minute deadline,12GiBdisk/4GiBRAM/85CGPU guards and no automatic retries are registered in the protocol. Only this task's actors/server may be cleaned up.

Current full regression:93passed,5conditional skips. The small synthetic summary test validates analysis/sign/window logic only; the actual3,520CARLAframes above are separate evidence. B1 cumulative2M/3M training and the v13hold1 run remain separate ongoing experiments, not replaced by these low-level diagnostics.
