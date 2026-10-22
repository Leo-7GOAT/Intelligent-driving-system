# Acceleration feedback and route curriculum — 2026-10-22

## Actual control result

The legacy mapping applies brake whenever requested acceleration is below -0.05 m/s². Actual CARLA frames showed large braking for mild deceleration, whereas raw coasting could slow the vehicle naturally. The upstream [low-speed braking issue](https://github.com/carla-simulator/carla/issues/3906) corroborates similar reports, including 0.9.15; it does not prove the internal cause here. CARLA's [0.9.15 Ackermann source](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/AckermannController.cpp) also uses acceleration feedback. This project implements its own measured-acceleration controller; it does not replace physical execution with autopilot or a separate simulator.

Two new frozen batches execute all four registered native references at the same target speed1m/s, with unchanged MPC, vehicle, goal gates and failure priority:

| Version | Run prefix | Completion / ticks | Speed RMS per route (m/s) | Jerk RMS per route (m/s³) |
|---|---|---|---|---|
| Legacy | 112622Z | 4/4 / 1932 | .364, .366, .353, .369 | 131.40, 140.05, 141.62, 130.12 |
| Acceleration PI | 121852Z | 4/4 / 1932 | .514, .536, .478, .532 | 102.71, 96.39, 100.50, 106.78 |
| Measured feedforward + PI | 122427Z | 4/4 / 1922 | .135, .085, .103, .188 | 44.56, 55.46, 45.42, 74.53 |

Run names start `20261022T` and end `_native_curve_control`. All eight new episodes have zero observed collisions, sampled footprint departures and QP failures. Reference arrays agree within1e-6 absolute; unchanged controller/trajectory/vehicle source and physical calibration are checked. Actuator configurations are recorded separately. These are development comparisons without independent repetitions or a population confidence interval.

PI alone is too slow, causing overshoot and emergency braking. For feedforward, `fit_carla_throttle.py` fits throttle-only dynamics from PI cases0–2, using0.2s measured velocity differences at0.3–2m/s. It excludes the0.6s after a brake command from model fitting only; all such frames remain in full performance statistics. The fitted model is `a = 7.3265144 * throttle - 0.0838300 * speed - 0.6463184`. RMS fitting errors are .157/.143/.144m/s²; case3, held out from fitting, has .132m/s² over229 selected windows. This is development model checking, not independent confirmation. Calibration records coefficients, source trace hashes and fit-file hash.

Feedforward reduces frames below -5m/s² from41/42/38/36 to4/2/3/7. Completion times are24.95/25.85/23.10/22.20s. Remaining problems are explicit: minimum acceleration still reaches -18.74/-17.52/-18.34/-24.04m/s², with startup and final-braking spikes. Lateral RMS becomes .0188/.0636/.0214/.0626m, worse on the larger turns than legacy .0159/.0419/.0187/.0396m. Every frame is retained; episodes are not padded after termination. No hard-real-time or PPO benefit is claimed.

Both alternatives use fixed first gear and export filter/integrator state. Default actuation remains unchanged. Ordinary training episodes reject these modes until internal-state observations are integrated; native diagnostics are explicitly permitted. Emergency requests at or below -2.4m/s² retain direct braking; final hold and overspeed braking remain. A training version needs shared H/B1 exposure and matched adaptation. The first prepared batch121810Z ran zero CARLA ticks because a preflight test incorrectly used exact float equality; its not-executed receipt is preserved. Corrected tests and fresh batches were used. The owned server was verified empty and stopped after the actual runs.

## Learning evidence and new experiment

v17 completes3,001,352 physical steps,1174 optimization phases and32,033 optimizer steps, with exact reload and supervisor exit0. Final development performance is0/10: eight deadlines, two unseen collisions. Empty cases stop1.53m/.86m short with heading errors .52/.67rad; dynamic7301 enters position tolerance but fails yaw .76rad. Its transient1/10 at1M regresses to0/10 at2M and3M. The last100 stochastic training episodes contain only2 successes. Heading PBRS is not established as an improvement.

Fresh v18 uses v16's605-input MLP, hold10, PPO/gSDE and progress weight0.25, with heading PBRS disabled. It replaces300k full-length empty pretraining with fixed stages:6m/12m/18m/23m empty tasks for1M physical steps each, then1M mixed tasks,5M total. Stage switches occur only at reset using accumulated physical counters. There is no teacher, validation-dependent promotion or hidden warm-start budget. The same schedule is available to H/H0/B1/B2/B3; this single-seed diagnostic is not a fair architecture comparison by itself.

Active batch: `runs/20261022T123453Z_route_stages_v18_development`, job `route_stages_v18/B1_route_stages_seed11`, deadline about22:34 local. One CPU worker;12GiB disk/4GiB RAM/85°C GPU/no-progress guards; no automatic failure retry. Launch resources: about65.7GB disk free, GPU37°C. Frozen source and canonical v16 comparison are verified. Worker11116/supervisor7088 were verified by OS identity; old PID45864 was reused by an unrelated browser and left untouched.

At approximately135k steps, optimization evidence records2052 optimizer steps and actor-update L2 .177. Recent100 training episodes had80 successes, but the frozen100k six-metre diagnostic is deterministic0/8 versus stochastic7/8 for both train/validation starts. Those starts duplicate the same simple geometry and are not32 independent conditions. The inference audit now reads saved gSDE resampling frequency (1 action for hold10), rather than hard-coding10. Deterministic failures remain in the report. The unchanged ten full-task cases remain milestone evaluation.

## Reproduction

Independent checkout at published commit `abb2b1a` also passes105 tests/5conditional skips, reloads the published model, and reconstructs the two downloaded-data figures/summaries byte-identically. Its actual CARLA feedforward replay completes4/4 routes/1920 control ticks; speed RMS is .105/.148/.105/.166m/s, jerk68.53/77.73/67.94/71.95m/s³. The improvement relative to the original control persists, but detailed outcomes are not bitwise deterministic and braking spikes remain. Owned server cleanup and shutdown are verified.

Final local CPU regression passes105 tests with5conditional skips. At the1M checkpoint, six-metre deterministic inference improves to8/8 and stochastic inference is6/8 (same outcomes for the duplicate train/validation geometry). Full-task evaluation remains0/10: two empty deadlines and eight obstacle collisions. Training has entered the12m stage. Short-route competence is not mixed-task competence.

```powershell
python scripts/analyze_native_control.py --batch runs/20261022T121852Z_native_curve_control
python scripts/analyze_native_control.py --batch runs/20261022T122427Z_native_curve_control
python scripts/compare_native_control.py --baseline runs/20261022T112622Z_native_curve_control --candidate runs/20261022T122427Z_native_curve_control
python scripts/initialize_route_stages_development.py
```

The initializer writes `runs/route_stages_development.json`. Start `scripts/development_watchdog.py --batch BATCH_PATH` using its returned path. Do not initialize again when resuming the existing active batch. To recompute the fit, use a fresh copy without the derived `analysis/throttle_fit.json`; the fitter refuses to overwrite it.

The two complete control datasets are published as `acceleration_feedback_diagnostics.zip`:377 manifest files,2,487,456 bytes, SHA256`c1f4cedb03d085b476c540e69e53a501e93dff0c3087186e4fb3d82c4226d6ba`. GitHub uploaded digest and an actual download match; every member hash verifies. Both variants and their failures are retained; the earlier diagnostic asset is unchanged. Five-seed comparisons, healthy mixed-task B1, native-road PPO integration and fresh final CARLA conditions remain unfinished.
