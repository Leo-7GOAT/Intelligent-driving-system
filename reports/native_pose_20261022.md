# Native-road actuation and terminal-pose diagnosis — 2026-10-22

## Scope and decision

Real CARLA 0.9.15 native-road fixed-reference MPC has now executed four geometrically selected development routes. This is not PPO or Hybrid A* performance evidence. Both attempted actuator changes fail to resolve the measured comfort problem and remain opt-in diagnostics. No acceptance threshold is lowered, and no result supports architecture superiority.

## Actual fixed-reference comparison

All runs use the same four geometry IDs selected from the preceding native survey, the same MPC horizon20, dt0.05s, target speed1m/s and45s deadline. Reference arrays x/y/yaw/s/kappa/v/a/t agree within1e-6 absolute (only reset numerical noise). Goals require position≤0.8m, speed≤0.25m/s and heading≤0.35rad, with collision/offroad failure taking priority. Route heading changes are approximately−30,−60,+30,+60degrees. Twenty-four waypoint advances do not imply exactly24m: actual reference lengths are22.16–25.83m.

| Variant / run prefix | Completion | Control ticks | Per-case speed RMS (m/s) | Per-case jerk RMS (m/s³) |
|---|---:|---:|---|---|
| Legacy / 20261022T112622Z | 4/4 | 1932 | .364, .366, .353, .369 | 131.40, 140.05, 141.62, 130.12 |
| Mild coast / 20261022T114902Z | 4/4 | 1932 | .438, .446, .437, .440 | 134.02, 131.19, 136.88, 133.96 |
| Fixed first gear / 20261022T115357Z | 4/4 | 1942 | .336, .304, .330, .296 | 138.70, 134.05, 142.69, 134.21 |

Each run directory ends in `_native_curve_control`. All12episodes have zero observed collision events, zero sampled-footprint departure frames and zero MPC solve failures. Twelve footprint samples per actual frame are not a continuous geometric safety proof. Legacy lateral RMS is .0159/.0419/.0187/.0396m; fixed-first-gear is .0167/.0442/.0172/.0474m. Keeping good lateral tracking does not establish good longitudinal control.

Legacy completion times25.15/25.95/23.30/22.20s; coast25.00/26.15/23.30/22.15s; first gear25.20/26.05/23.45/22.40s. Actual deceleration minima remain approximately−22 to−25m/s². Number of frames below−5m/s²: legacy41/42/38/36; coast39/40/32/31; first gear38/44/40/42. Every spike and failed attempt is retained. Full control-cycle latency distributions are saved per case; baseline/coast have some50ms overruns. This is not a hard-real-time claim, and different concurrent CPU load prevents attributing latency differences solely to the actuator.

The coast option suppresses positive brake only for mild requested deceleration[-1,-.05)m/s², measured speed(0,2)m/s and speed setpoint>0.1m/s. Strong braking, overspeed and final hold remain. It worsens speed RMS and does not consistently reduce jerk, so it is not adopted. The first-gear option changes only drivetrain mode, preserving legacy throttle/brake formulas; it reduces speed RMS on these four routes but leaves sharp braking unresolved. Neither change has been used for PPO training or promoted as a validated calibration.

## Raw actuator evidence

The legacy batch contains `brake_probe`:20raw-control phases,1016controlticks including acceleration to measured initial thresholds0.5/1/1.5/2.5m/s, followed by brake0/.01/.04/.085/.2. No forced velocity or execution-time transform. All measured controls have hand_brake=false. Positive raw brake can produce approximately−25m/s² at low speed; zero-brake coast does not show that drop. These are transient threshold crossings, not equilibria or a proven simulator-internal diagnosis.

The coast batch contains two complete throttle probes,720ticks each:80constant-throttle ticks then40coast ticks for throttle .02/.04/.06/.08/.10/.12. Automatic transmission remains gear0 and does not move during the4s input. Fixed first gear produces end-of-input speeds .053/.065/.080/.331/.706/1.033m/s. This isolates startup gear behavior; it does not solve brake discontinuity. One automatic .10 phase reports the preceding reset's brake/handbrake on its first returned control frame; that observation is retained, not called a clean all-false result. The bridge now logs returned applied-control fields as well as requested raw controls so their frame delay is visible.

Owned CARLA launcher11116 was reused between the two later batches without another tick owner. After all probes and cases, an empty-world/version/OS-identity check stopped the owned shipping process; the stop receipt is in the115357Zbatch. Historical112622Zhas its own stop receipt. No user process was stopped.

## Learning endpoints and new hypothesis

v16 B1 completes3,001,791physical steps/1175optimization phases/32,781optimizer steps, exact model reload,0/10at1M/2M/3M. At3M six noncollision episodes reach position tolerance and stop but retain heading error0.60–0.88rad, above the unchanged0.35radgate; four narrow/unseen episodes collide. Their actual physical traces and terminal audits remain local.

v13 B1hold1 safely resumes its clean scheduled window end at898,560steps and finishes1,000,960steps/391phases/11,186optimizer steps, exactreload,0/10(all deadlines). Supervisor exits0; no completed H/H0 job is rerun. The continuation helper can select jobs and explicitly accept a verified scheduled window end, but refuses failed jobs, manual STOP and resource STOP. It preserves frozen source/config/RNG/optimizer/normalization and inherited counters.

Fresh v17seed11 tests only added terminal-pose PBRS: Phi_heading=-0.1*(1-cos(yaw-goal_yaw))*exp(-distance/6m), inside existing physical-time/terminal-zero shaping. Base reward,605inputs,MLP,PPO,vehicle,curriculum and development cases matchv16. No helper controls or relaxed goals. Offline replay of all ten v16final traces verifies the potential identity; discounted-return differences are numerical zero, not a new policy score. Disabled shaping preserves old checkpoint signatures.

v17is actually training toward its registered3Mbudget in a two-hour guarded window. At1M it scores1/10(one unseen7302success, eight deadlines, one unseen7301collision);100kand300kare0/10. This small single-seed result does not establish improvement. Source is frozen at launch and is unaffected by subsequent actuator edits. Five independent policies, healthy matched B1, strong-B3 learning-value evidence, full ablative comparisons, native-road PPO observations/rewards and fresh120-condition CARLA confirmation remain unfinished.

## Reproduction and evidence locations

All three original run datasets are now published as the separate `native_actuation_diagnostics.zip` asset on the existing development pre-release. Earlier model/video assets are unchanged. The new ZIP contains559hashedfiles,3,909,935bytes; SHA256`5f8822bc6caca88429e5830f829b916352c8e6588f0debe637b69399a65160a5`. GitHub reports uploaded with that digest, and an actual download verifies every member. Each contains frozen source hashes, calibration, protocol, all case traces, scene/reference/terminal state and timing summaries. `analysis/verified_summary.json` recomputes sensor/world frame matching, timestamps, segment-projected tracking errors, jerk and velocity derivatives. First derivative sample is omitted, never zero-filled; episodes are not padded after termination.

```powershell
python scripts/analyze_native_control.py --batch runs/20261022T112622Z_native_curve_control
python scripts/analyze_native_control.py --batch runs/20261022T114902Z_native_curve_control
python scripts/analyze_native_control.py --batch runs/20261022T115357Z_native_curve_control
python scripts/compare_native_control.py --baseline runs/20261022T112622Z_native_curve_control --candidate runs/20261022T114902Z_native_curve_control
python scripts/compare_native_control.py --baseline runs/20261022T112622Z_native_curve_control --candidate runs/20261022T115357Z_native_curve_control
```

To create a new actual native diagnostic, first obtain a local survey/calibration batch using the existing steering/survey entry points. Then run `python scripts/native_curve_control.py --prepare SURVEY_BATCH --adapter legacy` (or the explicitly experimental`coast_mild_v1`/`manual_first_v1`). The returned batch freezes code and a900s deadline. Verify exclusive server readiness, set NAVSTACK_CARLA_PYTHON to the isolated3.7interpreter, and run that batch's `source/scripts/native_curve_control.py --batch BATCH`. Use the existing identity-checked stop script after empty-world verification. Preparing a batch is not a measurement; running it requires CARLA.

Final local and independent checkout CPU regression both pass102tests with5conditional skips. Remote code067db54 is verified. All three downloaded-data figures and summaries reconstruct byte-identically. The independently installed environment reloads the existing145-file camera model asset with frozen normalization. A fresh actualCARLA replay from that checkout completes4/4native routes/1924controlticks, again with severe deceleration/jerk; detailed trajectories differ, so only categorical outcomes reproduce. This is repeatability, not a new independent condition/seed. Owned server was verified empty and stopped.

At20:09, v17reaches2M and scores0/10(nine deadlines, one collision), regressing from1/10at1M. Both empty cases stop3.09–3.12m short, with heading error~0.35rad. Their complete physical-frame terminal audits are retained. The registered3Mtraining continues; no endpoint is selected retrospectively.
