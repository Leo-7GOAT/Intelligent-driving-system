# Native recovery and flat-policy stability — development evidence

Research acceptance remains incomplete. These are one-seed diagnoses on previously inspected development conditions, not independent confirmation or evidence of hierarchical superiority.

## Interrupted CARLA training and charged costs

The previous native batch `20261022T152120Z_native_road_learning` stopped without a final H model or B1 start. H has 94,329 readable control frames and a truncated gzip footer/tail. Its last complete optimization is at 94,079 controls; its latest retained model is at 30,080. The original logs are preserved without repair. The process-death cause is unknown; the machine's last boot predates this interruption.

H0 completed 100,160 controls / 103,631 training world ticks / 626 optimizer steps and exact reload; all four training-road probes time out. It is retained and not rerun.

`scripts/audit_interrupted_native.py` records readable-prefix counts, errors and checkpoint hashes. `scripts/prepare_native_recovery.py` registers one recovery child per parent and refuses duplicate preparation. Batch `20261023T010623Z_native_recovery` resumes H's learning state from 30,080, starts a new measured CARLA episode, and budgets 69,920 additional controls before a fresh B1 100k run. The old checkpoint lacks RNG state, so seed 11 is explicitly restarted; this is not seamless simulation continuation.

Every new 10k threshold now saves model, normalization, optimizer-containing SB3 state and RNG state in a partial directory before publishing a complete directory and latest pointer. Saved files have hashes. Retained model lineage and charged interaction costs are separate: H inherits at least 94,329 consumed controls, including 64,249 not represented in the restored model. The unflushed tail is unknown. This recovery is **not budget-matched superiority evidence**.

The new server and bridge use the existing CARLA 0.9.15 installation/client. Server ownership belongs to this recovery batch; never apply the previous batch's PID cleanup. At 09:25 local H had 49,580 new controls / 154 optimization phases and was OS-confirmed live. These are a timestamped snapshot, not final totals. The batch deadline is 11:06 local.

## Completed paired learning-rate experiment

Batch `20261023T011009Z_update_stability` starts both branches from the same v19 checkpoint at 4,898,060 controls. Model, normalization, optimizer, environment and RNG states are identical; only the constant learning rate changes. Original-rate initial optimization records reproduce the parent continuation. All four endpoints were registered before execution. Counts below include actual rollout overshoot.

| Additional controls, original / lower | Original 3e-4 full task | Lower 1e-4 full task | 23m parking probes, original / lower |
|---|---:|---:|---:|
| 50,125 / 50,106 | 4/10 | 4/10 | 2/2 / 2/2 |
| 100,525 / 100,182 | 1/10 | 6/10 | 0/2 / 2/2 |
| 151,243 / 150,420 | 4/10 | 5/10 | 2/2 / 2/2 |
| 201,464 / 200,516 | 4/10 | 5/10 | 2/2 / 2/2 |

All 6m/12m/18m probes are 2/2 at every endpoint in both branches. Static-obstacle evaluation remains 0/2 at all eight endpoints. Approximate KL median/P95/max is 0.04109/0.07312/0.10506 versus 0.02800/0.04868/0.11167. Lower learning rate reduces typical observed changes, but does not bound KL or establish robust performance. The early-stop target is not a strict KL constraint.

The final lower-rate full-task result consists of both empty cases, both dynamic cases and one narrow case. The two static cases, both unseen cases and the other narrow case collide. All failures remain. During additional lower-rate training all 190 static episodes collide; original-rate static training has 161 collisions and one deadline. Parking improvement has not solved obstacle negotiation.

Rebuild every endpoint and verify source, initial checkpoint, evaluated model and case-list hashes:

```powershell
python scripts/analyze_update_stability.py --batch runs/20261023T011009Z_update_stability
```

Outputs are `analysis/summary.json` and `analysis/update_stability.png`; the plot has been visually inspected. This is one seed with correlated branches, not a seed-level confidence interval or retrospective final-model selection.

## Action-mean saturation diagnosis and registered intervention

The final lower-rate checkpoint's six static/narrow/unseen replays use frozen normalization. Removing obstacles from an unexecuted observation probe changes steering commands, ruling out complete input insensitivity. In all five collision episodes, however, the Gaussian longitudinal mean is outside the actuator range at every decision. On the two static cases its raw values range from 3.81 to 5.18 with mean standard deviation about 0.51; actual normalized acceleration is clipped to +1 throughout. Collisions occur at the configured 3 m/s speed limit. This supports an exploration/saturation hypothesis, not proof of causality.

`scripts/audit_obstacle_response.py` retains actual executed traces, raw Gaussian means/stds and unexecuted obstacle-removal probes. It restores the world before every real step and checks frozen normalization. Output: `obstacle_response_lower/results.json` within the completed rate experiment.

New batch `20261023T012208Z_bounded_mean_pair` branches from the **fixed final** lower-rate endpoint at 5,098,576 controls. It keeps all tensors, optimizer/RNG/environment state, reward, task distribution, action hold, variance and learning rate identical. One policy replaces the Gaussian mean mu by tanh(mu). This bounds the mean, not Gaussian samples: SB3 likelihoods, gSDE and PPO remain unchanged, and normal actuator clipping still applies. No planner, teacher, action mask or safety rule controls B1.

Milestones are zero, +100k, +300k and +600k controls in each branch, with all ten complete tasks and eight distance probes. The zero-update outcomes are 5/10 and 6/10: this immediate parameterization effect is explicitly **not learning gain**. Both retain all eight parking probes. Static cases are the primary diagnostic; no best endpoint will replace the registered endpoint sequence. At +100k controls the branches score 3/10 and 6/10; both static and unseen cases still collide, so there is not yet support for the primary obstacle-learning hypothesis. The actual worker is running under a one-hour deadline, resource checks and an identity-verified process-handle watchdog. All ancestral and discarded sibling costs remain in parent evidence; this is not from-scratch training.

## Verification and remaining limits

Full CPU regression after the policy changes: **123 passed / 5 conditional skips**, including the dedicated Windows retained-handle test. Tests cover true optimizer updates, Gaussian/gSDE log probabilities at out-of-range sampled actions, exact parameter/optimizer transfer, save/reload, native checkpoint RNG/hashes, and duplicate-recovery prevention. Optional live tests skipped here are not counted as CARLA validation.

The new code/results are not yet included in the previously published ZIPs. The independent-checkout live v3 replay remains pending until the native queue releases CARLA. Five independent policies, healthy full-task B1, native static/dynamic competence, feedback and strong-B3 contributions, comfort improvements and fresh 120-condition confirmation remain required.
