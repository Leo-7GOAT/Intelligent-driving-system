# Flat PPO diagnosis and isolated development variants, 2026-10-22

Research acceptance remains incomplete. The previous goal turn made concrete code, training and publication progress; this turn adds actual frozen-model diagnostics, a network implementation, tests and new bounded training. All results below are single-seed development evidence. Existing final conditions remain historical; no confirmation set is opened.

## Subsequent endpoint and cumulative continuation, 19:04 local

**19:18 update:** the unchanged continuation has completed its2M checkpoint at2,000,379physical steps,783phases/25,904optimizer steps, exactreloadtrue, and0/10. Six cases time out; both narrow and both unseen cases collide. Empty minimum goal distances are1.196/1.313m, still outside0.8m; final states stop farther away and brake throughout the last5s. None of the ten cases enters the goal region. The adverse change from two to four collision cases is retained. The worker proceeds to the already-registered3M milestone; this intermediate result does not justify saying that more training has solved B1. Raw `validation_2000000.json` and the new `terminal_audit_2000000.json` are in the continuation job.

This section supersedes the earlier live v15/v16 snapshots below. v15 pooled B1 has completed1,001,990 physical steps,392phases/13,016optimizer steps and0/10 development; v16 progress-reward B1 has completed1,001,373steps,392phases/14,224optimizer steps and0/10. Both supervisor exits are0; both exact reload checks pass. v16 records139successful training episodes among1,436, but only2among the final100. Training success is not substituted for deterministic validation success.

`scripts/audit_terminal_failures.py` checks every saved physical frame against simultaneous position/speed/heading gates and retains the recorded safety-priority outcome. The v16 endpoint has eight deadlines and two unseen collisions. Neither empty case ever enters the0.8m positional tolerance: it stops2.767/2.670m short at9.25s, with zero speed and requested braking throughout the final5s. Both narrow cases stop about11.7m short. Static/dynamic cases also never enter tolerance. This is not a bookkeeping-only parking failure and does not justify relaxing success criteria. The pooled model has six cases with zero progress; empty progress is only5.991/5.958m. The raw audits, input SHA256s and thresholds are saved as `terminal_audit_1000000.json` in each corresponding job. They are offline analyses of existing evidence, not additional independent evaluations.

New batch `20261022T110104Z_progress_v16_continuation` starts from the exact v16 final model, optimizer, normalization, environment and RNG snapshot; it uses the parent's immutable training source and configuration. Milestones are cumulative2M/3M, not two/three million additional interactions, with a two-hour deadline around21:01local and a two-minute drain. The reason is to test whether a longer training span resolves the observed failure after forward motion has been learned. The earlier0/10 remains in the learning curve. This is neither a new seed nor evidence that the changed reward is adequate.

Continuation preparation locks every parent owner, checks successful supervisor exits, checkpoint/source hashes, counter agreement and task/PPO consistency, then copies bytes and records ancestry. Missing/inconsistent checkpoints, active owners and nonadvancing budgets fail before a new batch is created. Unit fixtures explicitly test filesystem/provenance behavior, not policy performance. Actual resumed learning is separately verified: phase399 at1,019,292physical steps has14,383optimizer steps and actor-updateL2 .2154, exceeding the inherited392phases/14,224updates. Worker25592 and supervisor13284 were OS-confirmed from this batch; identities must be revalidated before later action.

At launch there was approximately62.5GiB free disk and35.3GiB available RAM; GPU40C. Only this one-thread CPU worker and the pre-existing v13 B1 worker train. Standard12GiBdisk/4GiBRAM/85CGPU/300sno-progress guards and no automatic failed-job retry apply. v13 B1 was still alive at663,153steps at19:03. No CARLA server was launched.

Full regression after the continuation and terminal-audit changes:92passed,5conditional skips (`reports/continuation_regression_20261022.xml`). The XML is local because it contains workspace paths. Earlier counts below describe earlier source versions.

Reproduce the terminal audit from an existing local batch:

```powershell
.\.venv\Scripts\python.exe scripts/audit_terminal_failures.py --batch runs/20261022T104130Z_progress_v16_development --job B1_progress025_hold10_seed11 --target 1000000
```

The active continuation has already been initialized and launched. Recovery, only after ownership/status inspection, uses:

```powershell
$batch = (Get-Content runs/progress_v16_continuation.json -Raw | ConvertFrom-Json).batch
.\.venv\Scripts\python.exe scripts/development_watchdog.py --batch $batch
```

## Completed endpoints

The common-curriculum v13 H/H0 pair is complete: H1,001,775 physical steps,397 PPO phases,14,390 optimizer steps,8/10; H0 1,002,283 steps,395phases,15,100optimizer steps,6/10. H's earlier100k10/10 peak remains visible alongside its later8/10 regression. The v13 hold1 B1 is still running, not replaced or silently restarted.

The v14 hold10 B1 frequency diagnostic completed1,000,149physical steps,100,096actions,391PPOphases and13,937optimizer steps. Actor L2 change was7.5002 and exact reload passed. There were5successful training episodes among1,218, but deterministic development finished0/10, with zero progress and deadline termination in all10 cases. The supervisor records exit0. This rules out treating its reduced jerk as adequate task learning. No collision/bounds episodes at the endpoint reflect immobility, not successful safe navigation.

Raw batches: `runs/20261022T093852Z_staged_v13_development` and `runs/20261022T102056Z_flat_frequency_v14_development`. Completed milestone summaries verify153/159frozen source files respectively. No failed model, rollout or endpoint has been removed.

## Actual control and feature diagnostics

`scripts/audit_flat_features.py` replayed the declared full23m empty conditions7301/7302 using the exact300k checkpoints and each batch's frozen source. Normalization stayed frozen. All actuators were driven by the original deterministic PPO output. Feature-zeroing was a separate, unexecuted sensitivity probe; it did not change the vehicle trajectory and is not a trained ablation.

| Model | Empty outcomes | Mean first-layer tanh saturation | Mean extra-zero action-mean L2 | Late behavior |
|---|---|---:|---:|---|
| v9,351inputs,hold1 | 2/2success | 4.23% | Not applicable | Both finished before30s |
| v13,605inputs,hold1 | 0/2 | 17.22% | 1.405 | Last15s mean speed0, braking100% of ticks |
| v14,605inputs,hold10 | 0/2 | 10.52% | 1.112 | Last15s mean speed0.040m/s |

No core observation values were clipped in these replays. The v14 jerk RMS was about1.78m/s³, versus about11.63 for v13 and15.93 for the successful older v9 examples. Smoothness alone does not explain completion. The old/new contrast changes input exposure and the training version; it does not causally prove that added information is harmful.

Evidence: `runs/flat_feature_audit_20261022/{v9_full23,v13_hold1,v14_hold10}` contains results and every actual control tick. `late_control_summary.json` marks absent late samples as missing because the successful episodes ended earlier. `flat_control_diagnostics.png` was rebuilt and visually inspected; successful traces are not padded to45s.

```powershell
.\.venv\Scripts\python.exe scripts/plot_flat_feature_audit.py --root runs/flat_feature_audit_20261022
```

## v15: candidate-set encoding for direct controls

The representation diagnostic motivates one isolated architecture trial, not a claim that saturation is the root cause. B1 receives the same605inputs, but its actor encodes the357global/outcome values separately, applies a shared8→32→16network to each candidate, mean/max pools the31embeddings, then fuses the result into a64-unit actor latent. The standard SB3 continuous head and gSDE still directly request acceleration and steering. There is no planner, MPC, teacher, action selector or hidden safety replacement. The critic remains605→64→64.

`FlatCandidateActorCriticPolicy` has77,363total parameters versus86,211for the flat MLP. This difference is disclosed; it is not an equal-network feedback ablation. H/H0 have76,866each in the current candidate version. The new tests verify actor mean/std invariance to candidate order, nonzero gradients from candidate features, actual PPO updates in every actor branch, resume and exact reload.

Batch `20261022T103301Z_flat_pool_v15_development` started18:34local. Relative to v14, configuration/cases/PPO options are identical except actor policy class; a machine-checked comparison is saved. Same fresh seed11,hold10,common300k empty phase,1Mphysical budget and two-hour guard window ending~20:33. At100k and300k it remains0/10; training continues. The scope is a healthy-baseline development trial, not H superiority. Source/configuration changes do not affect the frozen v13/v14 jobs.

A follow-up300k replay of the pooled actor also failed both declared empty cases (progress6.800/6.590m). Its global-context first layer showed no saturation and its unexecuted extra-zero action-mean L2 averaged0.445. That first layer excludes candidate blocks and is not directly the same quantity as the dense actor's first layer. This counterexample further limits a saturation-only explanation: improving the representation diagnostic did not establish task learning. Raw evidence is in `runs/flat_feature_audit_20261022/v15_pool_hold10`.

## Reward finding and v16

`scripts/audit_progress_signal.py` recomputed the discounted physical rewards of all38episodes from four declared saved evaluations, matching every recorded return within1e-6. This is an offline reward calculation on unchanged trajectories, not a new policy evaluation.

For the full23m empty task, the300k v14 trajectory advances10.078m and times out with return12.992. Its1M stationary deadline trajectory receives13.010. Correct terminal-zero potential shaping contributes a fixed initial-potential offset to same-duration terminal episodes; it does not preserve partial progress as an objective. Motion costs slightly favor stationary failures. This is a reward-objective/learning issue, not an error in the PBRS identity, and successful completion still has much higher return (the v9 examples exceed38).

New v16 changes only `reward.progress_weight` from0 to0.25 relative to the v14 MLP. The added term is0.25times the signed reduction in actual Euclidean distance to the final goal each physical step, including negative progress. ExistingPBRS40, events, vehicle, observations, actor, normalization, curriculum, budget and cases remain unchanged. It is explicitly a base-task reward change, **not policy-invariant shaping**. Before any H-versus-B1 claim, all relevant methods must be offered the same objective and retrained; the current run is only a B1 learning diagnostic.

The coefficient has a bounded rationale for this fixed23m/45s task: the discounted signed-progress sum is at most0.25×23=5.75, below the discounted bounds-vs-deadline event margin10×0.99^45≈6.36. Unit tests cover safe success, waiting, loops, stationary/partial deadlines and early/late collision or bounds failures. This is neither a safety guarantee nor a coefficient established for other route lengths. Re-scoring the saved10m partial trajectory adds2.349return; the stationary trajectory adds0. Reward-only counterfactual scores are never substituted for measured success rates.

Batch `20261022T104130Z_progress_v16_development` started18:41:55local. Fresh MLP seed11, common300k-empty phase then mixed, same1Mphysical budget, one worker, deadline~20:41. Its protocol and comparison record freeze this single difference. Actual PPO updates are verified: phase21 at53,449physical steps had811optimizer steps and actor-updateL2 .1691. This is not convergence evidence.

At the next inspection, v16 had passed317ksteps, with79training successes among462completed episodes. Its100k development endpoint was2/10 (both empty cases succeed); its300k endpoint regressed to0/10, with empty progress about21m and collisions in the other inspected families. These are both retained. The variant supplies evidence of basic learning but has not stabilized parking or solved mixed tasks. It must not be selected as a successful final method using only its earlier milestone. At the same inspection v15 exceeded633k with no successful training episodes; v13 B1 exceeded552k with one. These are cumulative developmental runs, not equal-budget formal comparisons.

## Current execution and verification

Three bounded CPU training workers remain: v13 B1 hold1, v15 pooled B1 hold10 and v16 progress-reward B1 hold10. At launch, disk remained above67GB and available memory above35GiB; each worker uses one Torch thread. Existing guards require12GiB disk/4GiB available memory, GPU below85C, progressing logs within300s, no automatic failed-job retries. No CARLA server was launched this turn; all prior actual CARLA/video evidence remains unchanged.

Full regression:79passed,5conditional skips. The learning trials preserve failed references and include their full costs. No new independent seed, formal confidence interval, generalization threshold, CARLA success threshold or hierarchy advantage is claimed. Remaining full research requirements are unchanged.

Recovery uses the existing pointer, not a new initializer:

```powershell
$batch = (Get-Content runs/flat_pool_development.json -Raw | ConvertFrom-Json).batch
.\.venv\Scripts\python.exe scripts/development_watchdog.py --batch $batch
# For the separate reward trial use runs/progress_development.json instead.
```

On a fresh checkout, `initialize_flat_pool_development.py` or `initialize_progress_development.py` creates the respective protocol; the watchdog starts it. A live lease prevents duplicate ownership. Inspect process identity, completed/failure markers and logs before any recovery.
