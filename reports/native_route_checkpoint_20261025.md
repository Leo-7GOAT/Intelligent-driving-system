# Fresh route PPO checkpoint diagnosis — 2026-10-25

The live seed11 fresh H0/H/B1 batch remains
`20261025T085433Z_native_route_learning`, frozen `b495e71`, with identical
feedforward actuation, four actuator-state inputs, route candidates, 7,200-node
Dubins-guided planner and shared 4,705-input observations. The optional newer
batch-SAT backend is not substituted into this active batch. H0 ablates the
99 candidate witness/execution features; it retains actuator history. H/B1 are
queued, and neither has a new endpoint at the time of this intermediate report.

## Immutable checkpoint and reproducible initialization

`H0/checkpoint_000010000` actually saves **10,186 physical controls**, 1,024
decisions and 64 optimizer steps. Its model SHA256 is
`ddf19e0f5b8cfbb7e471b1a1d70b35ae2a1d391d8403ab32348845b04eef0fcf`.
All checkpoint/source hashes are verified before inference. Two independent
reloads reproduce deterministic actions on the first four recorded initial
scenes with frozen normalization; that earlier receipt is
`H0/checkpoint_10000_reload_receipt.json`.

The new diagnostic reconstructs the fresh seed11 PPO constructor using the
recorded hyperparameters and exact frozen policy source. It checks that no local
environment reset or physical step occurs during construction. The checkpoint's
actor distance from this initial actor is **0.056107908487319946**, matching the
actual corresponding optimization log. Both weight sets then receive exactly
the same checkpoint-normalized recorded inputs. This isolates weight changes
from input-statistics changes; it does not compare two executed policies.

## Actual offline findings

| Known training road | Initial-weight argmax | Checkpoint argmax | Checkpoint selected-reference result |
|---|---:|---:|---|
| 0 | 24 | 22 | Valid reference, 108 samples |
| 1 | 0 (wait) | 22 | Invalid: start/goal body collision or bounds gate |
| 2 | 30 | 30 | Valid reference, 109 samples |
| 3 | 29 | 30 | Valid reference, 116 samples |

The initialized maximum probabilities are approximately .032274–.032286;
the checkpoint values are .032414–.032458, versus uniform 1/31≈.032258.
Entropy remains approximately3.43398 (maximum ln31≈3.433987).
Three initial argmax selections change, but the road1 argmax is still rejected
by the real production reference gate. Parameter changes and successful
stochastic training episodes therefore do not establish reliable deterministic
decision competence. The active run is not retuned based on this intermediate
probe; the declared endpoint and all failed cases remain required.

Only seven distinct selected-action production reference queries are needed
because repeated actions on the same scene share a query. The snapshot holder
has no `step` method. All state fields remain unchanged; there are **zero new
CARLA controls and zero optimizer updates**. Valid reference availability is
not successful execution. The live gzip streams are not yet closed, so these
stored-scene findings are not presented as a complete raw-frame endpoint audit.

## Reproduction

```powershell
$env:NAVSTACK_AUDIT_SOURCE_ROOT=(Resolve-Path 'runs/20261025T085433Z_native_route_learning/source').Path
python scripts/diagnose_native_checkpoint_selection.py --batch runs/20261025T085433Z_native_route_learning --method H0 --checkpoint checkpoint_000010000 --output runs/checkpoint_selection_rebuilt.json
```

Use the exact original frozen source and immutable checkpoint; output must be
new. Evidence is `H0/checkpoint_10000_selection_verified.json`, including exact
checkpoint/method config equality, saved action-counter matching and all frozen
source hashes. The earlier narrower receipt is preserved separately. This is a
single-seed intermediate development diagnostic, not a task success rate,
five-seed comparison, unseen confirmation or learned-high-level advantage.

The next actual low-level compute experiment has a separate checker,
`compare_native_rule_collision.py`. It requires two normally cleaned-up,
whole-frame-audited B3 runs with identical complete frozen source hashes,
calibration, four tasks and every setting except scalar versus batch SAT.
It retains failures, action differences, unmatched control tails, actual state
and command deltas, actuator outputs and complete-cycle timing distributions.
Its synthetic rejection/reporting tests are not actual CARLA execution. No
second tick owner or new deadline-bound trial has been started while PPO owns
the current server.

Targeted rejection/reporting and existing native audit/observation tests:
**29 passed**. These tests do not run CARLA. The previous complete CPU regression
remains267passed/five conditional skips; it predates the new read-only tools.

Independent checkout `09f90a7502533bbdb885cae5709ad4d5e6a79ead` with the
separate fixed-dependency environment also passes all29targeted tests. It
actually reruns this diagnostic against the original frozen source and immutable
checkpoint: the complete verified result is **byte-identical**, SHA256
`291838f259700aba7e628d97a6307edee74c0e4bf6839d7e5a7abe86ba78258e`.
Receipt: `runs/checkpoint_selection_reproduction_receipt_20261025.json` in that
checkout. This is an independent offline reconstruction of the same model and
data, not another independent training seed or another actual drive.

## Immutable training-log prefix figure

`scripts/plot_native_training_prefix.py` captures only complete JSONL lines
without modifying the running writer. Snapshot files, config, protocol and plot
source are hashed. Replay verifies these saved files and never reads newer
training rows. The first H0 figure uses57optimization records through18,148
controls and26completed training episodes:16success/10deadline. It displays all
episode rewards/progress, actor L2 change and categorical entropy deficit from
ln31. Actor change reaches.06888445, but the logged entropy remains3.43398130
versus maximum3.43398720. Successful stochastic training episodes are present
from early in the run; their presence alone is not evidence of learning gain.

Figure and manifest:
`H0/analysis/training_prefix_20261025_1815.png` and adjacent JSON; saved inputs
are in `training_prefix_20261025_1815_inputs`. Figure SHA256:
`06149d403cdb04dcffaa9716e052e75cc5c13b0b429383ffa244458d7c5f2be0`.
The figure was visually inspected and independently rebuilt from the frozen
plot script and saved inputs with the separate fixed-dependency environment;
both PNG and result JSON are byte-identical. All failures remain on the plots.
This is a stochastic **training-log prefix**, not validation, final testing,
an independent-seed interval or a completed CARLA raw-frame audit.

```powershell
python scripts/plot_native_training_prefix.py --capture-batch runs/20261025T085433Z_native_route_learning --method H0 --output runs/training_prefix_new.png
python scripts/plot_native_training_prefix.py --snapshot runs/training_prefix_new_inputs --output runs/training_prefix_replayed.png
```

Use new output paths. This reversible plotting work does not change the live
training source, low level, policy, normalization statistics or tick ownership.
