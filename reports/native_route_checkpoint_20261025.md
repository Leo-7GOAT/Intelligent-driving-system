# Fresh route PPO checkpoint diagnosis — 2026-10-25

## Completed H endpoint, 20:55 local

Fresh H finishes **20,665 actual CARLA training controls**,2,080decisions,
65optimization phases/**130optimizer steps**. The20,480nominal target has
185charged overshoot controls. No parent weights/statistics, teacher or
optimizer intervention is used. Actor L2 change is **0.1835962832** and final
weights reload exactly. Training takes **8,630.72s** and21,594world ticks,
including929reset ticks. All29completed training outcomes remain:20success,
8deadline and **one out-of-bounds**. Training success is not a held-out score;
the training bounds failure must not disappear behind clean evaluation episodes.

The final frozen policy completes **2/4 known training-road probes**:

| Road | Outcome | Controls | Progress (m) | Selected waits | Plan/MPC failures | Intervened controls |
|---|---|---:|---:|---:|---:|---:|
| 0 | Success | 341 | 24.3164 | 1 | 0/0 | 0 |
| 1 | Deadline | 900 | 17.0442 | 65 | 0/0 | 0 |
| 2 | Success | 329 | 22.4616 | 0 | 0/0 | 0 |
| 3 | Deadline | 900 | 17.8448 | 48 | 0/0 | 0 |

Road1 ends with65consecutive policy-selected wait decisions/650controls,
moving only0.03985m over that tail; final goal distance is7.3034m. Road3
contains48wait decisions in total, with18consecutive terminal waits/180controls
and0.003554m tail displacement; final goal distance is2.8790m. These are
ordinary action0 selections, not low-level failure fallback or counted safety
takeovers. Low tracking error on the reference-covered portions cannot redeem
the unfinished task: reference coverage is only27.78%/46.67% on the failures.
All900controls and45simulated seconds of each failed case remain in totals.

Full decision P95 by road is **1.8549/3.7844/1.7597/2.2413s**; maxima are
2.0870/9.5776/2.1031/7.6180s. These exceed the0.5s decision period frequently;
no general realtime claim follows from successful synchronous execution.

The raw auditor binds **23,135 controls** (20,665training+2,470evaluation) to
the closed original CARLA stream. Evaluation uses2,594world ticks, including
124reset ticks; method total is24,188world ticks. Raw audit SHA256:
`98fe83f81f1bf01be8a70266a659264084d3e4b2837067855de7c0ecc9e0a9ae`.
Model SHA256:
`5926e13b49f289e4b848b10cce078d951c7d7f49b379cf0f49634e1ca317a8d4`;
normalizer SHA256:
`ee3d4079c1132a32ddf32901832d404fb88b90addb1f89d8cf299e2d341c1fd4`.

Independent checkout7605ba0 with a separate fixed environment reconstructs
the entire raw audit and all-eight-case H0/H figure byte-for-byte. Figure SHA:
`49776bdc6839ffff5bf3507dbb627de815282380d5229457ffe3eb168375f50d`.
Local figure `H/analysis/H0_H_frozen_endpoints.png` explicitly says PARTIAL2/3
registered arms and retains every case in each displayed arm; B1 is training.
Receipt: independent checkout `runs/route_H_endpoint_reproduction_20261025.json`.
Descriptive raw-bound action history: `H/analysis/H0_H_endpoint_actions.json`.

H **2/4**, H0 **3/4**, and previously audited B3 **4/4** do not support a
feedback or PPO-value improvement in this small development comparison.
H moves farther on road1 but regresses a previously successful road3. The
H/H0 environments, nominal budget and initializer are paired; each policy has
its own trained normalizer and executed state distribution. This is one seed
and four reused training geometries, not an independent confirmation or a
causal estimate from initial-state reference availability. Wait for B1's full
endpoint, preserve the failures, then run the separately declared
[initial/trained weight contrast](../docs/native_weight_contrast.md) only after
the original complete lifecycle and owned-server cleanup.

### H20k checkpoint integrity before endpoint evaluation

The immutable20k checkpoint has20,034controls,2,016actions,28completed
episodes,63optimization phases/126Adam steps and20,932world ticks. Its actor
L2 difference **0.1689163595** matches the real optimization record. All
frozen source/config/model/normalization/RNG hashes and counters verify.
Initial-weight argmax22/0/30/29 changes to30/28/30/30 under the same checkpoint
normalizer; selected reference lengths102/97/109/116 are all valid. Those
queries execute no controls and are not four successes—the final endpoint
above still fails two roads. Maximum probabilities remain0.032397..0.032549,
with entropy near uniform.

`H/checkpoint_20000_selection_verified.json` reproduces independently
byte-for-byte, SHA256
`b77df9e3fae119db3596105081663c743b18b945254d92ebbbc280d4d3920ff1`.
Model SHA is63eaeda04a1983e32f7cc3db8b8b50fbca72a665e6293264c9d3b3d31d4849fa;
it is distinct from the final20,665-control endpoint model.

## Completed H0 endpoint, 18:27 local

The later [optimizer replay](native_route_optimizer_replay_20261025.md) also
reconstructs64actual stochastic decisions and the final tensors/Adam/statistics
exactly; its discarded fixed-data branches are not new driving results.

H0 finishes fresh training with **20,702 actual CARLA controls**,2,080decisions,
65optimization phases/130optimizer steps. The20,480 target overshoots by222
controls because complete32-decision rollouts are collected; all are charged.
Training walltime is5,312.48s, with29completed episodes(19success/10deadline).
Actor L2 change is.07824054; final save/reload weights agree exactly. No teacher,
inherited model or normalization lineage is used.

Frozen deterministic evaluation on all four **known training roads** completes
**3/4**:

| Road | Outcome | Controls | Progress (m) | Planning failures | Interventions |
|---|---|---:|---:|---:|---:|
| 0 | Success | 337 | 24.1104 | 0 | 0 |
| 1 | Deadline, stationary | 900 | 0 | 90 | 900 |
| 2 | Success | 329 | 22.5262 | 0 | 0 |
| 3 | Success | 321 | 20.2868 | 0 | 0 |

Road1's tracking errors are correctly missing (`no_valid_reference`), not zero.
All failed decisions remain in totals. Successful roads take16.85/16.45/16.05
simulated seconds. Full decision P95 by road is.1658/.0788/.1998/5.2251s;
road3max is6.9071s, so no general realtime claim follows. Road1's cheap invalid
decisions and zero collision count do not make its stationary failure useful.
Its900intervened controls remain a substantial productive-safety failure.

The closed method stream verifies **20,702 training +1,887 evaluation =22,589
applied controls** against original CARLA frames, including actuator history,
commands, terminal evidence and reset counts. Training world ticks21,631 and
evaluation ticks2,011 total23,642. Raw-audit SHA256:
`c533950446a5ae79a6fa600c6ed82c0ffc3ab42308aa2ff568ebe211f0d5fdeb`.
Final model SHA256:
`aa7b6c3d0d0ed6e54a234b55f575a246607b0f6e4c8a7fe51cc4ace5c6b3bbc3`.
Normalization SHA256:
`d0e4bf935a5204bb271e0102898f349f0e6ae265be91df8d832b993f98270fdf`.

The independent fixed-dependency checkout rebuilds the **raw audit and four-road
trajectory PNG byte-identically**. Figure SHA256:
`99c4260fb6222de3cf02b1a0ed2542d8d766bfa7cba5c10c64dd275a0d2f9ae2`;
receipt there is `runs/route_H0_endpoint_reproduction_20261025.json`. Figure
`H0/analysis/frozen_endpoint.png` explicitly labels1/3registered arms and retains
the failed road. It was visually inspected. All190deterministic policy decisions
(34/90/33/33) also reconstruct exactly from the saved final model and audited
feedback/history with frozen normalization; local receipt
`H0/all_policy_decisions_reproduced.json`. No extra driving or PPO update occurs
in these reconstructions.

This is an honest positive **development endpoint**, not unseen confirmation or
evidence of PPO's causal contribution. The previous4/4strong rule and this3/4H0
have different decision mechanisms/information use, and changes from older0/4
runs also include low-level/interface changes. Five independent training seeds,
healthy complete-task B1, all core classes and paired statistical claims remain
open. The original worker now actually trains **H**; B1 remains queued. The
batch/server is still live and must not be restarted or used by another worker.

```powershell
python runs/20261025T085433Z_native_route_learning/source/scripts/audit_native_method_endpoint.py --batch runs/20261025T085433Z_native_route_learning --method H0 --output runs/H0_audit_rebuilt.json
python scripts/plot_native_endpoints.py --batch runs/20261025T085433Z_native_route_learning --method H0 --output runs/H0_endpoint_rebuilt.png
```

## H intermediate checkpoint, 19:33 local — not an endpoint score

The same frozen worker actually trains H after closing H0. Its first immutable
checkpoint contains **10,188physical controls**,1,024decisions,32optimization
phases/64optimizer steps (H0's corresponding threshold checkpoint had10,186
controls). H model SHA256 is
`f9b3b93b9f9ec1c626d75f0e7c2853423a95d6e2a92e981cef0819d644db01d4`;
normalization SHA256
`0ea54a9200b8846781a836d5a3effc903d7ca22169f71f41c6dfb6deb35899d6`.
Artifact/config/action counts and all frozen source hashes verify before
inference. Seed-reconstructed actor distance.11378152668476105 matches the
actual optimization log; no local reset, physical integration or update occurs
in the diagnostic process.

Using H's own frozen checkpoint normalization for both weight sets:

| Known-road initial scene | Initial-weight argmax | H checkpoint argmax | Selected production reference |
|---|---:|---:|---|
| 0 | 22 | 22 | Valid,108samples |
| 1 | 0(wait) | 28 | Valid,97samples |
| 2 | 30 | 22 | Valid,101samples |
| 3 | 29 | 30 | Valid,116samples |

All four selected initial references are available; the H0 intermediate/final
road1 selection was invalid. These are **seven offline reference queries**,
not four completed drives. H/H0 have their own learned normalization and
recorded scene snapshots, so this table is not a controlled causal estimate of
feedback contribution. H's maximum probabilities remain.032375–.032436 and
entropy3.433983–3.433985, still close to uniform. No unseen or convergence claim
is made, and the original H training continues without retuning.

Root evidence: `H/checkpoint_10000_selection_verified.json`. A separate fixed-
dependency interpreter in the independent checkout reruns the analysis and
reproduces the whole result **byte-identically**, SHA256
`1025c4d4a96c66b698f4a7451476373a6246b12b1dec276d9459ef356280dba5`.
Receipt there: `runs/route_H_checkpoint_10000_reproduction_receipt_20261025.json`.
Zero new CARLA controls, zero PPO updates, normalization unchanged. The live H
transition/bridge stream has not closed and has no complete endpoint raw audit.

The following sections retain the H0 intermediate evidence as captured.

The live seed11 fresh H0/H/B1 batch remains
`20261025T085433Z_native_route_learning`, frozen `b495e71`, with identical
feedforward actuation, four actuator-state inputs, route candidates, 7,200-node
Dubins-guided planner and shared 4,705-input observations. The optional newer
batch-SAT backend is not substituted into this active batch. H0 ablates the
99 candidate witness/execution features; it retains actuator history. H/B1 are
queued at the time of the intermediate checkpoint diagnosis below. The newer
completed-H0/current-H state above supersedes that historical live snapshot.

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
