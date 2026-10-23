# Recovery-state aggregation development

This is a one-seed supervised diagnostic followed by separately counted PPO,
not pure reinforcement learning, a fair final architecture comparison or an
independent confirmation. The PPO hierarchy and all original acceptance gates
remain required. Earlier failed models/results are retained.

## Registered intervention

Batch `20261023T030731Z_recovery_demonstrations` starts from the fixed supervised
checkpoint of `20261023T021813Z_demonstration_warmstart`, not the best earlier
PPO endpoint. Two rounds collect seeds 8201–8208 crossed with empty, static,
narrow and dynamic training scenes. The first six seeds are fitted; the last
two are held out. Original examples are retained with their original split.

On every actual state, a private planner/MPC workspace returns a teacher action
label without advancing physics or changing learner observations/metrics. Each
10-control block executes either the teacher or deterministic learner according
to a seeded Bernoulli draw. Registered teacher probabilities are 0.5 and 0.25.
Every executed source, both requested actions, queried label, actual state,
planner/controller diagnostics and query cost are recorded. There is no state
teleport, no removal of failed episodes, and no teacher in learned-policy
evaluation or PPO training. Mixture success is reported separately.

After each collection round, all accumulated training examples receive 200
supervised epochs. Original training-only normalization stays frozen through
aggregation; heldout examples never fit statistics or gradients. A separate
Adam optimizer, learning rate 1e-3, action MSE plus 0.5 value MSE and global
gradient norm limit 0.5 are used. Recovery value targets are discounted rewards
from the actually executed mixture, not hypothetical expert returns or PPO
targets. Normalization resumes its ordinary running update during later PPO.

The second supervised endpoint feeds fixed PPO milestones 25k and 100k controls,
with the same disclosed hold1/gSDE/rate settings as the parent experiment. All
18 development tasks/probes are evaluated at every registered endpoint. No best
intermediate result replaces a later failure. This differs from PPO-only hold10
studies and requires comparable extra supervision and cost accounting in any
future formal H/H0/B1/B2 comparison.

The batch has one CPU worker, a two-hour wall limit, minimum 12GiB free disk and
4GiB RAM, 85C GPU stop threshold, a process-identity watchdog, progress checks
and zero automatic retries. Completed round models/data and partial failure
evidence are preserved. Frozen source hashes identify the executed implementation.

## Observed collection and pure-policy outcomes

| Round | Actual controls/teacher queries | Teacher executes | Learner executes | Mixture successes | Additional supervised optimizer steps |
|---|---:|---:|---:|---:|---:|
| 1 | 10,871 | 4,662 | 6,209 | 32/32 | 12,800 |
| 2 | 10,850 | 2,808 | 8,042 | 27/32 | 19,400 |

Five round-two mixture failures remain in the training/heldout evidence.
Inherited costs are 10,817 original teacher controls, 10,817 direct replay
controls and 6,400 original supervised updates. The parent experiment's 102,400
PPO controls are discarded sibling development cost, not part of the restored
model lineage. All are disclosed rather than claiming training from scratch.

| Learned endpoint (no teacher) | Complete development list | 6/12/18/23m parking successes, out of two each |
|---|---:|---|
| Supervised round 1 | 4/10 | 0 / 1 / 0 / 2 |
| Supervised round 2 | 1/10 | 0 / 0 / 0 / 1 |
| PPO at 25,600 actual controls | 0/10 | 0 / 0 / 0 / 0 |
| PPO at 102,400 actual controls | 4/10 | 0 / 2 / 1 / 2 |

The first four successes are empty and dynamic cases. Static, narrow and unseen
remain failures. The second round regresses; its lower score is retained.
Final PPO training is complete: 102,400 direct controls, 40 optimization phases,
16,000 optimizer steps and actor L2 change 2.88833 from its supervised starting
point. Every training control was checked against raw direct-control records,
episode clocks and counters; training includes 65 deadlines, 72 collisions and
23 successes. Its full development evaluation has four successes, five collisions
and one deadline. The four successes are empty/dynamic cases. These are training
and development results, not independent confirmation. An additional fixed-policy static probe of
the round-two model is 0/8 for both training and validation layouts, in both
deterministic and stochastic inference. Thus failure is not confined to unseen
layouts. These repeated probes of one model are not independent policies.

## Separating a possible supervised loss interaction

At the exact round-one initialization and first predetermined round-two
minibatch, action MSE is 0.02650 and value MSE 67.1296. Action-gradient norm is
0.20018 versus critic-gradient norm 25.8317. The joint gradient clip multiplies
both by 0.019355. This is a measured coupling, not proof that it causes failure;
Adam's scaling and later minibatches also matter.

Batch `20261023T031926Z_recovery_actor_only` therefore repeats only the round-two
supervised fit from the same round-one checkpoint, same combined data/split,
frozen normalization, permutation seed 13, Adam settings and exactly 200 epochs.
It changes only supervised value weight from 0.5 to zero. The completed original
round-two fit is the control and is not rerun. This adds no collection controls
or PPO updates; it separately charges another 19,400 supervised optimizer steps
and actual evaluation. The critic must remain bit-identical and PPO counters
zero. All 18 cases are retained. This completed fit obtains 2/10 full-task
successes and parking counts 0 / 0 / 1 / 1, versus the original 1/10. Its 8,144
evaluation controls, unchanged critic, frozen statistics and zero PPO updates
were verified. Training/holdout action MSE is 0.003140/0.018436. The measured
gradient interaction is not established as the root cause or an effective fix.

## Teacher history and current-state labels

The eight-case history audit physically replays 3,120 recorded controls and
queries a fresh planner/controller at each actual state without executing that
new label. Every replayed state matches its source exactly. Held-reference and
fresh labels agree near decision boundaries but can differ within the ten-control
block; the largest longitudinal difference is 2.0 (opposite normalized extremes).
This identifies hidden reference/controller history in supervision, not proof
that removing it will solve learning.

A separate actual-execution audit runs the current-state teacher on seed8301
empty/static/narrow/dynamic training scenes: 4/4 successes, 1,341 controls.
These are teacher results, not learned-policy results or independent policies.

Batch `20261023T033822Z_current_state_relabel` is running under a two-hour
guard (deadline 2026-10-23 13:38 local). It replays all 96 retained cases, targeting
32,538 controls/teacher queries. It executes only the recorded original commands;
new labels do not control the replay. Observations, returns, case IDs, masks and
execution-source flags remain unchanged. Historical macro labels are retained
as historical fields and are not new hierarchical supervision. Partial evidence
and failures are retained. This run performs zero PPO/supervised updates.

After completion, `verify_relabel_teacher.py` checks all input/source hashes,
non-action arrays, every new label against its requested command, every executed
command against the original source, clocks, terminal results and costs. The
registered actor-only comparison can then use the same round-one initial model,
frozen statistics, seed13, 200 epochs and 19,400 optimizer steps as the completed
actor-only control. Only direct action targets change. This fit has not started
at this snapshot; neither a learned improvement nor causal explanation is claimed.

## Verification

`scripts/analyze_recovery_demonstrations.py` checks source and model hashes,
every teacher label and actually selected command, consecutive physical states,
time steps, terminal metrics, teacher/learner counts, train/heldout masks, frozen
normalization and independent BC/PPO accounting. Evaluation traces must contain
only direct controls. All prescribed endpoints are required unless `--partial`
explicitly labels the output provisional. The paired loss diagnostic preserves
the original control's metadata and data hashes.

```powershell
python scripts/analyze_recovery_demonstrations.py --batch runs/20261023T030731Z_recovery_demonstrations
```

New tests execute both all-teacher and all-learner collection, verify label/source
separation, reject stale teacher workspaces, keep normalization frozen and ensure
actor-only fitting leaves the critic unchanged despite large value targets.
The full regression after current-state collection support was 135 passed/five
conditional skips. A subsequent invariant test rejects changed visits, returns,
split masks and invalid relabeled actions; later regression receipts identify
their precise source version.
These are lightweight dynamics diagnostics, not CARLA frame evidence. None of
the new recovery-aggregation models/data is a release asset yet.

Source `ff6b5782b99d81e2562397f161ff7d01c9746ba0` was pushed and its remote
hash verified. Both the working repository and independent pinned-environment
checkout pass 136 tests with five conditional skips and two existing SB3
deprecation warnings. Independent reconstruction produces byte-identical
`verified_summary.json` and `recovery_learning.png`; all five completed
aggregation/actor-only models load and give repeatable finite predictions with
frozen normalization. The receipt is
`runs/prerelease_20261022/native_road_checkout/runs/recovery_aggregation_reproduction.json`.
This checks local diagnostic artifacts; it does not claim they were downloaded
from a release or that the research acceptance thresholds have been achieved.
