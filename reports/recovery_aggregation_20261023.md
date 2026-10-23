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

Batch `20261023T033822Z_current_state_relabel` completed under its two-hour
guard, with normal worker/guard exits. It replayed all 96 retained cases:
32,538 controls/teacher queries and1,371.404seconds of teacher-query time.
It executed only the recorded original commands;
new labels do not control the replay. Observations, returns, case IDs, masks and
execution-source flags remain unchanged. Historical macro labels are retained
as historical fields and are not new hierarchical supervision. Partial evidence
and failures are retained. This run performs zero PPO/supervised updates.

`verify_relabel_teacher.py` checked all input/source hashes,
non-action arrays, every new label against its requested command, every executed
command against the original source, clocks, terminal results and costs. The
registered actor-only comparison used the same round-one initial model,
frozen statistics, seed13, 200 epochs and 19,400 optimizer steps as the completed
actor-only control. Only direct action targets change. The completed fit
`20261023T042840Z_current_state_actor_fit` executes19,400supervised updates with
actor L2 change28.81247. The critic, exploration log-standard-deviation and
normalization remain unchanged; PPO updates remain zero. Full development is
1/10 (one dynamic success, six deadlines, three collisions), versus the control's
2/10. Parking6/12/18/23m is1/2/0/0 out of two each. All10,397evaluation controls
were audited against policy requests, physical clocks and terminal results.
The label-only change therefore does not solve competence; its result is retained.
Training/holdout action MSE is0.002497/0.013924, with different labels from the
control; these errors are not directly interchangeable task-quality measures.

The original 32-case group has now completed: 10,817 replay controls and new
teacher queries, 522.402 seconds of teacher-query time. Its independent partial
verifier checks all commands/labels/clocks and confirms every non-action array
is unchanged. This partial receipt cannot authorize fitting before all groups
finish; the verifier explicitly rejects an incomplete batch by default.

The next completed group also passed independent verification: original plus
round-one data total21,688controls/queries and932.882seconds of teacher-query
time. These partial checks were superseded by full verification, also completed
in the independent pinned environment. A bounded local helper waited for the
identity-checked worker and clean guard exit, then verified and prepared the
registered fit; the training worker was separately launched and guarded.
Its first Windows PowerShell
launch failed before fitting because a hashing cmdlet was unavailable; that
attempt is retained and the corrected PowerShell7 helper is separately recorded.
The data worker was unaffected. Preparation also checks the completed control's
original data hashes and identical fitting-function AST, in addition to fixed
initialization, splits and normalization.

## Pure-learner aggregation continuation

Batch `20261023T043924Z_current_state_aggregation` completed normally at13:02local
under its two-hour guard. It starts from the fixed current-state
label endpoint above, including its poor result, rather than switching to a
better earlier checkpoint. Two rounds collect seeds8401–8408 crossed with
empty/static/narrow/dynamic layouts; first six seeds fit, last two stay held out.
Teacher execution probability is zero in both rounds. The current-state teacher
only labels actual pure-policy visits. Every failed trajectory is retained.

All32,538ancestor observations/actions/returns are carried forward with their
original split (24,831training/7,707heldout examples). The613-control diagnostic
dataset is excluded because it includes heldout and validation cases. Each round
adds its new visits and runs200actor-only epochs with frozen statistics, retaining
both endpoint evaluations on the same18development cases. No PPO stage is
registered in this data-coverage diagnostic. Additional interactions, queries,
supervised updates and sibling development costs are separately recorded.
This intervention combines new visits with extra optimization; it is not a
causal isolation of coverage from training duration, an independent-seed result,
or a substitute for the required PPO hierarchy comparisons.

Both retained endpoints score4/10 (empty and dynamic successes only). Round one
has three collisions and three deadlines; round two has six deadlines, all with
terminal speed zero. Round-two static progress is11.31/13.38m and unseen progress
5.02/4.46m, so low collision counts must not be described as completed navigation.
Parking probes at6/12/18/23m are0/0/1/2 then0/2/1/2 successes out of two each.

| Round | New actual controls / queries | Teacher executes | Collection success (32 cases) | Supervised updates | Train / heldout action MSE |
|---|---:|---:|---:|---:|---:|
| 1 | 18,669 | 0 | 10 | 30,800 | .002024 / .019416 |
| 2 | 9,917 | 0 | 23 | 36,400 | .001792 / .039147 |

Collection cases include declared train and heldout layouts; their success is
not independent development success. Final training/heldout example counts are
46,400/14,724. Both critic and log_std tensors remain exactly unchanged, original
normalization remains frozen, and no PPO update occurs in these two rounds.
The final model SHA256 is
`a22756645fe203b46c827d573063ff26dae41973b93f5e33ef30147e67dff318`.

The strengthened analyzer independently recomputes every collection learner
action from the preceding checkpoint and stored observation. Batched inference
differences above2e-6 are checked again with the collector's single-row operation,
requiring exact equality (90/96 cases); maximum batched differences are4.40e-6
and6.32e-6. It also checks evaluated actions against actual requested commands,
physical clocks, actor parameter changes, and normal process exit. All28,586
collection controls and21,644evaluation controls pass. The result figure was
visually inspected. The earlier mixture study also passes the stronger checks;
its new derivative lives in `strengthened_analysis/`, while its original
`analysis/` summary and figure retain their previously published receipt hashes.

### Actual PPO continuation

Batch `20261023T050438Z_aggregated_ppo` was launched at13:05local and completed
normally at13:22local, within its14:04:38deadline, with one CPU worker,
12GiB disk/4GiB RAM/85C guards and no automatic retries.
It starts from the fixed round-two endpoint, not a selected best checkpoint.
The zero-update evaluation must exactly match the parent's success, parking and
control counts before PPO proceeds. Original PPO seed11 is retained; the
supervised permutation seeds are not new independent policy seeds. PPO options
are checked against the inherited model: learning rate1e-4, target KL.03,
entropy.001, gSDE resampled each control,2560-control rollouts, ten epochs.
No teacher query, teacher execution or additional supervised fit is performed.
Training normalization updates resume; evaluation statistics remain frozen.
All0/25k/100k milestones, failures and ancestral supervision costs are retained.
This is still demonstration-assisted B1 development, not a pure-PPO comparison.

At13:12local the first PPO endpoint is complete and raw-audited:25,600actual
controls,4,000optimizersteps,actorL2change1.665761. Full-task completion regresses
from4/10 to0/10 (seven deadlines,three collisions); all eight parking probes fail.
Its14,243evaluation controls remain recorded. The final endpoint completes
102,400controls/40optimization phases/16,000optimizer steps,actorL2change3.253203.
It remains0/10, all ten full tasks deadline; all eight parking probes also deadline.
The final evaluation uses16,200controls. All102,400training and41,325evaluation
controls pass action mapping, physical clock and terminal checks. Training has
102deadlines,25collisions,four successes,two bounds failures,plus an unfinished
episode. Actor reload is exact. Neither failed endpoint is replaced or hidden.
The cause is not established by this outcome. The subsequent bounded factorial
below tests this batch directly; the older warmstart factorial used different models.

### Normalization factorial and exploration scale

`normalization_25k_factorial/` completed72physical episodes and47,507controls
under its600-second resource guard (normalexit0, no retries). There are no
optimizer updates or teacher queries. All18declared development tasks/probes
are evaluated in each of the four policy/statistics combinations.

| Network weights | Observation statistics | Full tasks | Parking at6/12/18/23m (each out of2) |
|---|---|---:|---|
| Before PPO | Before PPO | 4/10 | 0/2/1/2 |
| Before PPO | After25,600controls | 4/10 | 2/2/2/2 |
| After25,600controls | Before PPO | 0/10 | 0/0/0/0 |
| After25,600controls | After25,600controls | 0/10 | 0/0/0/0 |

This factorial does not support statistics drift alone as the explanation for
losing all parking competence. The changed network fails with either statistics
snapshot; the untrained-in-PPO network retains full-task completion under both.
Statistics still affect safety: the old network with new statistics has three
collisions versus none in its original full-task evaluation. Swapped combinations
are counterfactual diagnostics, not eligible replacement models or final evidence.

`verify_normalization_factorial.py` checks every command, physical clock and
terminal metric. The25,125controls of both unchanged combinations reproduce the
original evaluations' actions and actual states exactly; all four combinations,
72episodes and47,507controls are required. Model, statistics, frozen environment
and diagnostic source hashes are recorded. Verification is in
`normalization_25k_factorial/verification.json`.

An offline distribution probe uses438fixed observations, selecting every16th
training-mask visit from aggregation round two. It introduces no interactions,
queries or updates. At PPO initialization, mean log_std is-2 but the actual
conditional Gaussian action standard deviation has median1.0243 in both
longitudinal and steering dimensions (95thpercentile1.0493). Probability outside
the normalized[-1,1] actuator range averages36.18%/32.98%. At25k the corresponding
medians are.9700/.9715, with43.28%/32.80% outside bounds. SB3's logged noise
parameter scale near.135 is therefore not the actual conditional action standard
deviation on these observations. This is a measurement on fixed training visits,
not a measured on-policy clipping frequency or a proof of the cause of failure.

The registered next contrast changes only initial gSDE log_std from-2 to-4,
reducing conditional standard deviation by exp(-2); log_std remains trainable.
The actor mean, all other parameter tensors, optimizer state, RNG, physical
state and normalization must match the original zero-update checkpoint exactly.
The baseline's completed0/25k/100kresults are retained and not repeated. The new
branch also retains these milestones with the same PPO objective, learning rate,
reward and cases. Its deterministic zero-update success/parking must match before
training proceeds. Preparation requires the baseline's full completed audit and
the factorial verification. Batch `20261023T052413Z_reduced_exploration` was
actually launched at13:24local with a14:24deadline, the same resource thresholds
and no automatic retries. Worker and guard OS identities were verified. Initial
checkpoint comparison passes: only log_std tensors differ; all other checkpoint
files, including RNG/physical resume state and normalization, are byte-identical.
The new branch's results are pending; smaller exploration is a hypothesis, not
an established remedy. Its initial-scale and later PPO updates are separately
reported from all inherited supervision and the completed comparator's102,400steps.

The new zero-update endpoint is already verified: all10,882actions and actual
states exactly match the original evaluation, and independent reload confirms
only log_std differs. Real PPO then begins. At7,680controls it has363optimizer
steps across three phases, fewer than the original1,200steps at the same budget
because the unchanged KL guard stops epochs earlier. This mediated change is
reported as an actual computational difference, not equal optimizer exposure.

```powershell
python scripts/verify_normalization_factorial.py --batch runs/20261023T050438Z_aggregated_ppo --output runs/20261023T050438Z_aggregated_ppo/normalization_25k_factorial
python scripts/measure_exploration_scale.py --batch runs/20261023T050438Z_aggregated_ppo --dataset runs/20261023T043924Z_current_state_aggregation/round_02/recovery_examples.npz --output runs/new_exploration_scale.json
```

```powershell
# Existing run: inspect runs/aggregated_ppo.json; do not prepare it twice.
python scripts/train_aggregated_ppo.py --parent runs/20261023T043924Z_current_state_aggregation
python scripts/analyze_recovery_demonstrations.py --batch runs/20261023T043924Z_current_state_aggregation
# After the PPO batch completes:
python scripts/analyze_recovery_demonstrations.py --batch runs/20261023T050438Z_aggregated_ppo
```

## Pure-learner visited-state audit

The fixed actor-only control was run on four predetermined static cases: original
demonstration training layouts8101/8107 and existing development layouts7301/7302.
It alone executes every control while the current-state teacher only supplies
unexecuted query labels. All four collide, after137/164/159/153controls respectively;
all613controls and queries are charged. There are zero optimizer updates and no
teacher intervention. Both validation trajectories (312controls) match the original
policy evaluation's actions and actual states exactly, ruling out query-induced
changes in those recorded executions. These repeated cases are not independent
policies or fresh confirmation. Seed8107 remains a heldout demonstration case;
its diagnostic labels are not admitted to the training dataset.

The actor's steering MSE on original static heldout demonstration visits is
0.000288 against held-reference labels, or0.000655 against current-state labels.
On its own four static trajectories, current-state steering MSE is
0.00658/0.01296/0.00654/0.00614. Different state distributions and target definitions
make these descriptive diagnostics, not a controlled superiority comparison.
The much larger error on self-visited states motivates checking recovery-state
coverage, alongside the already registered teacher-history label contrast.
It does not prove that either change alone fixes closed-loop learning.

Raw evidence and a visually inspected path/action figure are in the actor-only
batch's `learner_state_label_audit/`. The figure overlays actual teacher and
learner paths on the same two training layouts; same-state queried controls
never execute. Rebuild and verify with:

```powershell
python scripts/verify_learner_label_audit.py --parent runs/20261023T031926Z_recovery_actor_only --output runs/20261023T031926Z_recovery_actor_only/learner_state_label_audit
python scripts/plot_learner_label_audit.py --audit runs/20261023T031926Z_recovery_actor_only/learner_state_label_audit --relabel runs/20261023T033822Z_current_state_relabel
```

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

The later source `f49ee75c0ac378e17cf5179f585bbc6a510f58c4` is also pushed and
remote-verified. Both worktrees pass137tests/five conditional skips/two existing
warnings. The independent environment reconstructs the completed fixed-label
comparison byte-for-byte (summary SHA256
`9ae6d701643f9474fdf55209076d8c6738d10d877ef3198d8a946ec1276471c0`), checks both
model hashes and all18,541evaluation controls. Receipt:
`runs/prerelease_20261022/native_road_checkout/runs/actor_label_contrast_reproduction.json`.
The later completed pure-learner aggregation and live PPO continuation are
described above. Current root regression is137passed/five conditional skips/two
existing warnings (`runs/aggregated_ppo_regression.xml`); later independent
reproduction receipts identify their exact source version.

Source `6d351d586e43577bf44ef9c5e2d51c587515ded4` is pushed and remote-matched.
Root and independent pinned environment both pass137tests/five conditional skips
with two existing SB3 deprecation warnings. Independent reconstruction matches
the completed pure-learner aggregation summary byte-for-byte (SHA256
`2a7f8039b1a60c78b09db33e4cd74b56e4e8f543611895ec9721ade4bcd9865b`)
and its inspected figure. Both models reload and each gives64repeatable finite
predictions with frozen normalization. Receipt:
`runs/prerelease_20261022/native_road_checkout/runs/pure_learner_aggregation_reproduction.json`.
These are local artifact checks; no new release asset upload is claimed.
