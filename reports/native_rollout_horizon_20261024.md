# Native PPO rollout horizon development study

This is a seed-11 development investigation on four previously used training
roads. It is not independent confirmation, a new seed, or evidence of a
hierarchical advantage. The research acceptance targets remain unchanged.

## Measured motivation

`scripts/diagnose_native_rollout_horizon.py` binds the completed
`20261024T050724Z_native_continuation` logs to their raw audits and saved learner
parameters. H0/H each contain 8,000 logged decisions and 250 updates; B1 contains
8,096 decisions and 253 updates. All use 32 decisions per rollout, minibatches of
32, two epochs, gamma 0.9949874371 and GAE lambda 0.9746794345.

The diagnostic computes the derivative of the GAE return with respect to the
last bootstrap value, holding all other value estimates fixed. A true terminal
blocks this dependency. Its numerical implementation is checked against SB3's
actual RolloutBuffer with terminal and nonterminal sequences.

| Recorded decisions | Actual 32-decision groups: mean derivative | Hypothetical 256-decision groups: mean derivative | Actual groups without a terminal |
|---|---:|---:|---:|
| H0 | 0.541608 | 0.089967 | 64.8% |
| H | 0.541608 | 0.089967 | 64.8% |
| B1 | 0.407484 | 0.060005 | 30.83% |

The corresponding fractions of decisions with a nonzero dependency are
81.3% versus 18.625% for H0/H, and 57.695% versus 10.993% for B1.
Every logged decision is retained, including the final partial hypothetical
group. The regrouped data span changing behavior policies: they are **not**
on-policy training, reconstructed counterfactual returns, a measurement of
actual critic error, or proof that longer rollouts improve driving. These
numbers motivate a controlled real-training test only.

Per-method evidence is stored in
`runs/20261024T050724Z_native_continuation/{H0,H,B1}/rollout_horizon_diagnostic.json`.
No new CARLA controls or optimizer steps were used for this diagnostic.

Source commit `92eee2b` passed the full CPU regression: 193 tests passed,
five conditional tests skipped, and two existing SB3 deprecation warnings.
The independent checkout passed all 19 intervention/recovery/staging tests and
rebuilt all three horizon JSON files byte-for-byte. Receipts are
`runs/native_rollout_regression_20261024.xml` and, in
`runs/prerelease_20261022/native_road_checkout/`,
`runs/native_rollout_targeted_regression.xml` and
`runs/native_rollout_horizon_reproduction/reproduction.json`.
This uses local archived raw evidence, not a newly downloaded release asset.

## Prospective real-training comparison

`scripts/prepare_native_rollout_pair.py` requires the preceding six-arm clipping
study to be normally complete, all six raw frame audits to match, and the owned
server to have been stopped. It always selects the **global-clipping** parent
for H0, H and B1, independently of which clipping arm scores better.

Each method forks its same evaluated parent into 32- and 256-decision rollouts.
Both inherit exact policy weights, Adam state, normalization and saved RNG;
the buffer is freshly constructed. Minibatch size 32, two epochs, global
clipping, learning rate, rewards, observations, action catalogs, road geometry,
controller, and four training-road evaluation conditions remain fixed. The
longer collection changes GAE boundaries and the timing of policy updates;
these are part of the declared intervention, not separate improvements.

The default budget is 102,400 additional applied controls per arm, a six-hour
batch window, and the existing disk/RAM/GPU/no-progress guards. Whole-rollout
overshoot is retained, particularly when B1 episodes end early. Report actual
controls, reset/world ticks, decisions, optimizer steps and wall time; nominal
budgets are not a claim of exactly equal measured cost. Interrupted ancestor
costs and unknown unflushed tails remain recorded separately from model lineage.

The six arms are H0_32, H0_256, H_32, H_256, B1_32, B1_256. All are correlated
forks of seed 11. Evaluate every frozen road, retaining all failures. Completion,
collision/bounds, progress and intervention determine whether to continue the
development hypothesis; KL or gradient changes alone cannot establish benefit.
Mixed outcomes require further development evidence and cannot be averaged
into a formal five-seed or unseen-condition conclusion.

Registration refuses implicit rollout changes, simultaneous clipping changes,
changed parent artifacts, and duplicate endpoint claims. A corrupt staging copy
cannot consume a parent endpoint. The preparation command is:

```powershell
.venv\Scripts\python.exe scripts/prepare_native_rollout_pair.py --parent runs/20261024T064940Z_native_recovery --additional-controls 102400 --wall-seconds 21600
```

Preparation alone does not start training. Runtime batch identity, actual
processes, progress and outcomes are recorded in the latest handoff; do not
infer a running job from this prospective protocol.

## Actual schedule verification and outcome analysis

The registered batch is now `20261024T075110Z_native_rollout_pair`. Its first
H0_rollout32 checkpoint at 51,200 additional real CARLA controls contains
5,120 decisions, 160 collection/optimization phases and 320 Adam steps.
`analysis/H0_rollout32_checkpoint_000051200_schedule.json` binds these counters
to the saved learner and checkpoint hash. This is a running-checkpoint schedule
audit, not a complete endpoint/raw-frame audit or a driving evaluation.

Source `d35572b` adds `scripts/analyze_native_rollout_pair.py`. Unlike the clipping
analyzer, it does not require gradient measurements that this protocol never
collected. It rebuilds each completed raw frame audit; checks the unchanged
parent learning parameters, exact collection and optimizer boundaries, declared
budget and full-rollout overshoot; and retains inherited costs and all outcomes.
The pair must share exact starting checkpoint bytes, task configuration and
measured road/task identities. Actual starting pose differences are reported.
The default requires all three complete pairs; `--method H0`, `H` or `B1`
explicitly requests partial evidence. It does not infer a confidence interval
from four roads or correlated seed-11 forks.

```powershell
.venv\Scripts\python.exe scripts/analyze_native_rollout_pair.py --batch runs/20261024T075110Z_native_rollout_pair --output runs/20261024T075110Z_native_rollout_pair/analysis/complete_pair.json
```

Run only once all required endpoint audits exist; the analyzer refuses to
overwrite prior results. Six tests pass in both worktrees, including real PPO
collection at 32 and 256 decisions with respectively two and sixteen Adam steps,
changed-parameter rejection, mismatched-parent/road rejection, and retention of
failed outcomes and unequal actual overshoot. These checks verify analysis and
collection behavior, not a benefit of the longer rollout.

At 16:15 local, H0_rollout32 has completed 102,400 new controls, 320 phases and
640 Adam steps, retaining 222,720 controls in its model lineage. Training has
113 deadline episodes and no success. All four frozen training-road probes
also end at the deadline: mean progress 1.840831m and mean intervention 91.3889%.
The raw audit verifies 106,000 training/evaluation controls. Saved-model schedule,
four measured-start actions and action/terminal-pose diagnostics all pass.
The endpoint analysis is `analysis/H0_rollout32_endpoint_analysis.json`.

At that snapshot H0_rollout256 was training, with its first10k-threshold checkpoint at
10,240 controls/1,024 decisions/four phases/64Adam steps. The unchanged parent
parameters and the real256-decision boundaries are verified in
`analysis/H0_rollout256_checkpoint_000010000_schedule.json`. This arm is not
complete at that snapshot. The completed pair below supersedes this status.

## Completed H0 pair and physical reward reconstruction

Both H0 arms now finish 102,400 new controls, 10,240 decisions, 640 Adam steps
and 113 completed training episodes, all deadlines. Each retains 222,720 lineage
controls; each raw audit verifies 106,000 training/evaluation controls. Collection
phases differ as registered: 320 versus 40. Four paired evaluation starts have
zero measured position, heading and speed differences.

| H0 rollout | Frozen training-road completion | Mean progress | Mean intervention |
|---|---:|---:|---:|
| 32 decisions | 0/4 | 1.840831 m | 91.3889% |
| 256 decisions | 0/4 | 1.149521 m | 96.3889% |

Long minus short progress is -0.691310m. This completed H0 pair provides no
performance support for longer collection. H and B1 are still pending; this is
not the completed three-method comparison or a general rejection of long PPO
rollouts. All conditions are training roads and both models share seed-11 ancestry.
The authoritative partial report is `analysis/H0_pair.json` in the live batch.

`scripts/diagnose_native_episode_returns.py` reconstructs all 226 completed
physical episode returns across these two arms. Each arm also retains 700
controls in its unfinished final episode. With control gamma 0.9994976094477416,
the maximum error in the discounted terminal-zero distance-potential identity is
1.19e-14. Within a road and the same 900-control deadline, discounted event/time
terms are constant and potential differences are numerical noise; observed
return differences come from risk and comfort. This confirms the implemented
shaping identity, not a reward bug or causal proof that shaping prevents learning.
No reward was changed and no new training was attributed to this analysis.
Historical progress-reward, heading-potential and curriculum interventions remain
documented failures; this is not a claim that those ideas have never been tried.

The independent checkout at `ac59741` reproduces the H0 pair and both return
diagnostics byte-for-byte in `runs/native_rollout_H0_reproduction/`. The pair's
SHA256 is `459f91f223dfd3e876a426e7873dc23066614896324b68795c4416eaa565ef4c`.
This reproduction uses local raw evidence, not a new release download.

## Recovery hardening without changing the active learner

Source `ac59741` rejects a missing or altered RNG file whenever checkpoint
metadata records its expected hash. Only genuine legacy checkpoints lacking
both file and hash retain the explicit fresh-RNG path. Recovery staging now
verifies model, normalization and RNG before and after copying, before consuming
the parent recovery claim. Tests cover interrupted versus not-started rollout
arms, inherited cost/optimizer accounting, corrupted source/copy, and actual PPO
restoration with missing RNG. Full CPU regression: 203 passed, five conditional
skips, two existing SB3 warnings; independent targeted regression: 17 passed.
The running batch's frozen source is unchanged.

## Available actions at the recorded failures

`scripts/audit_native_action_availability.py` queries the production decoder,
Hybrid A*, trajectory parameterization and reference safety checks at each
measured initial state and immediately before the last evaluation decision.
It does not reset/step a plant, issue CARLA RPC, execute MPC or update PPO.
Archived source bytes, source text, raw-audit inputs and task geometry are bound;
only Git newline normalization is tolerated when comparing current source.
Dynamic scenes are explicitly unsupported without per-frame actor reconstruction.

For H0_rollout32, the four final decisions select actions 30/20/30/30, all
`plan_failed`, while the same states admit 26/10/26/22 reference-producing actions.
For H0_rollout256, all four select action30 and fail planning, while the states
admit 26/6/26/22 alternatives. Every such reference ends closer to the goal.
The selected action's mode and failure reason reproduce in all eight cases.
Thus these particular failures are not explained by complete absence of local
reference choices. This supports investigating learned action ranking; it does
not prove the alternatives are safe under execution, lead to eventual parking,
or establish which training intervention will repair the policy. None of these
distant-state references contains a complete parking pose.

Evidence: `{H0_rollout32,H0_rollout256}/action_availability_v2.json` in the live
batch. Earlier `action_availability.json` files retain the stricter byte-only
source check; their local results are preserved. Three tests check no physical
step, near/far parking reference distinction, unsupported dynamic/time-budget
rejection, and source-change detection including newline-only checkout variation.

At source `530d168`, all three tests pass in both worktrees. The independent
checkout reproduces both complete availability reports byte-for-byte, including
all 16 states and 496 action queries. Its receipt is
`runs/native_action_availability_v2_reproduction/reproduction.json`; report hashes
are `eb1b33efc567e7b9966516d98d2330df9fe8d2a26ab6cf5996d2873ff7a8b2af`
and `7e061cf4ef8948c55de512350c2a57f401862eb8e538b5e5ebe52dbb5050a664`.
No extra CARLA controls were consumed. The initial byte-only reproduction
attempt remains recorded as a source-newline mismatch; it was not a successful
reproduction. Historical clipping H_global was also rejected because its frozen
native source differs; no historical H availability result is claimed.

## Complete decision-sequence inference

Source `6cc39cd` adds `scripts/audit_native_policy_sequence.py`. It reconstructs
observations before every evaluation decision using measured state, previous
applied command, last selected behavior and execution outcome. Its read-only
plant holder has no step method. It requires the unchanged recorded ego-arc
interface, empty input queue and no unreconstructed dynamic actors. Model,
normalizer, configuration, trace, source and availability reports are bound to
the completed endpoint. Normalization statistics remain unchanged.

All **360 decisions per H0 arm**, including each repeated failed selection,
match the frozen deterministic policy. The eight final decisions remain failed
plans even though reference-producing alternatives retain nonzero probability:

| Case | H0_32 max single-action probability | H0_32 probability on reference actions | H0_256 max single-action probability | H0_256 probability on reference actions |
|---|---:|---:|---:|---:|
| 0 | 0.06809 | 0.84016 | 0.04611 | 0.84381 |
| 1 | 0.04531 | 0.38034 | 0.03869 | 0.21331 |
| 2 | 0.05358 | 0.84623 | 0.04312 | 0.84954 |
| 3 | 0.06157 | 0.77140 | 0.04457 | 0.75530 |

The largest-probability action is one failed choice; probability mass on many
alternatives can nevertheless be substantial. This identifies weak and locally
incorrect deterministic ranking, not a saved-model loading discrepancy. It is
not a stochastic-policy performance measurement: all 113 completed stochastic
training episodes in each arm still timed out. Neither random sampling nor an
action mask is adopted from this offline finding, and the registered deterministic
comparison remains unchanged. No causal attribution to a particular network,
reward or optimizer component is established.

Three tests compare every reconstructed raw observation with the production
environment's output across reference, failed-plan and wait decisions, for
enabled and ablated feedback; unsupported input queues are rejected. They pass
locally and in the independent checkout. These synthetic tests are not CARLA
performance evidence. The independent checkout additionally reproduces both
actual 360-decision reports byte-for-byte. Receipts are
`runs/native_policy_sequence_reproduction/reproduction.json`, with hashes
`e34830b3af7b03348231e9915875939bcc5d509b9fd5b874753535ed2184ebd2`
and `24a2952935fb3afc1ed77f9b9c15219fe83dfcedd99add2f3234afb66c7614f9`.
Original reports are `{H0_rollout32,H0_rollout256}/policy_sequence.json` in the
live batch; no new CARLA controls or PPO updates were used for reconstruction.

## H short-rollout endpoint: training successes, failed frozen evaluation

H_rollout32 now completes **102,713** new controls (313 retained rollout
overshoot), 10,272 actions, 321 phases and 642 Adam steps. Its lineage is223,033
controls; training world ticks106,308 include3,595reset ticks. Model reload is
exact, actor L2 change3.534195, and training wall time2249.82s. The complete raw
audit verifies **106,313** training/evaluation controls. Training has110deadlines
and5successes, with490controls in the unfinished final episode. All five
successes occur on road7d46c3cc23 and include respectively100/30/30/10/10
intervention controls (10/3/3/1/1 failed plans). They are not five independent
training seeds, intervention-free successes or a stable completion-rate claim.

Frozen deterministic evaluation remains **0/4**, all deadlines, with no plan or
MPC failures and zero interventions. Mean progress is14.987119m. The individual
failure evidence is:

| Case | Progress (m) | Final goal distance (m) | Final heading error (rad) | Wait decisions /90 | Available final moving references |
|---|---:|---:|---:|---:|---:|
| 0 | 20.147606 | 4.484173 | 0.459003 | 48 | 0 |
| 1 | 21.097537 | 3.250058 | 0.082785 | 53 | 0 |
| 2 | 15.589474 | 7.192884 | 0.022576 | 53 | 0 |
| 3 | 3.113859 | 17.609960 | 1.193539 | 68 | 0 |

The final action is wait in all four cases. Production reference reconstruction
rejects all30moving choices at each measured final decision state. All four
current footprints remain free; cases0/1/3have no free candidate target footprint,
while case2has two clear endpoints whose searches exhaust the open set. A search
failure is not a proof that no path exists under every planner/discretization.
This differs from H0, where multiple moving references remained available but
the selected action failed. H's evidence points toward earlier decisions entering
states with no continuation under the current catalog/search, not merely
voluntary waiting despite available references. It does not establish whether
more training, a different candidate set or a continuation-aware feedback feature
will repair the problem. Historical route-frame candidate diagnostics also failed
to repair several terminal states and must not be presented as an untested cure.

All360H evaluation decisions reproduce from the frozen model and measured
history. Final wait probabilities are.08790/.13544/.13432/.15486; available
reference mass is zero in all four. Episode-return reconstruction covers all115
completed training episodes and verifies the distance-potential identity to1.90e-14.
The H256 arm has now completed; the paired result below supersedes its earlier
running status. Keep all arms and the registered evaluation unchanged.

Evidence in the live batch: `H_rollout32/endpoint_raw_audit.json`,
`action_learning_diagnostic.json`, `action_availability_v2.json`,
`policy_sequence.json`, `episode_return_diagnostic.json`, and
`analysis/H_rollout32_endpoint_analysis.json`. The raw-audit SHA256 is
`23a96a82f0e4e61f103f8e61acca2c9f3008d4d2d140db090a9c28ac2b587b7b`.
The separate [native rule evaluator](native_rule_protocol_20261024.md) is ready
for a later same-interface diagnostic; it has not been started while this batch
owns CARLA, and provides no B3 result yet.

Independent checkout at `bee41c4` reproduces all six H endpoint/diagnostic files
listed above byte-for-byte, including the106,313frame audit and360policy actions.
Receipt: `runs/native_H_rollout32_reproduction/reproduction.json` in that checkout.
This uses local archived evidence and no new CARLA controls, optimizer steps or
release download. That earlier reproduction covered only the short arm; the
complete H pair and its reproduction are reported below.

## Audited visualization of the first three endpoints

`analysis/first_three_endpoints_v2.png` in the live batch shows every frozen road
for H0_32, H0_256 and H_32: all12failures are retained. Gray cells are measured
non-driving regions, blue lines are actual CARLA trajectories, red footprints
and arrows show final pose, and green marks show goal position/yaw. The title
explicitly says PARTIAL3/6; its JSON companion lists the still omitted H256/B1
arms. The image is diagnostic development evidence, not a complete pair or
generalization claim.

```powershell
.venv\Scripts\python.exe scripts/plot_native_endpoints.py --batch runs/20261024T075110Z_native_rollout_pair --method H0_rollout32 --method H0_rollout256 --method H_rollout32 --output NEW_PARTIAL_FIGURE.png
```

The default without `--method` continues to require every registered arm. The
complete clipping-study figure was regenerated byte-identically to its published
version (SHA256847e27ff7bf5a5e7136117f2e83eb71ce38b43f7284811ce8a471ef701b00014).
Independent checkout at362a11a reproduces the new partial figure byte-for-byte,
SHA256589f39c1b7e592c48b1939171d08e5edcc39309da556389ec45f91c44de98937,
with matching case inventories and omitted-arm labels. Receipt:
`runs/native_partial_plot_reproduction/reproduction.json` in that checkout.
No new vehicle controls were consumed by plotting.

## Completed H rollout pair

H_rollout256 completes 102,400 new controls, 10,240 decisions, 40 optimization
phases and 640 Adam steps. Retained lineage and charged controls are 222,720,
with no unknown ancestor tail. Training consumes 105,933 world ticks, including
3,533 reset ticks, and 1,771.89 seconds of wall time. All 113 completed training
episodes time out; the final unfinished episode contributes 700 controls. Actor
L2 change is 2.666812 and model reload is exact. Its raw audit verifies every
one of 106,000 training and evaluation controls against CARLA bridge frames.

| Measure | H rollout 32 | H rollout 256 |
|---|---:|---:|
| New training controls | 102,713 | 102,400 |
| Rollout overshoot | 313 | 0 |
| Optimization phases / Adam steps | 321 / 642 | 40 / 640 |
| Completed training successes / episodes | 5 / 115 | 0 / 113 |
| Frozen evaluation successes | 0 / 4 | 0 / 4 |
| Mean progress (m) | 14.987119 | 14.939079 |
| Mean intervention rate | 0% | 48.3333% |

Both branches start from identical saved learning state; measured evaluation
start position, yaw and speed match exactly in every case. The long-minus-short
progress difference is -0.048040 m and success-count difference is zero. The
313-control budget difference is retained, not silently rounded away. Longer
rollouts do not repair H in this development comparison and increase its
dependence on failed-plan braking. These same-parent seed-11 branches and four
training roads do not establish an independent-seed effect or a general verdict
on rollout length. The B1 pair remains unfinished; no complete three-method
rollout conclusion is claimed.

| H256 case | Progress (m) | Final goal distance (m) | Final yaw error (rad) | Failed-plan decisions / 90 | Wait decisions / 90 | Final available moving references |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 20.218632 | 4.413147 | 0.447043 | 50 | 0 | 0 |
| 1 | 21.158885 | 3.188710 | 0.102609 | 0 | 53 | 0 |
| 2 | 15.306134 | 7.476224 | 0.098706 | 55 | 0 | 2 |
| 3 | 3.072665 | 17.651154 | 1.231346 | 69 | 0 | 0 |

All four evaluations time out and no control satisfies the complete goal-pose
criteria. Cases 0, 2 and 3 end by repeatedly selecting action 30, which fails
planning; case 1 waits. Offline production reference generation finds no moving
reference at the final decision states of cases 0, 1 and 3. Case 2 still has
actions 1 and 2 available, but the policy selects the failed action instead.
Thus the longer rollout retains both catalog-continuation and action-ranking
failures; it does not simply turn all voluntary waits into useful movement.
Available references remain hypothetical until actually tracked and do not
prove eventual parking. Search failures are not global infeasibility proofs.

All 360 deterministic evaluation decisions reproduce from the measured history
and frozen model without changing normalization. The final selected-action
probabilities are 0.03710, 0.04138, 0.03607 and 0.03719. Case 2 assigns 0.06903
total probability to its two reference-producing alternatives. Reconstructed
returns for all 113 completed training episodes verify the terminal-zero
potential identity with maximum absolute error 1.99e-14. None of these offline
checks consumes simulator controls or performs additional optimization.

Evidence: `H_rollout256/{endpoint_raw_audit,action_learning_diagnostic,
action_availability_v2,policy_sequence,episode_return_diagnostic}.json` and
`analysis/H_pair.json` in the registered batch. Raw-audit SHA256:
`7bc4d0b1b63bc27d80e251cf000f99c3ac5f7eed15baf915e7815b8330d97839`.
Paired-report SHA256:
`9ebacce2df160ae968adb5e92919be89a35e90139d5673fbbc77e486820fbba2`.

The independent checkout at `362a11a` reproduces all five H256 diagnostic files
and the complete H paired report byte-for-byte. This includes the 106,000-frame
audit, all 360 policy decisions and all 113 completed episode returns. Receipt:
`runs/native_H_rollout256_reproduction/reproduction.json` in that checkout.
This verifies local archived data and does not claim a newly downloaded release
asset, additional simulator execution or any research acceptance gate.
