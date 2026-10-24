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
