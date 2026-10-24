# Failure analysis

The completed native H rollout pair adds 102,713 / 102,400 controls from the same
parent checkpoint. Both frozen evaluations score 0/4. Increasing rollout length
from 32 to 256 changes mean progress from 14.987119 to 14.939079 m and increases
mean intervention rate from 0% to 48.3333%. All 212,313 training/evaluation
controls across both arms pass raw-frame audit. H256 has no moving reference
at three final decision states; at the fourth, two references remain available
but the policy repeatedly selects a failed plan. All 360 H256 decisions reproduce.
This does not support adopting longer rollouts for H, and it leaves both early
choices entering catalog dead ends and weak action ranking unresolved. It is a
same-parent, seed-11 development comparison, not independent-seed evidence.
B1 is still running. See the [paired report](../reports/native_rollout_horizon_20261024.md).

The newest completed H short-rollout endpoint uses102,713additional native
controls:5training successes on one road (all with1–10failed plans),110training
deadlines, then0/4frozen evaluation. All106,313training/evaluation controls match
CARLA frames. Its frozen cases wait with zero interventions, but all30moving
actions fail production reference generation at every final state. The current
footprints remain free; three cases have no free candidate endpoints and the
fourth has two clear endpoints with exhausted searches. Unlike H0 below, these
states have no available moving reference under the current catalog/planner.
The distinction favors investigating earlier choices and continuation feasibility,
not simply adding a wait penalty. It is not proof of global geometric
impossibility or an established solution. All360decisions and six diagnostic
files reproduce independently; the completed H256 comparison is above and B1
remains incomplete.
See the [endpoint report](../reports/native_rollout_horizon_20261024.md).

Current native development evidence (2026-10-24) is in the
[rollout comparison report](../reports/native_rollout_horizon_20261024.md).
Both H0 rollout lengths complete 102,400 additional CARLA controls at 0/4 frozen
training-road successes. All 720 deterministic evaluation decisions reconstructed
from measured feedback reproduce exactly in a separate checkout. At all
eight final decisions the selected plan fails, while 6–26 alternative actions can
generate production timed references. The failed action's probability is only
3.87–6.81%; reference alternatives collectively retain 21.3–85.0%. This narrows
these particular failures to local action ranking, without establishing why the
learner acquired it or whether alternatives would track and finish safely.
Both arms' 113 completed stochastic training episodes also time out: probability
mass is not evidence that stochastic deployment solves the task. The H pair is
reported above; B1 remains pending, and evaluation rules/low level remain frozen.

Physical reward reconstruction across 226 completed episodes verifies the
terminal-zero distance-PBRS identity to 1.19e-14. Same-road/equal-time failures
have identical discounted event/time and shaping contributions; risk and comfort
explain their remaining return differences. This is expected shaping behavior,
not a newly discovered reward implementation error. Historical curriculum and
reward interventions below remain relevant counterevidence to simple remedies.

The following dated findings preserve earlier development versions and do not
override the current native integration or live-run status.

Latest completed [normalization/retention contrast](../reports/recovery_aggregation_20261023.md):
after 102,400 PPO controls, reduced noise with updating observation statistics
scores 0/10 (six collisions), while the matched frozen-statistics branch scores
4/10 (six deadlines) and parking 8/8. Both contain inherited supervision and one
correlated policy lineage. All static, narrow and unseen cases still fail.
At its 25k endpoint all nine deadline cases brake while stationary in the last
five seconds; at 100k the remaining failures end 8.68–19.04 m from the goal.
Freezing statistics helps this final retention comparison but does not establish
general competence, safety or a causal effect across independent seeds.
Native-road PPO has also been implemented and actually trained; its latest
H/B1 recovery probes remain 0/4. The dated paragraphs below preserve historical
diagnoses and are superseded by their linked later reports where indicated.

The [acceleration/curriculum report](../reports/actuation_curriculum_20261022.md) preserves PI overshoot, residual feedforward braking spikes, v17 final3M failure and v18 stochastic/deterministic parking discrepancy. High recent training success is not validation success.

Latest measured failures: [native control and terminal pose](../reports/native_pose_20261022.md). v16six near-goal stops fail yaw; v13full1M remains0/10. Mild coasting and manual-first-gear CARLA trials do not fix large brake spikes, despite4/4route completion. Default actuation remains unchanged; native-road PPO boundaries are still unimplemented.

This file records development findings, including negative results. Historical inspected final sets are no longer unseen confirmation sets.

The [latest flat-policy diagnosis](../reports/flat_diagnosis_20261022.md) records a completed1M hold10 baseline with0/10 and zero progress, feature/actual-control audits, and38verified physical-return reconstructions. Stationary deadline behavior slightly out-rewards partial-progress deadline behavior under the existing objective, although normal success remains better. Candidate-set encoding and the separate signed-progress reward trial have also completed1M at0/10. Physical-frame audits show the latter stops2.67–2.77m before the empty goals and never enters the goal region in any development case. An unchanged cumulative2M/3M continuation is running to test training span; existing failures remain. The reward trial is not policy-invariant shaping and requires matched retraining before any architecture claim.

Latest completed experiments are in [the camera/curriculum report](../reports/camera_staged_20261022.md): v11 B1 has zero successful training episodes in 1,505 episodes/1M physical steps; v12 H remains unsuccessful on both unseen layouts; CARLA H regresses from 5/6 at 25k to 4/6 at 100k. v13 now tests a common full-length basic-task pretraining stage, with no reward or control changes and no extra uncounted budget. Its H also regresses from 10/10 at 100k to 8/10 at 300k; do not hide this by selecting the earlier score. Camera infrastructure failures are recorded separately from policy deadlines.

| Observation | Evidence | Interpretation and action |
|---|---|---|
| Historical B1 gives0/40 while H and B3 both give30/40 | Oct16 archived paired evaluations | Cannot attribute the gap to learned high-level decision making; rebuild a healthy flat baseline and compare against strongB3 |
| B1 learns6m parking at1M but fails23m/45s | v6 model,8/8 short and0/8 full-route inference | Training-distance specialization; fresh fixed23m versus uniform6–23m development is running |
| Fresh fixed23m B1 passes23m at300k but fails6/12/18m | v9 two conditions per distance | Basic long-route motion/parking can be learned; distance generalization remains inadequate |
| H200k8/10 versusH0 6/10; both300k8/10 | v8 matched450-input development | One checkpoint difference is unstable; do not claim feedback contribution before sufficient independent runs |
| H300k fails unseen, H0 fails static | v8 raw per-family results | Same total completion can conceal different failure mechanisms; retain per-family reports |
| Frozen learned CARLA H/H0 both2/6 | v8 real-frame transfer, no infrastructure failures | Low collision counts coexist with four deadlines; no successful transfer or safety claim |
| Failed CARLA cases repeat candidate actions | `action_failure_audit.json` and raw traces | Search-budget exhaustion or endpoint body collision/bounds, rather than a sensor-disconnect explanation; matched real CARLA adaptation is running |
| Terminal target offsets could leave an unachievable final lateral correction | Geometric tests and B3 physical diagnostics | Added an explicit terminal return envelope; old experiments retained, new versions retrained |
| Legacy PID-only owner locks survive process death | Fault-injection and live reconnect tests | OS-held process leases now release on death; still no automatic recovery from a crash with live residual actors |

Calibration remains limited: steering was calibrated over a partial normalized range, native curved-road tracking is not yet established, and large measured CARLA braking deceleration spikes remain visible. Current curve cases are obstacle-defined curved routes on a straight roadbed. Controller and policy improvements must be evaluated with progress, time, intervention, failure and comfort together.

Current development is not a new final-test protocol. A future confirmation must freeze method/configuration, sufficient independently trained seeds, model-selection rules, budget, case generator, endpoints and multiplicity handling before inspecting its results.

## Recorded-state spatial audit

At all eight failed H/H0 frozen-CARLA terminal states, the shared offline planner can find paths to3–12distinct available candidate targets (speed duplicates excluded). The policy repeatedly chose different, unplannable targets. This narrows the issue to policy selection/learning or the quality of the feedback representation rather than universal absence of spatially feasible actions. It does not prove those alternate trajectories would track safely, and dynamic safety was deliberately outside the spatial audit. Source snapshots and all16initial/final state results are retained under `spatial_candidate_audit/` in the transfer batch.

## Completed follow-up evidence

The v9 fixed-distance and uniform-distance B1 jobs both finish1,000,960 physical steps with the same poor cross-distance results; they are no longer running. The v8 lightweight continuation ends H8/10 versus H0 9/10. Real CARLA adaptation of the original200k pair ends3/6 each at approximately100k world ticks, still with static/curve deadlines. Strong B3 on the identical v8 lightweight development cases achieves10/10, including both unseen layouts. These are counterexamples to a current feedback/PPO-superiority claim, not excluded outliers.

v12 now tests a shared candidate scorer with explicit geometry and unchanged reward/low level. H0 gets the same geometry and network but zero candidate witnesses/previous outcomes. All actions remain available; no teacher or auxiliary labels are introduced. Initial25k results (H0/10, H0 4/10) are retained. The new architecture has not earned an improvement claim. A separate mixed-task B1 is still training with its own frozen450-input version; it is not silently reused as a605-input formal comparator.
