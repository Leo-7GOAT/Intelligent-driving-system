# Research results and claim boundaries

The fresh-route H0 intermediate10,186-control checkpoint has genuine actor
changes but near-uniform probabilities and a rejected argmax on road1's saved
initial scene. [Seven production reference queries](../reports/native_route_checkpoint_20261025.md)
are offline snapshot diagnostics, not a completed CARLA policy evaluation.
The original H0/H/B1 batch continues unchanged; final endpoint results are pending.

2026-10-25: [exact road collision batching](../reports/road_collision_batch_20261025.md)
reduces offline complete-search median cost on one recorded development state,
with exact status/nodes/cost/path agreement. Scalar remains default; no live PPO
or task-success improvement is attributed to this optional compute change.

Latest complete native development evidence (2026-10-24 18:38): all six H0/H/B1
32/256 rollout arms score0/4, covering639,009 audited training/evaluation controls.
B1 long improves mean progress from2.203275 to6.637067m but remains unsuccessful;
H long does not improve progress or success and raises intervention fraction.
The actual same-low-level B3 diagnostic scores3/4, with1,977 verified controls,
zero training or optimizer updates, and no interventions. Its fourth road reaches
a state with no moving references then waits until deadline. Matched initial
poses/task identities are verified and outputs reproduce independently. See the
[complete rollout results](../reports/native_rollout_horizon_20261024.md) and
[B3 result and diagnostic](../reports/native_rule_protocol_20261024.md).
Neither experiment supplies new independent seeds or held-out confirmation.
The following descriptions preserve earlier development stages.

The latest native PPO investigation is the
[rollout horizon development report](../reports/native_rollout_horizon_20261024.md).
The completed approximately100k native H0/H/B1 checkpoints all score0/4 on the
four training-road probes. The subsequent
[complete H0/H/B1 clipping comparison](../reports/native_clipping_pair_20261024.md)
also scores0/4 in every arm; larger actor gradients do not produce improved
driving. The offline rollout
diagnostic measures GAE endpoint-value sensitivity, not actual value error or
performance improvement. Its proposed real-training comparison retains the
existing research gates and must be distinguished from historical versions below.

Latest [acceleration/curriculum report](../reports/actuation_curriculum_20261022.md) supersedes earlier live status: v17 completes3M at0/10. Measured feedforward improves speed/jerk on four development routes while retaining braking spikes and worsening some lateral errors. This is control evidence, not PPO benefit. Fresh v18 route-stage training is running; its stochastic and deterministic short-route outcomes are reported separately.

Latest native-road and learning endpoints are in [the native/pose report](../reports/native_pose_20261022.md): v16B1 completes3M at0/10, v13B1 completes1M at0/10, v17is an ongoing single-seed shaping experiment (1/10at1M). Three real four-route fixed-reference CARLA variants each complete4/4 but all retain severe longitudinal jerk. No revised actuator is adopted. These supersede earlier live-status paragraphs below.

Latest completed v11/v12/CARLA endpoints and the ongoing common-curriculum experiment are recorded in [the 18:10 continuation report](../reports/camera_staged_20261022.md). It supersedes older live-status entries below. v12 H/H0 finish approximately 1M lightweight physical steps at 8/10 and 6/10; separate v11 B1 finishes at 0/10. Actual v12 CARLA adaptation finishes approximately 100k world ticks per model at 4/6 and 3/6; H's earlier 5/6 regresses. No new five-seed or final-confirmation result is claimed. Real success/failure camera evidence is illustrative only.

**Research acceptance remains unmet.** The [acceptance ledger](research_acceptance.md) retains the original targets; a development pre-release does not lower them.

Detailed current cases are in [the development report](../reports/development_20261022.md). Historical H/B1 lightweight results comprise three complete1M pairs: H30/40 andB1 0/40 for each. B3 also achieves30/40; two H seeds are incomplete. Historical CARLA adaptation has two paired seeds at approximately50k world steps, H9/30 and10/30 versusB1 0/30, with H successes confined to dynamic cases. These results do not establish a healthy-baseline hierarchy advantage or PPO necessity.

The v8 single-seed feedback development has H/H0 completion6/10 each at100k,8/10 versus6/10 at200k,8/10 each at300k, and8/10 versus9/10 at1M. Both frozen200k policies score2/6 in actual CARLA transfer with calibrated vehicle geometry; adapting those same200k policies for approximately100k actual world ticks each gives3/6 each. There is no established feedback advantage. Strong B3 on the same v8 lightweight development cases completes10/10, including both unseen cases failed by H. Its rule-selection compute and full traces are retained.

The new v12 experiment independently trains shared candidate-scoring H/H0 policies from scratch, keeping reward, action catalog and low level fixed. It exposes explicit target geometry with fixed scales; its matched ablation removes candidate witnesses and previous outcomes. At25k, H completes0/10 and H0 4/10; at100k, H completes8/10 and H0 6/10. Fixed100k CARLA calibrated transfer gives4/6 versus3/6; equal-budget real CARLA adaptation has started. These small developmental differences are not confirmed benefits. The v11 flat baseline is a separate450-input experiment (0/10 at100k and300k) and cannot be silently treated as the final605-input matched comparator.

No formal paired95% confidence interval is reported for this single-seed development. Repeated conditions from one model are not independent training seeds. Completion, interventions and progress are reported jointly; failed episodes stay in the denominator.

The release asset reproduces a declared200k development checkpoint rather than a best-seed or best-test selection. Its complete20-episode CPU replay was verified locally against the recorded outcomes. A fresh GitHub clone independently installed pinned dependencies, passed66CPU tests with5conditional skips, and reproduced all20 cases. Release SHA and asset download status are recorded separately in the release record.
