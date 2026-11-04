# Executed demonstrations and closed-loop learning diagnostic

This is development work with one policy seed, not independent confirmation,
pure reinforcement learning, or evidence of hierarchy superiority. The full
research acceptance gates remain unmet.

## Motivation and protocol

The completed lower-rate, tanh and softsign PPO studies retain some parking
ability but still collide on both static and unseen development cases. This
diagnostic tests whether direct-control learning can use productive executed
actions before attributing every failure to PPO exploration.

Collection batch `20261023T021232Z_executed_demonstrations` freezes 32 cases:
seeds 8101–8108 crossed with empty, static, narrow and dynamic scenes. Seeds
8101–8106 are training demonstrations; 8107–8108 are held out from both gradients
and normalization fitting. All cases use the training scene generator. Existing
development cases 7301/7302 remain separate and are never fitted.

B3 executes Hybrid A* and MPC in the lightweight vehicle plant. A separate B1
plant then replays the requested acceleration/steering commands, integrating
every control step. It never copies intermediate vehicle state. This produces
correct B1 previous-action observations instead of copying hierarchical action
semantics. Both physical interactions are charged separately. High-level
observations/actions are also retained for comparable future teacher access.

Teacher and direct replay each complete **32/32** cases, consuming **10,817
controls each**; the teacher makes 1,088 decisions. The maximum replay state
difference is **2.66573e-7** across position, yaw, speed, steering, acceleration
and time. All action labels and raw paired states were verified. No failed
episode was discarded. Dataset SHA256:
`dfaecf8cd4a381dc71c50e991f2d34c866c75928977e4945517fadf7df8cee71`.

## Supervised initialization and actual PPO

Batch `20261023T021813Z_demonstration_warmstart` starts a fresh seed-11 B1 MLP
with 605 observations. It uses exactly 200 supervised epochs, separate Adam
updates, and training-only normalization statistics. There are 8,113 training
examples and 2,704 held-out examples. The 6,400 supervised optimizer steps are
**not PPO updates**; PPO's optimizer state and update count remain empty/zero
at the supervised checkpoint. Supervised actor parameter L2 change is 10.5109.

The final training/holdout action MSE is 0.00118996/0.00117452. Despite that small
teacher-state prediction error, the BC-only policy completes **0/10** full
development cases: four out-of-bounds and six collisions. Parking probes score
0/2, 2/2, 0/2 and 0/2 at 6/12/18/23m. All outcomes are retained, not replaced
with a best fitted checkpoint.

An additional fixed-policy diagnostic actually executes the learned policy on
four seed-8101 training scenes and four seed-8107 held-out demonstration scenes.
Each group achieves only **1/4**, both successes being narrow scenes. Empty and
static cases leave bounds; dynamic cases collide. This shows that failure is
not confined to validation-layout changes. Low offline MSE does not establish
closed-loop competence. The audit records per-feature training-range departures
and normalized clipping without changing observations or fitted statistics.

The registered subsequent PPO endpoints are 0, 25,000 and 100,000 additional
physical controls. PPO uses learning rate 1e-4, gSDE resampling every control,
log standard deviation -2, entropy 0.001 and KL target 0.03. All endpoints are
now complete, including normal worker/guard exit and exact model reload:

| Actual PPO controls | PPO optimizer steps | Full development task | 6/12/18/23m probes (successes out of two each) |
|---:|---:|---:|---|
| 0 | 0 | 0/10 | 0 / 2 / 0 / 0 |
| 25,600 | 4,000 | 0/10 | 0 / 0 / 0 / 0 |
| 102,400 | 16,000 | 0/10 | 2 / 0 / 0 / 0 |

Both PPO endpoints have eight deadlines and two narrow-scene collisions. Actor
L2 changes relative to supervised initialization are 1.65107 and 2.99638;
updates occurred but did not solve the task. Evaluation separately consumed
6,268 / 15,723 / 13,933 controls. Every raw evaluation control and terminal
metric matches its summary; final plots were visually inspected. The fixed
deadline and resource/process-handle guard exited normally, with no retries.

The frozen-policy/normalizer factorial crosses supervised and 25,600-step
weights with both saved normalization states. All four combinations remain
**0/10** on the full development list. Only supervised weights with the later
normalizer achieve 2/2 at 6m; the other combinations score 0/2. The 48 executed
episodes retain all failures and unchanged normalization statistics under
`normalization_factorial`. Changing statistics alone does not recover full-task
competence; crossed pairs are diagnostic and are not promoted models.

For a concrete training-layout failure, the BC-only empty-scene replay stops
near x=25.753m with v=0.062m/s at 15.95s, short of its x=27m goal, then
accelerates to 1.135m/s while crossing x=27.082m at 17.55s and ultimately leaves
bounds. Teacher-state MSE conceals errors in terminal timing and recovery.
These observations motivate investigating physically visited recovery-state
coverage; they do not establish that a particular next intervention will work.

The next data-collection building block, `RecoveryTeacher`, is now implemented
as a separately invoked training oracle. A private planning/MPC workspace reads
the actual learner plant/world, computes a label, and verifies that actual state
and learner episode accounting are unchanged. Repeated queries without a real
control step and reuse after reset are rejected. It does not hook policy
inference or choose the action that the collector executes. Its integration
test uses a fixed planning node budget and reproduces the full actual B3 short
parking trajectory through a separate direct-control plant within 1e-4 in all
state fields. Query cost and planner/controller diagnostics are returned for
explicit accounting. A recovery-state collection/training experiment using
this interface has **not yet run**, so no improvement is claimed.

Direct control is hold1 (0.05s), whereas the previous PPO-only mean-transform
studies used hold10. Teacher decisions remain 0.5s with MPC controls at 0.05s.
Both the extra supervision and frequency change must be disclosed; a later
formal comparison must provide comparable supervision/information/budgets to
H/H0/B1/B2 and retain strong B3. Teacher success is not learned-policy success.

## Verification and reproduction

New tests execute a real teacher/direct plant replay, verify train-only
normalization, show actor learning without PPO updates and reload exact saved
weights. Full CPU regression: **132 passed, 5 conditional skips**, with two
existing SB3 schedule deprecation warnings. This does not substitute for native
CARLA validation. These demonstration cases are lightweight, not CARLA frames.

```powershell
python scripts/analyze_demonstration_learning.py --dataset runs/20261023T021232Z_executed_demonstrations --batch runs/20261023T021813Z_demonstration_warmstart
```

The analyzer requires all registered endpoints by default. During execution,
`--partial` explicitly labels a provisional figure/summary. It checks frozen
source, dataset hash, every replay control/action label, training-only example
counts, evaluated models, ordered case lists and separate BC/PPO counters.
Outputs are `analysis/verified_summary.json` and
`analysis/demonstration_learning.png`. Raw traces, failure outcomes and the
supervised checkpoint remain in their batch directories. The new dataset and
models have not yet been uploaded as a release asset.
