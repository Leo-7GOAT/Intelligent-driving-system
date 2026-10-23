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
log standard deviation -2, entropy 0.001 and KL target 0.03. At 10:30 local the
worker had reached 25,600 controls/4,000 actual PPO optimizer steps and was
evaluating that endpoint; the final result was not yet available. This is a
timestamped observation, not a final performance claim. The worker is guarded
by a fixed deadline, available memory/disk/GPU checks and a verified process
handle with no automatic retry.

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
