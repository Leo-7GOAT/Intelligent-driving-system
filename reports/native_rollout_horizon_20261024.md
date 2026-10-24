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
