# Training-state approach diagnosis (31 October)

The completed v4 B1 PPO score remains **7/12, zero collision/bounds failures,
five deadlines**. Four failures stop 1.45–2.37 m before the goal. Their distance
gate fails while speed and heading pass; the fifth remains farther away and
moving. The original 0.8 m position tolerance is retained.

An offline reconstruction verifies all 7,257 evaluation control histories and
reproduces the final policy's last action in every condition. At the four
stopped failure states, the zero-PPO initialization also requests braking.
The final policy requests somewhat stronger braking. This is a comparison at
the final policy's visited states, not a replay of the initial policy or proof
that its complete trajectories would fail identically. Observation statistics
are exactly equal. Evidence: `native_B1_v4_terminal_policy_drift_20261031.json`.

Queries then use **training data only**: the first low-speed stopping state per
condition, selected without checking episode success, and fixed preceding
offsets. The initial 24 queries often yield a B3 wait command. A separate
27-pose interior goal-region probe recovers only one of ten wait samples and
still requests braking there. This prototype is not enabled in production.
Finite Dubins samples and bounded B3 search do not prove physical infeasibility.
The result does not justify changing the success region or treating an
analytical failure as a complete reachability certificate.

The extended fixed-offset query set adds 6/9-second approach samples. It has
40 counterfactual commands from eight conditions across the 39 completed
training episodes; the unfinished episode remains in the raw evidence. There
are 29 solved-reference labels and 11 wait labels. No command is executed and
no new CARLA interaction is claimed. The labels are model-based B3/MPC queries,
not measured expert executions.

Two actor-only fits are now prepared for a matched native gate. Both start
from the **same original zero-PPO checkpoint**, use the original 3,467 teacher
TRAIN rows and 512 additional MSE updates. Original holdout rows stay excluded.
The replay control uses those rows alone; the approach arm adds all 29 valid
solved-reference labels with weight 20, including negative accelerations.
Wait samples remain in the query log but are not goal-directed correction
targets. This selection is based on oracle validity, not positive acceleration
or later evaluation success. The critic, empty PPO optimizer, normalization
bytes, RNG bytes and log standard deviation remain unchanged.

Both models will run the same fixed twelve development conditions. Their
native results are pending; the additional fits are not PPO updates or new
independent seeds. Donor training/evaluation interactions are charged as input
costs. This is a baseline-strengthening experiment, not evidence that PPO has
improved or that the hierarchy is superior.

Evidence: `native_B1_v4_approach_pair_preparation_20261031.json`. Local immutable
query manifests and executed sources are under
`runs/B1_v4_training_stop_oracle_20261031` and
`runs/B1_v4_training_approach_oracle_20261031`. The native runtime remains frozen;
only the two explicitly new initializers are fitted.
