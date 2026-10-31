# Completed replay-initialized B1 PPO regression

The registered 20,480-control endpoint finishes 4/12 known development cases,
with four bounds failures, one collision and three deadlines. Its actor-fit
initializer finished 8/12. The trained endpoint is rejected as a healthy baseline.
All 27,579 physical controls and twelve measured-start actions verify; the
owned server was cleaned up normally. No failed episode is excluded.

`audit_flat_first_ppo_update.py` reconstructs the first 1,280 native training
decisions from measured histories, exact initializer normalization and saved
RNG. Every sampled action matches. After applying the registered fixed 1/20
learner reward units, the original two Adam steps reproduce **every parameter
in the first checkpoint exactly**, as well as its PPO metrics.

| Constant learning rate | Actual/discarded Adam steps | Exact whole-rollout KL mean | KL max |
|---|---:|---:|---:|
| 3e-5, reproduced actual update | 2 | 0.042839 | 0.777710 |
| 1e-5, discarded offline branch | 3 | 0.018322 | 0.258190 |
| 3e-6, discarded offline branch | 40 | 0.008218 | 0.284747 |

The original KL target is 0.01. SB3's minibatch early stopping does not enforce
a bound on the final full-rollout distribution; the reported minibatch estimate
for the original update is 0.020229. Over all sixteen actual phases the logged
KL median is 0.012189 and maximum 0.079229. Eleven phases exceed 0.01.
This supports testing a smaller learning rate while keeping the action space,
reward, rollout, termination and safety criteria fixed. It does not prove that
large updates caused every failure or that smaller updates improve driving.
Smaller learning rates can execute more optimizer steps before early stopping;
their compute costs must be reported separately.

The separate terminal-history audit reproduces all 7,099 evaluation controls
and all twelve final actions. Initial and trained policies are evaluated on
the trained policy's visited terminal states; those initial-policy predictions
are **not** counterfactual closed-loop results. Remaining braking, steering and
collision failures are preserved rather than attributed solely to update size.

The first reconstruction attempt omitted fixed reward units and failed its
checkpoint equality guard. That diagnostic was corrected; it never changed a
deployed model. Alternative in-memory optimizer branches are discarded and add
zero driving controls or deployed PPO updates.

Evidence: `native_B1_v4_replay_fit_PPO_endpoint_20261031.json`,
`native_B1_v4_replay_first_update_lr_audit_20261031.json` and
`native_B1_v4_replay_PPO_terminal_drift_20261031.json`.
Run the first-update audit with `NAVSTACK_AUDIT_SOURCE_ROOT` set to the completed
batch's archived `source`, using a fresh output path. Preparation now exposes
`--freeze-observation-statistics` as an explicit registered continuation profile;
37 supervised-preparation, entry-gate and replay checks pass.

All results here are development diagnostics. The main five-seed/new-condition
research acceptance remains UNMET.
