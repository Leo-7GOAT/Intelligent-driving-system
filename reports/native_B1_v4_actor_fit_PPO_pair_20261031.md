# Matched actor-fit-to-PPO endpoints

Both registered 20,480-control development continuations are complete. Every
condition and failure is retained. These are one-seed results on twelve known,
correlated development conditions, not independent final confirmation.

| Initialization | Actor-fit endpoint | Endpoint after PPO | Unsafe after PPO | Actual Adam steps | Verified controls, training plus evaluation |
|---|---:|---:|---:|---:|---:|
| Replay fit | 8/12 | 4/12 | 5/12 | 51 | 27,579 |
| Approach fit | 10/12 | 6/12 | 5/12 | 61 | 25,745 |

Both PPO endpoints have four bounds failures and one collision. Replay has
three deadlines, approach one. Core successes are respectively **1/4, 3/4,
0/4** and **2/4, 3/4, 1/4** for curve/parking, static and dynamic conditions.
Approach-only successes are cases 2 and 10; there are no replay-only successes.
The two-case difference does not establish a benefit across independent seeds.
Both are rejected as healthy full-task baselines, rather than used to claim a
hierarchy advantage.

`scripts/analyze_native_fit_ppo_pair.py` re-audits all **53,324** physical
controls, checks all input/model hashes and optimizer boundaries, and verifies
the common task, source, geometry, reward units and saved PPO schedule. Both
use 1,280-action rollouts, 320-sample minibatches, ten epochs, learning rate
3e-5, KL target 0.01 and frozen observation statistics. KL stopping produces
different actual Adam counts; these are explicitly reported. Input costs of
at least 44,440/44,131 controls are also retained separately. The actors received
different disclosed labels in the earlier matched fit experiment; this is not
a comparison between identically initialized policies.

Reproduce from the repository root with the completed/downloaded batch paths:

```text
python scripts/analyze_native_fit_ppo_pair.py --replay REPLAY_BATCH --approach APPROACH_BATCH --output NEW_COMPARISON.json
```

The lower-rate approach probe has now actually started. Its first checkpoint
binds 1,280 new native training controls, 40 actual Adam steps at 3e-6 and
unchanged observation RMS. Its logged minibatch KL is 0.007937. The fresh CARLA
trajectory differs slightly from the old run, and no final performance result
is available. The full 20,480-control endpoint remains the registered stopping
point. See `native_B1_v4_approach_lower_lr_first_update_20261031.json`.

The separate recovered seed22 H0 capsule is also now public and independently
reproduces all 15,887 segment controls, twelve measured-start actions and the
result PNG. Its original missing-probe guard snapshot and cleanup failure are
preserved, with original failure hashes and explicit private-path redactions.
See `H0_v4_seed22_recovered_public_verification_20261031.json` and
`H0_v4_seed22_recovered_public_failure_provenance_20261031.json`. Its inherited
lost work remains charged, so the recovered H/H0 pair is not an equal-consumed-
budget comparison. These publication checks add no driving or PPO updates.
