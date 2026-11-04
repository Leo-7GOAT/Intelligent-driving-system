# Completed lower-rate native B1 development probe

The registered 20,480-control probe completes **10/12** known development cases,
with two bounds failures, zero collisions and zero deadlines. Core successes
are **4/4 curve/parking, 3/4 static, 3/4 dynamic**. Cases 7 and 11 fail on the
same right-turn route, respectively after 615 and 163 controls. The **16.7%**
bounds-episode rate still fails the 5% engineering safety target. These are
development results, not independent final confirmation or research acceptance.

The 3e-5 approach-initialized counterpart completes 6/12 with five unsafe
episodes; the new 3e-6 probe adds successes in cases 0, 3, 8 and 9 without
losing a counterpart success. Both start from exactly identical policy tensors,
normalization and RNG bytes and use the same physical/worker source, task,
reward and 20,480-control budget. Actual Adam counts differ substantially under
KL stopping: **61 versus 610**. This is a one-seed development improvement,
not evidence of an independent-seed effect. The comparison re-audits all
**51,244** raw training/evaluation controls and verifies model/input hashes.
See `native_B1_v4_approach_learning_rate_pair_20261031.json` and reproduce with:

```text
python scripts/analyze_native_flat_lr_pair.py --baseline BASELINE_BATCH --probe PROBE_BATCH --output NEW_COMPARISON.json
```

The actor-only initializer also scored 10/12, but on different cases: the PPO
probe repairs initializer failures 5/9 and introduces failures 7/11. Thus this
probe does not increase success count above its supervised initialization, and
does not establish PPO value over the teacher or the strong rule high level.
All 25,499 probe controls and twelve measured-start policy actions verify.
The owned server stops normally; the queued 36-training/24-development-validation
H/H0 batch then actually starts. No original task or safety gate is relaxed.

The first-update auditor now obtains its actual constant baseline learning rate
from the saved initializer rather than assuming 3e-5. Its old public 3e-5
regression reproduces all non-source/non-timing fields exactly. On the new
3e-6 rollout, all **1,280** stochastic sampled actions and every first-checkpoint
parameter reproduce exactly, including **40** Adam steps. Exact full-rollout
KL mean/max are **0.008356/0.170723**; the mean is not a maximum-KL bound.
The other in-memory rate branches are discarded, add no driving and are not
closed-loop performance evidence. See `native_flat_update_dynamic_rate_validation_20261031.json`
and `native_B1_v4_lower_lr_exact_first_update_20261031.json`.

For residual control diagnosis, a new explicit coverage profile queries B3/MPC
at absolute controls 0/40/80/120/160/200 of the first completed **training**
episode of each condition, regardless of outcome. All **72** fixed queries
produce bounded labels: 61 reference, three wait and eight dynamic-brake modes.
Original training frames and query/script/dataset hashes bind every sample.
No evaluation states, new teacher execution, PPO or supervised updates are used.
The mean normalized acknowledged-command disagreement is 0.168818. It is high
in condition 11 but small in condition 7's sampled early states; this does not
explain its later post-obstacle departure or prove that the oracle would recover.
The dataset remains diagnostic and is not deployed or fitted. See
`native_B1_v4_lower_lr_training_coverage_oracle_20261031.json`.
