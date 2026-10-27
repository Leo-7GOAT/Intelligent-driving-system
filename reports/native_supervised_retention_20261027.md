# Real PPO retention after supervision (2026-10-27)

The teacher-initialized direct model starts at3/4known full static roads.
After20,480actual CARLA training controls/640PPO Adam updates its frozen endpoint
is0/4: three deadlines and one bounds failure. All23,427training/evaluation
controls verify against measured bridge frames. Teacher data costs1,926controls
and896supervised updates are separate; charged physical-control lower bound is
22,406. This is one correlated seed on development conditions.
[Complete cost/model/frame ledger](native_supervised_retention_20261027.json).

On the same teacher examples, action MSE rises from0.0000597before PPO to0.001735
at2,560controls and0.01299at the final endpoint. Correlated holdout MSE rises
from0.001272to0.01417. Observation means, variances and count are exactly unchanged.
This establishes loss of the initial behavior on those inputs and rules out
observation-statistic drift in this phase. It does not isolate every cause of
failure or establish unseen-condition performance.

The next diagnostic031740restarts from the SAME supervised actor/critic,
untouched PPO optimizer, observation-normalizer file and RNG payload. All33policy
state tensors are bit-identical to the sparse control's initialization. Vehicle,
road, obstacles, continuous20Hz simulated actions, PPO schedule and fixed nominal
20,480-control budget are unchanged. It uses the existing shared guarded-progress
profile: global-distance progress0→0.5per metre, collision−30→−90and bounds−20→−90.
These three changes are reported as one reward profile, not attributed solely
to progress shaping. Termination/success tolerances are unchanged. Its bounded
objective argument is not a driving-safety guarantee. The profile remains
available to H/H0/B1; no formal hierarchy comparison is made by this B1diagnostic.

H/H0are also initialized from the same B3physical teacher data: all1,926controls
supply194macro decisions, including the failed road. Each receives896actor-only
supervised steps. Every fifth decision is excluded from normalization/gradient
fitting; its accuracy is only correlated fit evidence. The critic and PPOoptimizer
remain untouched; neither policy has new PPO training. Their actual CARLA
endpoints are still pending in032549. Different label rates, losses and compute
are disclosed rather than presented as identical optimizer workloads.

The first failed PPO endpoint remains unchanged and is not replaced by whichever
reward variant performs better. No final confirmation conditions were used.

```powershell
python scripts/prepare_native_supervised_ppo.py --parent <complete_audited_supervised_B1> --controls 20480 --wall-seconds 7200 --guarded-progress
python scripts/prepare_native_hierarchical_bc.py --parent <complete_audited_B3> --steps 896
```

Preparation is not execution. Use the registered existing batch instead of
repeating a parent. Full research acceptance, a sufficiently trained healthy
PPO baseline, five independent seeds,120new paired CARLA conditions, and a
validated PPO high-level contribution remain open.
