# Native development execution, 2026-10-28

This remains a development prerelease. The original research acceptance gates
are unchanged and unmet. These twelve known conditions are not independent final
confirmation; all failures remain in their original runs.

| Completed endpoint | Success | Actual new training controls | Interpretation |
|---|---:|---:|---|
| B1 cumulative102400 | 2/12 | 81920 in continuation | Seven bounds failures and one collision; rejected as healthy baseline |
| Frozen B2 body-spline | 7/12 | 0 | Better than legacy4/12; deployment change,not retrained B2 or equal-compute evidence |
| Shared-mask BC seed11 H/H0 | 11/12,11/12 | 0 | Same original physical mask; learned feedback ablated inside H0actor/critic |
| Shared-mask BC seed22 H/H0 | 10/12,12/12 | 0 | Both pass original complete-inventory initialization gate; no learned-feedback benefit established |

Reports: [B1](native_B1_long_PPO_endpoint_20261028.json),
[B2](native_B2_body_spline_endpoint_20261028.json),
[seed11](native_shared_mask_seed11_BC_endpoint_20261028.json),
[seed22](native_shared_mask_seed22_BC_endpoint_20261028.json).
Every completed endpoint has a per-method raw frame audit,normal guard exit and
owned-empty server stop. The interrupted B2 attempt's1486controls remain charged.

The B1 critic's final tanh-feature/linear-head absolute output upper bound is
7.1442, while an unsafe terminal event rewards -90. This is a measured symptom,
not a proven sole cause of failure. The next registered experiment changes only
the disjoint critic Adam group's rate from3e-5 to9e-4. Actor rate3e-5,all neural
tensors,normalization,RNG,environment configuration,physical runtime source,
rollout320,epochs10,KL target.01 and20480-control endpoint match control140111.
No trained optimizer is discarded. The new ledger charges the known7123parent
selection-evaluation controls omitted by the old control; both use the same
4329physical teacher controls. [Initialization equality/cost receipt](native_B1_critic_rate_initialization_20261028.json).

Actual sequential local queue as of09:36local:

1. B1 critic-rate batch013034: real training active,20480controls,7200s wall cap.
2. Common-mask seed11 H/H0 PPO013316: queued,20480controls per method,7200s.
3. Common-mask seed22 H/H0 PPO013440: queued,same budgets.

Each child requires normal predecessor completion,owned-empty server cleanup,
released lifecycle lease and unchanged prepared protocol before activation.
There is one CARLA server at a time,no automatic retry,and unchanged12GiB disk,
4GiB RAM and85C GPU limits. These are running local programs;later unimplemented
research is not represented as autonomous background work.

[Five fixed seeds have15verified actor-supervised initializers](native_fixed_seed_initializations_20261028.json).
This is not five independently trained CARLA policies. PPO value against the
strong12/12rule,healthy B1,feedback benefit,proper new validation/final partitions,
120new paired final conditions,unseen transfer,and remaining ablations remain open.

Full current local CPU regression:612passed,1skipped,4CARLA deselections in143.83s.
The30targeted critic/mask/PPO/cost checks also passed5.70s. Hosted CI and fresh
checkout checks for this source version must be recorded separately after push.
