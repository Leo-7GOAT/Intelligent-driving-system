# Native runtime and learned-feedback audit

Every recorded evaluation decision/control is included, including failed
episodes. These are twelve known development conditions, one trained seed.
No new driving, fitting or PPO updates are added by this audit.

| Endpoint | Success/12 | Unsafe | Controls | Environment decision P95 / P99 / max | Above nominal period |
|---|---:|---:|---:|---:|---:|
| H PPO v4 | 12 | 0 | 4,337 | 348 / 418 / 4,224 ms | 0.68% of439decisions;500ms period |
| H0 PPO v4 | 11 | 0 | 4,886 | 705 / 787 / 5,600 ms | 10.12% of494decisions;500ms period |
| B1 PPO v4 | 7 | 0 | 7,257 | 233 / 308 / 430 ms | 38.29% of7257decisions;50ms period |
| B1 approach actor fit | 10 | 2 | 4,942 | 199 / 264 / 410 ms | 49.80% of4942decisions;50ms period |

The timers measure the environment step, including its next observation, but
**exclude policy prediction, normalization and runner serialization**. The
control-cycle timer separately excludes planning and later candidate-feedback
work. Thus these are not full-runner latency measurements or a hard realtime
certificate. Quantiles pool actual samples; per-episode quantiles are not averaged.
H's low P95 does not erase its4.22-second outlier. Terminal sensor drain time
is retained in the measured cycle rather than deleted as an inconvenient tail.

Feedback computation alone has P95 **194/207/220/187ms**, respectively. B1's
common information includes the same candidate feedback despite its direct
actuator output. This is substantial compute cost at20Hz, not a planner call
or hidden MPC execution in B1. An exact-output compute optimization is warranted
before larger future campaigns; active and queued frozen sources remain intact.

| Endpoint | Stationary frames, abs(v)<0.05m/s | Actual brake frames | Non-reference control modes |
|---|---:|---:|---|
| H PPO v4 | 347 (8.00%) | 340 (7.84%) | dynamic brake290, failed-plan brake10, policy wait2, terminal brake2 |
| H0 PPO v4 | 1188 (24.31%) | 1170 (23.95%) | dynamic brake290, failed-plan brake460, policy wait394, terminal brake1 |
| B1 PPO v4 | 2630 (36.24%) | 2607 (35.92%) | All7257controls remain direct policy commands |
| B1 approach actor fit | 440 (8.90%) | 478 (9.67%) | All4942controls remain direct policy commands |

Stationary frames include starts and legitimate pedestrian yielding; this
threshold alone does not diagnose stagnation. All episode durations, failures
and success-only travel times remain in the JSON reports. H's completed trips
last15.25–22.50simulation seconds; H0's case5 reaches the45-second deadline.
Braking, policy waiting, terminal braking and shield intervention are distinct.
Shared external shield interventions are zero in these endpoints. H/H0 still
have their disclosed internal dynamic/planning fallbacks, so native system
comparisons cannot establish a pure learned-policy safety advantage over B1.

At each of the twelve final sensor refreshes per method, the trace's refreshed
evidence omits the earlier control report. The analyzer joins the original
audited native response by **episode and exact frame**. All frames regain an
actual actuator report; none is dropped or imputed as zero brake.

The separate learned-input intervention reproduces **all439 H and494 H0
decisions** from measured histories. With weights, observation statistics and
the common physical action mask fixed, hiding feedback changes H's selected
action in **10/439decisions** across cases1/3/4/7/10/11. Mean probability total
variation is0.03245. H0, already hiding that feedback, gives **zero action or
probability changes** as a negative control. All9,223underlying control states
are checked and no counterfactual command is executed.

This establishes limited policy sensitivity to learned feedback. It does not
establish a performance benefit: H's exclusive success case5 is not among those
changed decisions on H's own visited states. The counterfactual would visit
different later states if executed. Five-seed retraining ablations and new
confirmation conditions remain necessary; original research acceptance is UNMET.

Evidence: `native_H_v4_seed11_runtime_20261031.json`,
`native_H0_v4_seed11_runtime_20261031.json`,
`native_B1_v4_PPO_runtime_20261031.json`,
`native_B1_v4_approach_fit_runtime_20261031.json`,
`native_H_v4_feedback_reliance_20261031.json`,
`native_H0_v4_feedback_reliance_20261031.json`.

Reproduce runtime statistics with
`python scripts/summarize_native_runtime.py --batch <complete-batch> --method H --output <new.json>`.
For reliance, set `NAVSTACK_AUDIT_SOURCE_ROOT` to that batch's frozen `source`
and run `python scripts/audit_native_feedback_reliance.py` with the same arguments.
The dynamic-history, shared-mask and terminal-frame regression subset passes
14tests in8.36seconds. Actual complete native input reconstruction is reported
separately from CPU tests.
