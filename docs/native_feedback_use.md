# Measured-input feedback sensitivity — 2026-11-02

The five previously completed H/H0 pairs were examined without additional CARLA
controls or optimizer updates. Each method uses its own measured evaluation
history, frozen source, model and normalization. Every sixteenth decision plus
each case's first/last decision is sampled; all sampled original greedy actions
reconstruct exactly. The underlying 40,814 H and 41,089 H0 control-frame histories
are accounted for, including measured dynamic actors and previous execution state.
These remain 24 known development conditions.

| Learned inputs hidden | H changed actions / 446 | H mean probability total variation | H0 changed actions / 448 |
|---|---:|---:|---:|
| Candidate feasibility witnesses | 17 (3.81%) | 0.04199 | 0 |
| Previous measured execution outcome | 7 (1.57%) | 0.01463 | 0 |
| Both | 11 (2.47%) | 0.04006 | 0 |

Inputs are zeroed **after normalization**, matching H0's learned-input ablation.
The original physical endpoint availability mask is retained exactly for every
counterfactual. H0's probabilities are exactly invariant, providing a functional
negative control. H has between 9 and 31 available actions in the sampled states
(mean 27.77); no sampled state is forced to a single choice by that mask.

Feedback reaches H's learned actor and changes its distribution. Most sampled
greedy actions remain unchanged. Neither observation proves that the changed
decisions help or hurt physical driving: these are retrospective input
interventions, not counterfactual rollouts. The original performance comparison
still has a zero-crossing 95% interval, and added PPO value against strong B3
remains unestablished. This diagnosis supports no new superiority claim.

[All five seeds, input hashes and individual results](../reports/native_feedback_use_cohort_20261101.json).
The report retains all ten method/seed diagnostics; none is selected by outcome.

## Planning failures are episode concentrated

The earlier aggregate H/H0 planning-failure counts are 8/78. Recounting every
actual recorded decision mode across all ten method/seed endpoints shows:

| Seed / condition | Method | Outcome | Planning failures |
|---|---|---|---:|
| 11 / 14 | H | Success | 8 |
| 44 / 15 | H0 | Deadline | 78 |
| All other policy/seed/condition episodes | Both | Retained in the report | 0 |

All 78 H0 failures occur in its single timeout. Across explicitly mutually
successful paired conditions, H/H0 counts are 8/0. That conditional sensitivity
analysis does not remove the timeout from completion statistics: H remains
120/120 and H0 119/120. Neither the aggregate 8/78 count nor a selected successful
subset establishes a stable planning-efficiency benefit. H is slightly slower
in mutually successful conditions for every seed in the existing cost report.

[All-outcome and conditional counts, trace hashes and failed episodes](../reports/native_feedback_planning_concentration_20261102.json).
Every summary count is checked against actual `plan_failed` decision modes;
the complete report rebuilds exactly. No additional learning or driving occurs.

```powershell
python scripts/analyze_feedback_planning_concentration.py --output <fresh-report.json>
```

## Decision timing has distinct scopes

Historical PPO `decision_cycle_times` starts inside the environment and excludes
normalization and policy prediction. B3's `full_decision_times` includes rule
selection plus the environment step. Those fields cannot be directly compared
as complete decision latency; no PPO compute advantage is established by that
comparison. Missing historical prediction times are not reconstructed or filled.

Newly prepared PPO evaluations now record `policy_prediction_seconds` (including
normalization) and `full_decision_seconds` (prediction plus the complete environment
step) per decision. Per-case `policy_prediction_times` and `full_decision_times`
summarize count, median, P95, P99, max and over-period fraction at the actual
configured decision period. Setup and trace serialization remain outside that
scope, as in the rule evaluator. The old environment-only timing remains intact.

This logging change does not change actions, physics, rewards, termination,
training, or already frozen batches. Unit checks cover both discrete and direct
continuous actions, exact single prediction/step invocation, preprocessing and
full-decision boundaries. The related partition/cohort/rule suite passes 27 tests.
Future measured timing still needs normal source/hardware/load disclosure; a
timing field or synchronous simulator execution alone is not real-time validation
or evidence of added PPO value.

To reconstruct a per-method diagnostic from restored public cohort data:

```powershell
python scripts/diagnose_native_feedback_use.py --batch <restored-batch> --method H --output <fresh-output.json>
```

Repeat for H0 and all five registered batches using the filenames recorded in
the report, then rebuild the summary:

```powershell
python scripts/summarize_native_feedback_use.py --inputs <diagnostic-directory> --output <fresh-summary.json>
```

The [public cohort recipe](../protocols/public_native_matched_broader_feedback_cohort_20261101.json)
identifies each original capsule. Each diagnostic uses a fresh frozen-runtime
interpreter, verifies source/model/trace hashes and checks unchanged parameters
and normalization. No new simulator world is opened.
