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
