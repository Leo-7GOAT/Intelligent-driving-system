# H waiting-state input diagnostic — 2026-10-25

The actual frozen H endpoint finishes2/4known roads after20,665CARLA training
controls. Its two failures select action0/wait despite zero planning/MPC
failures or counted takeovers. The [complete endpoint report](native_route_checkpoint_20261025.md)
retains all outcomes and the one training bounds episode. This additional
diagnostic executes **no CARLA or local vehicle controls and no optimizer
updates**. It investigates a decision mechanism, not a repaired driving score.

## Inputs and intervention

The input trace is the complete raw-audited final H evaluation, model SHA256
`5926e13b49f289e4b848b10cce078d951c7d7f49b379cf0f49634e1ca317a8d4`,
normalizer SHA256
`ee3d4079c1132a32ddf32901832d404fb88b90addb1f89d8cf299e2d341c1fd4`.
Every actual action at all248decisions was already reconstructed from measured
state and observable history. This diagnostic selects initial and before-last
states from **all four roads**, adding first-wait and terminal-wait-start states
where present: **11distinct states** after deduplication. Decisions below are
zero-based. The reconstructed initial actor has the exact training seed and
architecture; its L2 distance to the final actor matches0.1835962832 in the
training log. Both sets of weights use the same final H frozen normalizer.

Four input conditions are compared for both weight sets: actual inputs, zero
candidate witnesses, zero previous execution outcome, and zero both feedback
blocks. Candidate witnesses are31slots×3features=93values at offsets5..7
of each eight-value candidate row. The six outcome values are599:605. The
complete ablation changes99values; the candidate poses/speeds, ego/goal/history,
road raster and four measured actuator states are checked unchanged in both
raw and normalized inputs. No action is masked or replaced during execution.

This is **inference-only, possibly out-of-distribution feature perturbation**
of one model. It is not retrained H0, a new independent seed, an estimate of
task-performance benefit, or evidence that removing feedback fixes driving.

## Observations

At all11states, zeroing only the six outcome features leaves the trained
argmax unchanged. At all six selected actual wait states (including the
successful road0's final braking state), zeroing93candidate-witness values
changes the trained argmax to a moving candidate. Zeroing all99values gives
the same actions there. Under actual inputs, the **initial weights also select
wait at every one of those six states**. The wait ranking is therefore present
before training on these recorded states; it cannot simply be described as a
confident learned wait rule. Training does change other selections, including
road1's initial wait→moving28, so this is not a claim of no policy learning.

For the five inspected wait states on failed roads, the unchanged production
planner can build the alternative candidate's reference:

| Road | Decision/stage | Actual action | Witness-zero action | Search nodes | Reference samples |
|---|---|---:|---:|---:|---:|
| 1 | 25, first/terminal wait start | 0 | 12 | 60 | 61 |
| 1 | 89, before last | 0 | 12 | 69 | 61 |
| 3 | 24, first wait | 0 | 12 | 36 | 55 |
| 3 | 72, terminal wait start | 0 | 2 | 5 | 25 |
| 3 | 89, before last | 0 | 2 | 4 | 21 |

All five queries return `reference`/successful plan and a terminal-goal prefix.
The sixth query, successful road0's final state, returns the appropriate
`parking_brake` with no trajectory. Each query asserts unchanged measured
vehicle state and an observation-only plant without a `step` method. A valid
queried reference is not an executed success, a full-path feasibility proof,
or a guarantee against later planner/controller/task failure.

At the five failed-road wait states, actual wait probability is only
**0.0323673..0.0323977**, near uniform1/31≈0.0322581. The winner/runner-up
probability gap is **2.4542e-5..1.0837e-4**. Zero-witness KL from actual is
**1.1625e-6..3.3019e-6**. Tiny changes in a nearly uniform distribution can
change a deterministic argmax; counting argmax flips alone overstates policy
confidence and says nothing about cumulative driving return. Full31-action
probabilities are retained for every condition/state/weight set.

The evidence rules out universal absence of candidate references at these five
inspected states. It supports investigating weak learned ranking and the
influence of candidate witnesses, while retaining the exact optimizer-replay
evidence and mixed endpoint outcomes. It does not justify disabling feedback
in the active policy or claiming a feedback benefit. Actual initial/trained
weight controls and any later deployment-only feature ablation must be
separately declared and executed after the original B1/owned-server lifecycle.

## Reproduction and numeric checks

Local canonical report:
`runs/20261025T085433Z_native_route_learning/H/analysis/feedback_input_interventions_v2.json`.
SHA256:
`15428f3fea4080bdfdc97837d73899143f33158ce698f3caf06759f7077c4947`.
Frozen diagnostic source is alongside it as
`feedback_input_intervention_v2_source.py`; it also uses the repository's
`scripts/evaluate_native_policy_weights.py` initialization helper and the
parent's unchanged frozen `source/` implementation. All source, model,
normalizer, configuration, raw-audit and trace identities are checked first.

A separate checkout7605ba0 with its fixed dependency environment receives a
bounded copy of immutable completed H analysis inputs, **not a complete
training asset**. It executes the copied frozen diagnostic and reproduces the
entire JSON byte-for-byte. Its inputs manifest is
`runs/feedback_input_reproduction_inputs_20261025.json`; local receipt is
`H/analysis/feedback_input_independent_reproduction.json`. Both runs have zero
new vehicle controls, optimizer updates or normalizer updates.

Canonical KL is computed after separately normalizing both probability arrays
in float64, with finite/nonnegative-with-roundoff assertions. The v1 prototype
and source are retained; its direct float32-sum accumulator is not used for
numeric conclusions. No failed execution or prior prototype was deleted.
