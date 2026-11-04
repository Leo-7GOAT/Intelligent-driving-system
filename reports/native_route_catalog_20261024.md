# Route-conditioned candidate development protocol

## Decision before the first run

The unchanged ego catalog scores3/4 under the existing rule and0/4 under H.
The rule's fourth road reaches a state1.95m from the goal with no moving
production reference. Earlier route-candidate offline probes did not repair
already-failed terminal states. This version tests whether following route
tangents earlier and explicitly targeting the terminal pose prevents that loss.
It does not assume the proposal works or treat rejected search as infeasibility.

`route_frame_terminal` activates the existing route proposal in the actual
environment: nearest-segment projection determines candidate arc length only;
lookaheads6/9/12m, five offsets and two speeds retain31 choices including wait.
Targets use route tangent headings and the existing terminal offset envelope.
The route endpoint must match the task goal. A target at that goal receives a
stopping trajectory using existing time parameterization. Measured state is
never projected, moved or assigned a speed. PPO still selects the action; no
mask or rule substitution is introduced.

The known map route is additional prior information. It is exposed through the
same candidate geometry to H, H0, B1 and the rule comparator. All other measured
road/task information, rewards, termination, calibration, Hybrid A*, MPC and
physical/decision periods remain as in the completed native rollout experiment.
The policy signature changes even though input size stays4701. Old weights and
normalization statistics must not be resumed under the new semantics.

First evaluate the rule on all four existing training roads with the same reset
seeds7301..7304. The rule uses existing goal-frame connector scoring and at most
three bounded fallback searches, now on the route targets. This also changes
the heuristic's feasibility check relative to ego arcs, so any improvement is a
catalog-plus-rule-interface development result, not isolated PPO or catalog
causality. Charge actual selection and complete decision-cycle P95/P99/max and
overruns. Preserve every outcome and raw frame. Use a fresh30-minute owned-server
batch, disk>=12GiB, RAM>=4GiB, GPU<85C, no-progress300s, zero automatic retries.

If coverage completes all four tasks, start new H0/H/B1 policies at seed11,
20,480 nominal actual training controls per method,32-action rollout, batch32,
two PPO epochs, global clipping and unchanged rewards. H0 ablates the99 feedback
values while retaining route geometry; H and B1 receive the same enabled
feedback. Budgets include actual complete-rollout overshoot. Record train and
frozen four-road endpoint probes separately; this is an initial development
budget, not convergence, independent seeds or a final confirmation claim.
If rule coverage fails, retain that failure and diagnose before spending the new
training budget. The already-exposed roads cannot become final test conditions.

No old batch or frozen source is modified. The prior rule3/4 and learned0/4
results remain part of the record. Final requirements for a healthy flat
baseline, strong rule comparison, five seeds and fresh confirmation are unchanged.

## First route-only result and registered continuation variant

The route-only rule run `20261024T105443Z_native_rule` actually completes3/4,
with1,993 applied controls and2,116 worldticks. Roads0/1/2 succeed in359/381/353
controls; road3 ends at its900-control deadline. Raw frame audit passes, owned
cleanup completes and the fresh-PPO staging gate rejects training with no batch
or pointer created. This is a failed coverage hypothesis, not an improvement.
Root source757b08f passes228 tests with5 conditional skips; the independent
checkout atd307fb1 passes23 targeted route/evaluator/staging tests.

Recorded road3 decisions13/17/18 choose28/18/22. Offline production queries at
those measured states find available local references with a target-to-goal
connector (respectively5/6/15/16/25/26,5/6/15/16,5/6/15/16), but the selected
targets lack that sufficient connector. The initial selected26 does have one.
This motivates a separate continuation variant, not a claim of causality or a
guarantee that a proposed endpoint equals the future executed state.

Before its first run, fix the following variant: `execution_feedback=continuation`
replaces the derivable distance/speed time proxy in each candidate's third
feedback slot with a static target-to-goal connector witness. Exact goal pose
with a free vehicle footprint is a positive zero-length witness. Candidate
geometry/speed retain the inputs needed to derive the old proxy. A distinct
signature prevents old-model reuse. H0 uses `continuation_ablated` and zeros the
same99 feedback values; H and B1 share the new features. No action is masked.

The corresponding rule adds a fixed5m score penalty when this witness is absent.
This finite preference retains the original ordering when all witnesses are
absent and still allows bounded search. The penalty is not tuned on final tests.
All other route-catalog conditions and the prospective four-road coverage gate
remain unchanged. The next diagnostic is a new batch; the failed route-only
result is retained. Both extra observation and rule selection costs are measured.

## Continuation execution status — 19:18 local

Sourcef5108d0 passes232 full regression tests,5conditional skips and2existing
deprecation warnings;12 targeted tests also pass. Actual continuation batch
`runs/20261024T110908Z_native_rule` is running under its owned process guard.
First three conditions have completed; fourth is still recording. No fresh PPO
batch has started. Current identities and deadline are in the handoff.

The failed route-only raw audit independently reproduces atd307fb1, SHA256
`fefc82ede87a5e93c9272e2b1023bcf5ac75dc4df9e8e0f95d926782e416cd87`.
Earlier fixed-reference4/4 control used1.0m/s and retained a complete reference;
the catalog rule commonly chooses1.5m/s and replans every0.5s. Those successes
cannot isolate speed, tracking and replanning effects in the present failures.

## Prospective rule speed contrast

Before running it, register `rule_speed_policy=paired_low_speed`: first obtain
the unchanged rule proposal, then select the identical-pose catalog entry at
0.75m/s. Wait remains wait. Both the original proposal and executed selection
are written to every trace row, and the raw verifier checks the registered
mapping. This is an explicit rule-only speed ablation, never a PPO action
override. The existing catalog, vehicle limits,45s task deadline, rewards,
route/continuation inputs, planning, MPC and all four conditions stay unchanged.
The lower speed is already one of the31 policy actions; no new easier task is
introduced. Geometry may evolve differently in the new closed-loop rollout.

Motivation: fixed-reference native control at1.0m/s previously finished4/4,
while the catalog rule generally selects1.5m/s and replans. Holding references
and reducing speed are distinct hypotheses. This run changes only B3's speed
selection, so its result can narrow the diagnosis before changing low-level
tracking or replanning. Run only after the active continuation batch completes
and its owned server is cleaned up. Keep the same30-minute/resource guards,
retain all failures, and require the unchanged4/4coverage gate before fresh PPO.
The gate is development scheduling, not a research acceptance criterion.
