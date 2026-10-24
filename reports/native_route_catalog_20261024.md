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
