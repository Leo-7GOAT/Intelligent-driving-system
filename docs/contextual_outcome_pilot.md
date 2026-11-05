> Delivery status, 2026-11-05: deferred by the user’s final three-hour cutoff. The waiting coordinator was stopped before any native job; prepared inputs are retained. No native result or benefit is claimed.

# Conditional execution-feedback development pilot

The seed22 recorded-state check finds a reusable terminal reference before the
policy replaces it with a nonterminal local reference. The current learned
reference/failure coefficients are global scalars. The new experiment tests
whether conditioning their residual contribution on the existing actor state
can distinguish when to retain or abandon an option. This is an unverified
development hypothesis.

`ContextualOutcomePolicy` keeps the signed31 observation codec, candidate set,
Hybrid A*, trajectory, MPC, safety mechanisms and task reward unchanged. A
32-unit actor branch predicts two free-signed feedback gains from existing
latent state and context. A separate critic branch models conditional value.
Both output layers start at zero. No action is forced, masked or replaced by
the new branches. Existing endpoint masks retain their original logits.

Migration preserves every original tensor and Adam state, normalization bytes
and saved RNG. Eight additional parameter states start with zero Adam moments;
the new hidden weights are random features, while their output layers are zero.
Initial action probabilities and values therefore match the original policy.
The two arms start from identical weights/Adam/RMS/RNG. Hc0 removes only the new
conditional branches; the older positive and negative feedback terms remain
available in both arms. This ablation differs from the current five-seed H/H0
cohort, which removes the new negative v1 terms in H0.

The pilot selects the already inspected seed22 canonical zero-PPO warm actor.
It is one exploratory pair, not new five-seed confirmation. Each arm receives
10,240 requested native training controls, at most319 rollout overshoot, the
same TRAIN48 curriculum and zero new supervision. A shorter pilot may not visit
all48 registered conditions; actual reset exposures must be reported. Each
frozen endpoint evaluates the same24 geometry-separated known validation
conditions. No new final-confirmation conditions are generated or inspected.

Preparation uses the immutable cohort's physical `navstack` source, adds only
the new policy/registration modules and admission gate, and inserts a narrow
saved-policy admission check into the frozen native worker. Current operational
scripts are snapshotted separately. The worker does not receive the root road
cache or another controller/planner version. Runtime physical config must equal
the canonical initializer config.

The queue waits on an authenticated PID/birth/executable handle for the current
matched B1 driver. It requires all20 distinct flat phases, every closed lifecycle
and matching protocol hashes before using2120. H/H0 may continue on2110. The
pilot never starts a third server. Every native phase needs4GiB free GPU,
16GiB free RAM and16GiB archive space at admission, with the existing watchdog,
4h phase limits and zero automatic retries. Queue wait is bounded at24h.
Queued activation preserves the original protocol and refreshes only its wall
window before the first lifecycle.

```powershell
.venv/Scripts/python.exe scripts/prepare_contextual_outcome_pilot.py --output D:/ARCHIVE/conditional_pilot --cohort D:/ARCHIVE/independent_cohort --flat D:/ARCHIVE/matched_flat --owner-pid CURRENT_VERIFIED_FLAT_DRIVER_PID
.venv/Scripts/python.exe scripts/run_contextual_outcome_pilot.py --output D:/ARCHIVE/conditional_pilot --validate-only
.venv/Scripts/python.exe D:/ARCHIVE/conditional_pilot/source/scripts/run_contextual_outcome_pilot.py --output D:/ARCHIVE/conditional_pilot
```

Use the actual live owner identity; never reuse a documented historical PID.
Preparation and validation do not start CARLA. The final command is a foreground
queue process and writes its waiting/execution status. Existing outputs are
preserved. Complete native raw/engine/optimizer/RMS audits and all recorded
static/moving NN decisions are required after execution. Prepared models and
lightweight optimizer checks are not evidence of native task improvement.

Local validation exposed a test-fixture limitation: its initial four-step
rollout contained only zero feedback, so it could not demonstrate learning in
the added branch. The corrected fixture starts with an actually planned,
reusable reference, asserts feedback exposure, verifies enabled-branch changes
and ablated-branch invariance, and checks save/load plus original-state isolation.
This short fixture test is not native driving or a performance result.
