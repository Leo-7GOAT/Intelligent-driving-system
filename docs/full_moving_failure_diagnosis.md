# Seed22 recorded dynamic-policy diagnosis

The complete closed H endpoint can now reconstruct **889 recorded decisions**
across all24known conditions:525static and364moving decisions,eight moving
conditions. Recorded actor identities,frames,states and previous controls drive
observation reconstruction. Accepted reference poses must still match the native
road-query certificate. A geometrically successful trajectory rejected by the
dynamic check has no active reference and requires no acceptance query.

The first two attempts failed because the old static-only auditor assumed that
every successful spatial search had an accepted timed reference. Both failures
and the executed auditor snapshot are retained. The corrected auditor validates
the dynamic rejection mode and rejects any unaccepted retained reference;it does
not weaken the certificate required for an accepted reference.25related CPU
checks pass. The physical implementation and live training protocol are unchanged.

The frozen failure coefficient is+0.0001164842979 with scale1000. A failed-label
context therefore subtracts0.1164843from that label's logit. Across58decisions
with negative failure context,removing only this actor term changes **zero**
deterministic actions. In the timed-out crossing-walker case20,all51subsequent
negative-context decisions choose the already failed label2. Its mean probability
is97.6180%with the learned term and97.8743%with that term hidden. The policy's
preference remains dominant despite the small learned penalty.

All52recorded planning failures in this case are `open_set_exhausted`,not a
node-budget timeout or an MPC solver failure. This does not prove the continuous
road problem infeasible. Alternatives must be checked from the same recorded
state before deciding whether stronger selection feedback could help or the
vehicle already reached a pose with no available forward recovery.

These are same-weight actor counterfactuals,not the independently trained H0
policy. No alternative is driven,and no counterfactual completion gain is
claimed. Weights,full Adam,Torch CPU RNG and timestep counters remain unchanged;
observation RMS also remains fixed. This expanded replay applies to seed22 H
only;older models keep their previously documented replay scope.

For an already closed,raw-audited compatible endpoint:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NAVSTACK_AUDIT_SOURCE_ROOT='D:/ARCHIVE/ENDPOINT/source'
.venv/Scripts/python.exe scripts/audit_native_reference_inference.py --batch D:/ARCHIVE/ENDPOINT --method H --all-actions --output D:/ARCHIVE/new_full_replay.json
.venv/Scripts/python.exe scripts/diagnose_outcome_feedback_actions.py --batch D:/ARCHIVE/ENDPOINT --all-actions --output D:/ARCHIVE/new_mechanism.json
```

Existing outputs are never overwritten. The mechanism command includes its own
complete reconstruction;running the standalone audit first is optional.
[Full proof,probabilities and failure boundaries](../reports/native_seed22_failure_term_mechanism_20261105.json).
The endpoint remains23/24against24/24for zero-PPO;this is not a PPO benefit.
Research acceptance remains unmet.
