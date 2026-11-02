# Broad TRAIN-only teacher preparation

The new flat-PPO development courses remain below the target: seed22 is 7/24
and seed11 11/24 on known validation conditions. Extra supervision is a separate
development intervention to investigate, not an established fix or PPO gain.

Existing H/H0 TRAIN trajectories were checked as a way to avoid new collection.
All ten streams account for 205,908 historical TRAIN controls; validation traces
are never used as label inputs. Requiring a complete successful episode and
matching engine control acknowledgements throughout leaves coverage of only
31/36 TRAIN conditions. H alone covers 25/36. No dataset or model fit was emitted.
[Exact source hashes and eligibility results](../reports/native_broader_teacher_reuse_preflight_20261102.json).
The raw state/request audit of these older interface streams must not be read
as a claim that every requested control was applied in its corresponding frame.
The existing physical outcomes remain preserved.

```powershell
python scripts/audit_broader_teacher_reuse.py --output <fresh-receipt.json>
```

Restored public cohort data can use a `--manifest` with all five registered
seed pairs and their restored batch paths. The complete receipt rebuilt
field-for-field. No new controls, labels, optimizer steps or final cases result.

A fresh **uniform B3 teacher collection on all 36 TRAIN conditions** is now
registered, using the current synchronous delivery interface. The physical
source/calibration is frozen from the completed current B1 pilot; the complete
rule configuration equals the previously verified strong B3 configuration,
including endpoint screening and shared terminal parking assistance.
[Locked preparation and limits](../protocols/native_B3_broader_TRAIN_teacher_20261102.json).

The observer waits on the exact existing broader-cohort process. After all four
courses complete and are audited, it collects all 36 conditions once, preserves
every failure, audits raw frames, and requires matching pedals, gear and explicit
sync acknowledgement for every control. It then builds direct-control views
with unchanged frozen observation code and actual measured states; no plant
integration creates data. Collection is limited to 3,600 seconds, conversion to
1,800 seconds, with existing resource guards and no automatic retry.

The rule tool's operational output is named `evaluation_trace.jsonl.gz`, but
its immutable scene manifest and `data_role=TRAIN_EXTRA_SUPERVISION` contain
only the 36 TRAIN conditions; no `native_validation` partition is present.
This is teacher-collection cost, not PPO interaction or held-out evaluation.
Model fitting and subsequent native PPO are **not started by this preparation**.
The future consumer must disclose the extra supervision and its physical costs.

Executed teacher braking becomes a nominal direct-policy braking label. Copying
the teacher's helper-intervention flag would incorrectly count another intervention
when direct control already requests full braking. A new physical replay unit
test confirms the existing conversion behavior; production `measured_demo.py`
was left unchanged after the contrary hypothesis failed that test. The direct
replay can finish earlier than the teacher because its nominal brake is not an
intervened frame. These views and teacher returns are not counterfactual B1
episodes or valid direct-policy value targets. Related regression: 34 passed.

An initial zero-control preparation omitted endpoint screening; the preflight
caught it and it is blocked as abandoned. A wait-only observer was replaced
before dispatch while checking the parking-history hypothesis. No active CARLA
worker was stopped, and the valid replacement physical protocol is unchanged.
[Current process identities and execution paths](handoff.md).
