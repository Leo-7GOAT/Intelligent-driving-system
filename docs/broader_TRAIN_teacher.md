# Broad TRAIN-only teacher preparation

Collection is now complete and audited: **35/36 TRAIN successes**, with one
static-vehicle deadline at condition21 retained. Curves and walkers each12/12,
static11/12. All12,556actualcontrols have matching synchronous engine controls;
1,115reset ticks are charged separately. No held-out validation or final labels
were used. [Collection evidence](../reports/native_B3_broader_TRAIN_teacher_collection_20261102.json).
Direct-view reconstruction is a separate read-only stage; collection success
does not establish a trained direct-policy result.

The new flat-PPO development courses remain below the target: seed22 is 7/24,
seed11 and seed33 each 11/24,and recovered44 15/24 on known validation conditions. Extra supervision is a separate
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

The separate actor-fit observer is now registered and waiting for this teacher
and its direct views to finish successfully. It then uses every fixed final
broader B1 endpoint (all five seeds, including the failed basic gates), with
1,024 auxiliary Adam steps each at learning rate 0.0001. Successful TRAIN
teacher episodes supply case-balanced action labels; failed teacher episodes
remain in the source inventory and are excluded from fitting by the fixed rule.
Every core family must have successful teacher coverage. The original
every-fifth-frame split remains a correlated fit diagnostic, with no holdout
gradients or checkpoint selection. Teacher returns are never value targets.

Only actor weights change. Critic, exploration, PPO optimizer and counters,
observation RMS and checkpoint RNG bytes are retained. No native driving or PPO
updates are launched by this observer. All five fit results must be reviewed
before a separately registered learned-policy experiment. The fit has a
600-second CPU limit per seed, a 12-hour overall dependency/fit limit and no
automatic retry. [Original fixed registration](../protocols/native_B1_broader_measured_actor_20261102.json).

The original broader queue stopped when seed44 reached the disk protection
threshold. Both wait-only downstream observers stopped before dispatch; no
teacher controls or actor fits occurred. After verified artifact relocation,
seed44 completed its remaining 8,960 controls from the saved 11,520-control model,
ending at15/24 on known VAL. The untouched seed55 is now running. Replacement teacher and fit observers wait on this
explicit recovery queue; the teacher physical protocol and actor-fit budget are
unchanged. The fit uses seed44's fixed recovered final endpoint and retains the
interruption disclosure. [Current fit registration](../protocols/native_B1_broader_measured_actor_v2_20261102.json),
[recovery registration](../protocols/native_B1_broader_disk_recovery_20261102.json).

```powershell
python scripts/teach_broader_b1_cohort.py --output <fresh-execution-directory>
python scripts/teach_broader_b1_cohort.py --output <same-execution-directory> --run
```

These commands require the existing registered local cohort and teacher jobs;
they are not a replacement for the public frozen-model reproduction workflow.

After all five fits finish, `scripts/prepare_measured_actor_basic_probe.py`
can stage the fixed canonical seed22 model on the original four TRAIN roads.
It requires a research review bound to all five completed fit/checkpoint hashes;
the decision is fixed to canonical22, with no best-seed selection. It preserves
the frozen direct-policy runtime and low-level source, records the additional
supervision, and sets native PPO updates to zero. Preparing or testing this
entry point does not start a probe. Its review and evidence guards are covered
by eight tests; the partition checks also pass (25 tests together). A subsequent
actual driving result is still required before claiming basic qualification.
