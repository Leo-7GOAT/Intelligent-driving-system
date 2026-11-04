# Interrupted independent-cohort recovery

On 2026-11-05 the recorded worker, guard, server and coordinators were absent.
Their last observations were around 23:10 on November4. The cause is unknown;
this is not recorded as a successful overnight run or a machine reboot.

Seed22 H retained12,660 training controls and400 Adam steps. Readable traces
contain12,780 controls,including120 after the saved learning state. The final
gzip stream lacks its end marker. Preserve the original bytes and report the
unflushed tail as unknown rather than treating the readable prefix as complete.

The recovery copies the exact model,normalizer,full optimizer and saved RNG.
It starts a fresh native episode at the saved curriculum cursor. The remaining
lineage budget is18,060 controls,with the original one-rollout stopping rule.
All interrupted interactions remain charged separately. This is a recovered
learner,not an additional independent seed or exact uninterrupted-budget proof.

`scripts/prepare_independent_outcome_recovery.py` requires the existing fresh
process-absence receipt,interruption audit and unchanged frozen source. It adds
only the narrow training-admission rule;the plant,planner,trajectory,MPC,reward
and policy code remain the parent's frozen bytes. The original attempt and six
completed jobs are preserved. The resumed coordinator skips authenticated
completed-job receipts and proceeds with the remaining registered phases.

The live invocation uses `scripts/run_independent_outcome_cohort.py run` with
`--cohort`, `--archive` and `--recovery-batch`. Its new execution directory is
`execution_resume_20261105`;the original completion markers remain the common
dependency interface. Existing execution directories prevent duplicate launch.
The B1 observer uses `scripts/run_matched_flat_cohort.py run` with `--output`
and `--resume-dependency`,which names the verified new coordinator identity.
It creates no native B1 job until all25 H/H0 phases close normally.

First saved resumed-update proof:947 new controls,all34 Adam states advance30,
and actual initial parameters,full Adam and Torch RNG match the saved checkpoint.
This verifies resumed learning only;the remainder and independent endpoints
are still pending. [Receipt](../reports/native_cohort_recovery_20261105.json).

The separate B2 deployment batch was prepared before this interruption:
501 frozen input files pass launcher validation and a recorded-start frozen
actor restore produces a Grid-selected timed reference. It has not driven in
CARLA. Its spatial search omits Hybrid's steering primitives and exact memo;
the common measured-steering retimer,road checks,MPC and safety remain active.
The runtime schema stays H to retain checkpoint semantics;the experiment is
explicitly labelled B2 by `reference_spatial_planner`. Source e1e032c passes
hosted CPU regression with1,431 passes,25 skips and4 CARLA exclusions.

Research acceptance remains unmet. Neither recovery nor these CPU checks prove
PPO benefit,five-seed superiority,unseen-condition performance or real-time use.
