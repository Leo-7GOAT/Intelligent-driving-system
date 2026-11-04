# Independent native outcome-feedback development cohort

This cohort addresses the missing independent training evidence. It does not
replace final confirmation or the still incomplete fair H/B1 comparison.
The earlier continued seed22 H/H0/B3 study remains unchanged: every method
completed the same54known entries,without a PPO completion advantage.

All five canonical independently fitted H actors (seeds11/22/33/44/55) start
at zero PPO controls. Their inherited TRAIN-only supervision and interaction
costs remain charged. A zero-effect30-to34-parameter migration appends the
existing reusable-reference and action-aligned failure branches. Empty Adam
states are materialized as zero buffers;this preserves the first original
parameter update,as checked by regression. Within each seed,H/H0 begin with
equal weight tensors,Adam,RMS and saved RNG. H0 disables only the new negative
failure branches. The original reference and geometric feedback remain visible.

The shared calibrated physical lowlevel is frozen to the completed B3version.
Each arm receives30,720new actual CARLA controls,with at most319rollout overshoot,
on the existing48TRAIN conditions. The existing64-exposure rehearsal schedule
repeats basic roads;it does not create additional independent conditions.
Actual reset order and exposure counts are audited. The24known validation
conditions remain separated from TRAIN by the existing route-gap check.

Per seed,the queue runs one shared zero-PPO24-condition baseline,then H training
and its24-condition endpoint,then H0 training and its24-condition endpoint.
The zero-PPO policies are identical because the appended terms start at zero.
All seeds,all failures and the final fixed-budget checkpoints are retained.
There is no checkpoint or seed selection based on validation score,and no new
supervised optimization. This is development validation;the final120conditions
are still unopened and the original acceptance thresholds remain unchanged.

Each native job has a four-hour guard and no automatic retries. The25native
jobs are sequential,with a bounded total queue window and immutable source.
Admission requires4GiBfreeGPU,16GiBfreeRAM and16GiBfree archive disk. The queue
uses2110–2112,permits only one explicitly identified disjoint peer,and never
stops the unowned2100server. Native frame,engine delivery,saved learning state,
reset schedule,measured starts and every static NN decision are checked.
Dynamic driving is covered by measured actor/raw-frame audits;only its starting
NN decision is reconstructed by the existing inference auditor.

From the repository root,prepare once into a fresh archive directory:

```powershell
.venv/Scripts/python.exe scripts/run_independent_outcome_cohort.py prepare --cohort D:/ThesisNativeArchive_20261104/20261104_independent_outcome_TRAIN48 --archive D:/ThesisNativeArchive_20261104
```

Run the frozen copy with the same runtime. The registered launcher receives
the recorded Python/CARLA-client locations explicitly:

```powershell
.venv/Scripts/python.exe D:/ThesisNativeArchive_20261104/20261104_independent_outcome_TRAIN48/source/scripts/run_independent_outcome_cohort.py run --cohort D:/ThesisNativeArchive_20261104/20261104_independent_outcome_TRAIN48 --archive D:/ThesisNativeArchive_20261104
```

Inspect `execution/status.json`,the named batch's measured heartbeat and OS
process identity. A missing terminal session is not a reason to restart.
`execution/failure.json` stops the queue and preserves the failed job;it is not
a score-dependent retry instruction. Local registrations contain runtime paths
and require sanitization before public asset packaging.
