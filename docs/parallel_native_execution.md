# Isolated parallel native cohorts

The previous disjoint2100 CARLA peer is absent. On2026-11-05 the original
two-server resource allowance was reassigned to H/H0 on2110–2112 and matched
B1 on2120–2122. Initial admission measured8,702MiB free GPU memory,47.02GiB
free RAM and425.70GiB archive space. No third server is started. Existing
ownership,port-overlap,temperature,memory,disk and per-job wall guards remain.

All20 prepared B1recipes were checked against the serial versions:only RPCport
and span differ. Frozen physical source,models,normalizers,TRAIN/known manifests,
PPOsettings and original30,720-control learning budgets are unchanged. The
original registration and old read-only observer logs remain;that observer was
intentionally superseded before any B1native job began. The new execution log
directory is `execution_parallel_20261105`. Training still retains every seed
and outcome and makes no automatic retries.

The already-started invocation uses:

```powershell
.venv/Scripts/python.exe scripts/run_matched_flat_cohort.py run --output D:/ARCHIVE/FLAT_COHORT --resume-dependency D:/ARCHIVE/VERIFIED_H_OWNER.json --parallel-port 2120
```

The identity is verified by PID,birth time and executable through a read-only
process handle. Existing execution/artifact directories prevent duplicate
launch. This command is not an instruction to restart the running queue.

First actual progress:seed11 zero-PPO basics4/4,1,425 audited controls,all starts
and strict applied-control audit pass. This is the supervised initializer,not
learned PPO improvement. The first retained training-state proof verifies5,120
native controls/decisions and147 Adam advances in each of35 parameter states,
exact actual initialization/full Adam/Torch RNG and fixed observation RMS.
The policy changes;the full course and post-training health remain pending.

The original target-KL early-stop rule remains active,so actual optimizer counts
must be read from saved states rather than inferred from epochs. Basic evaluation
costs are separate from learning and inherited teacher costs. Concurrency changes
wall timing:these measurements do not prove isolated compute gains or real-time
operation. The Grid driver now checks all25H/H0and all20B1closures before it
may use2110. [Actual receipt and all20transport-only checks](../reports/native_parallel_flat_execution_20261105.json).

Source de79cc5 passes hosted CPU regression:1,443 passed,25 skipped,4 CARLA
exclusions,232.10s.10 local scheduling/dependency checks also pass. These are
infrastructure checks;research acceptance and the full matched study remain unmet.
