# Frozen-policy native Grid deployment

This entry point executes the batch from `scripts/prepare_shared_native_grid.py`.
It retains the H checkpoint/observation/action schema and explicitly labels the
spatial replacement B2. It does not train a new B2 policy.

Validate all frozen source,calibration,model/normalizer hashes,the saved and
runtime environment configuration,and unchanged complete known24 task list:

```powershell
.venv/Scripts/python.exe scripts/run_shared_native_grid.py --batch D:/ARCHIVE/GRID_BATCH --validate-only
```

The actual prepared batch passes this command with501 source files. The new
driver's15 related CPU tests pass;that count overlaps earlier Grid tests.
An attempted launch while B1 remains incomplete is rejected before creating
an execution directory or connecting to CARLA. This was exercised against the
real prepared batch;no new Grid native controls have been produced.

After the existing H/H0 and matched-flat queues finish:

```powershell
.venv/Scripts/python.exe scripts/run_shared_native_grid.py --batch D:/ARCHIVE/GRID_BATCH --execution D:/ARCHIVE/GRID_BATCH/execution --after-flat D:/ARCHIVE/FLAT_COHORT
```

The driver requires all20 distinct matched-flat phase receipts,normal lifecycle
closure and matching protocol hashes. It measures4GiB free GPU memory,16GiB
RAM and16GiB archive headroom before launch. Queue activation changes only the
bounded wall window and retains the original protocol/preparation receipt.
The native launcher retains ownership checks,watchdog and zero automatic retries.

All24 episodes remain in the output,including failures. Closure requires raw
CARLA frame audit,all measured-start NN actions,complete static NN/reference
reconstruction and strict applied-control delivery audit. Moving actors retain
full raw/control checks,but moving NN sequence reconstruction extends only to
the initial action. The report includes individual episodes and costs;it does
not turn repeated known conditions into new independent or unseen samples.

The ready batch uses the first independently trained seed11 H actor. It tests
frozen-policy planner deployment,not five-seed retrained B2 performance or a
formal planner-contribution claim. Grid uses its body-spline certificate and
no Hybrid steering primitives/memo. The common measured-steering retimer,road
refinement,point constraints,MPC and safety remain enabled;their native outcome
is still pending. Research acceptance remains unmet.
