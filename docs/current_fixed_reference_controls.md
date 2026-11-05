# Current fixed-reference native controller study

The earlier four-route controller result used the64-cell indexed road grid and
an older control acknowledgement interface. Current training uses the256-cell
indexed grid, `batch_sync_v1` delivery and reference-centered MPC effort cost.
Old results remain historical evidence; this study prepares new current inputs.

Four first registered basic TRAIN roads are retained without outcome selection.
From each first saved hazard-free measured initial state, the immutable current
Hybrid A* pipeline receives one explicit action26 global-goal query. This is a
controller-study query, not an action attributed to PPO. Its registered internal
speed-search behavior is retained and all search attempts are recorded. No
plant integration, new native query, vehicle command or policy prediction occurs
in reference generation.

The production steering-rate retimer caps the reference at1m/s and keeps the
rest-to-stop boundary conditions. Full-body freedom on the current measured
grid and endpoint pose/speed are checked. All four inputs must qualify; failures
remain in the manifest and an incomplete catalog cannot launch controls. The
actual four generated nominal durations are approximately25.34,26.14,23.74 and
22.14seconds, within the common45second controller window.

Both native jobs share every reference array, current vehicle/actuator
calibration, source file,20Hz physics grid, event/termination rules and starting
reference clock. MPC uses the current reference-centered cost and20step horizon.
Timed pursuit/PI retains its existing registered parameters and vehicle limits.
There is no PPO, planner replanning or parking-assist override during execution.
This isolates tracking of the shared reference; it does not measure PPO value
or end-to-end hierarchical task success.

```powershell
.venv/Scripts/python.exe scripts/export_current_fixed_references.py --parent D:/ARCHIVE/H_TRAIN48_closed --output D:/ARCHIVE/current_references
.venv/Scripts/python.exe scripts/prepare_current_fixed_reference_controls.py --output D:/ARCHIVE/current_fixed_controls --training D:/ARCHIVE/H_TRAIN48_closed --catalog D:/ARCHIVE/current_references --dependency D:/ARCHIVE/deployment_controls --owner-pid CURRENT_VERIFIED_DEPLOYMENT_DRIVER_PID
.venv/Scripts/python.exe scripts/run_current_fixed_reference_controls.py --output D:/ARCHIVE/current_fixed_controls --validate-only
.venv/Scripts/python.exe D:/ARCHIVE/current_fixed_controls/source/scripts/run_current_fixed_reference_controls.py --output D:/ARCHIVE/current_fixed_controls
```

The foreground queue waits on the authenticated deployment driver and all eight
closed deployment probes before using2120. It never starts a third server. Wait
is bounded at24h;each job has a4h guard and zero retries. Existing resource
admission and owned cleanup remain mandatory. Queued activation changes only
the wall window before the first lifecycle.

Every saved control frame must bind to a bridge request/response and measured
tracking state. The engine-delivery auditor now recognizes the fixed-controller
trace schema as evaluation, rather than requiring a learner method list.
Complete paired analysis additionally requires identical calibration, source
and all eight reference arrays, and keeps every outcome. QP failure counts are
not directly comparable to a pursuit controller that does not solve a QP.

These are four known TRAIN geometries, not final or independently trained test
policies. Prepared references, model calculations and queued input validation
are not native controller performance results. Current native outcomes remain
unavailable until both jobs actually execute and audits close.
