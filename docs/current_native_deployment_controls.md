# Current native deployment controls

This development study freezes the current completed seed11 H and B1 models.
Each model uses the same24 known validation conditions in four new probes:
10Hz and4Hz command updates, and the common execution brake OFF and ON.
These eight probes repeat24 unique conditions; they are not192 independent
conditions or additional trained policies. All outcomes remain included.

The frequency probes keep CARLA physics at20Hz. H keeps its2Hz PPO decisions,
20Hz MPC prediction grid and original horizon, while recomputing the MPC command
at10/4Hz and holding it between computations. B1 issues its direct policy action
at10/4Hz. This is deployment sensitivity, not a retuned low-frequency controller
or retrained policy comparison. Every physical frame and safety/event check is
retained. Actual solver recomputations, held frames, held solver failures and
nonreference braking frames are counted separately. Full-cycle timing remains
distinct from recomputation-only timing and is not real-time certification.

An offline check found that the old hold wrapper shadowed the native reset's
`reference_centered_effort` flag instead of forwarding it to the actual MPC.
At a0.3rad equilibrium, it returned0.27rad while the centered solver returned
approximately0.300001rad. The wrapper now forwards the setting, invalidates any
held command, and records the actual effort profile even on held/failed frames.
The current H/H0/B1 queues do not use this wrapper and are unaffected. The new
trace audit rejects a cadence probe that loses the reference-centered cost.
Warm versus fresh OSQP solutions are compared within their numerical tolerance;
the production solver tolerances and physical limits are unchanged.

The safety probes add the same inexpensive current-state execution brake to
both methods. Both OFF and ON retain H's original trajectory safety check;
the stock shared-shield flag must not silently disable that check. Common-brake,
parking-assist and total intervention counts remain separate. No safety
training occurs here. A deployment intervention result cannot stand in for
retraining with a shared wrapper or claim road-safety certification.

Preparation copies each parent's physical source and bridge, changing only the
registered cadence or additive-brake overlay. Model, normalization and saved
RNG bytes are unchanged. All original closed raw/control/model/source hashes
must verify. The paired models must share seed11, calibration and the complete
validation list. Parent low-level training budgets are not reinterpreted.

```powershell
.venv/Scripts/python.exe scripts/prepare_current_native_deployment_controls.py --output D:/ARCHIVE/deployment_controls --h-parent D:/ARCHIVE/H_seed11_known24 --flat-parent D:/ARCHIVE/B1_seed11_known24 --pilot D:/ARCHIVE/conditional_pilot --owner-pid CURRENT_VERIFIED_PILOT_PID
.venv/Scripts/python.exe scripts/run_current_native_deployment_controls.py --output D:/ARCHIVE/deployment_controls --validate-only
.venv/Scripts/python.exe D:/ARCHIVE/deployment_controls/source/scripts/run_current_native_deployment_controls.py --output D:/ARCHIVE/deployment_controls
```

The foreground coordinator waits on the authenticated pilot PID/birth/executable
handle and requires all four pilot native lifecycles to close before using2120.
It may coexist with2110 but never starts a third server. Wait is bounded at24h;
each native probe has a4h limit and zero automatic retries. Admission requires
4GiB free GPU,16GiB RAM and16GiB archive space before each job. Queued activation
preserves original protocols and changes only the wall window.

After every probe, all native controls and engine delivery must verify. H also
reconstructs every static/moving policy decision; B1 verifies every measured
start. Saved models and normalization bytes must still match the parent, with
zero optimizer updates. This remains one selected development seed. Formal
contribution claims need adequate independent runs and a locked final protocol.
