# Failure analysis

This file records development findings, including negative results. Historical inspected final sets are no longer unseen confirmation sets.

| Observation | Evidence | Interpretation and action |
|---|---|---|
| Historical B1 gives0/40 while H and B3 both give30/40 | Oct16 archived paired evaluations | Cannot attribute the gap to learned high-level decision making; rebuild a healthy flat baseline and compare against strongB3 |
| B1 learns6m parking at1M but fails23m/45s | v6 model,8/8 short and0/8 full-route inference | Training-distance specialization; fresh fixed23m versus uniform6–23m development is running |
| Fresh fixed23m B1 passes23m at300k but fails6/12/18m | v9 two conditions per distance | Basic long-route motion/parking can be learned; distance generalization remains inadequate |
| H200k8/10 versusH0 6/10; both300k8/10 | v8 matched450-input development | One checkpoint difference is unstable; do not claim feedback contribution before sufficient independent runs |
| H300k fails unseen, H0 fails static | v8 raw per-family results | Same total completion can conceal different failure mechanisms; retain per-family reports |
| Frozen learned CARLA H/H0 both2/6 | v8 real-frame transfer, no infrastructure failures | Low collision counts coexist with four deadlines; no successful transfer or safety claim |
| Failed CARLA cases repeat candidate actions | `action_failure_audit.json` and raw traces | Search-budget exhaustion or endpoint body collision/bounds, rather than a sensor-disconnect explanation; matched real CARLA adaptation is running |
| Terminal target offsets could leave an unachievable final lateral correction | Geometric tests and B3 physical diagnostics | Added an explicit terminal return envelope; old experiments retained, new versions retrained |
| Legacy PID-only owner locks survive process death | Fault-injection and live reconnect tests | OS-held process leases now release on death; still no automatic recovery from a crash with live residual actors |

Calibration remains limited: steering was calibrated over a partial normalized range, native curved-road tracking is not yet established, and large measured CARLA braking deceleration spikes remain visible. Current curve cases are obstacle-defined curved routes on a straight roadbed. Controller and policy improvements must be evaluated with progress, time, intervention, failure and comfort together.

Current development is not a new final-test protocol. A future confirmation must freeze method/configuration, sufficient independently trained seeds, model-selection rules, budget, case generator, endpoints and multiplicity handling before inspecting its results.
