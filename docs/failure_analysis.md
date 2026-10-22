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

## Recorded-state spatial audit

At all eight failed H/H0 frozen-CARLA terminal states, the shared offline planner can find paths to3–12distinct available candidate targets (speed duplicates excluded). The policy repeatedly chose different, unplannable targets. This narrows the issue to policy selection/learning or the quality of the feedback representation rather than universal absence of spatially feasible actions. It does not prove those alternate trajectories would track safely, and dynamic safety was deliberately outside the spatial audit. Source snapshots and all16initial/final state results are retained under `spatial_candidate_audit/` in the transfer batch.

## Completed follow-up evidence

The v9 fixed-distance and uniform-distance B1 jobs both finish1,000,960 physical steps with the same poor cross-distance results; they are no longer running. The v8 lightweight continuation ends H8/10 versus H0 9/10. Real CARLA adaptation of the original200k pair ends3/6 each at approximately100k world ticks, still with static/curve deadlines. Strong B3 on the identical v8 lightweight development cases achieves10/10, including both unseen layouts. These are counterexamples to a current feedback/PPO-superiority claim, not excluded outliers.

v12 now tests a shared candidate scorer with explicit geometry and unchanged reward/low level. H0 gets the same geometry and network but zero candidate witnesses/previous outcomes. All actions remain available; no teacher or auxiliary labels are introduced. Initial25k results (H0/10, H0 4/10) are retained. The new architecture has not earned an improvement claim. A separate mixed-task B1 is still training with its own frozen450-input version; it is not silently reused as a605-input formal comparator.
