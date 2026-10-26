# Native hazards, physical-time cadence and reference lifecycle

These are seed11 development experiments on the same four known roads. Original
research acceptance remains unmet. Rule-only B3 results contain zero PPO updates
and cannot establish learned high-level value. The [measured ledger](native_hazards_cadence_summary_20261026.json)
retains run IDs, costs, outcomes and artifact/raw-stream hashes.

## Fresh direct-PPO cadence experiment

Fresh categorical B1 executes at10Hz rather than2Hz, with the same6m task,
12s deadline, measured vehicle, task reward, speed/steering catalog and seed11.
All32 initial policy tensor fingerprints and postinitialization Torch RNG match
the reconstructed historical2Hz initializer. Both schedules use nominal320
physical controls per rollout and10 full-batch Adam steps per rollout. Gamma
and GAE have the same decay per physical second;10Hz uses160 decisions/batch
instead of32. Actual episode-ending partial holds remain charged.

The new policy performs20,765 actual CARLA training controls,10,400 decisions
and650 Adam updates in428.01 training seconds. Its101 training endings include
41 successes,35 deadlines and25 bounds violations. Successes occur only on roads0
and2 (22/19 episodes), with none on roads1/3. Frozen evaluation is **0/4**, all
deadlines,960 additional controls. No teacher, replay, inherited policy or action
override is used. This does not meet the healthy-B1 gate.

The completed historical2Hz endpoint also scores0/4 after20,510 lineage
controls/650 Adam steps. It includes one physical reset/resume; this is a
development comparison, not a clean causal frequency ablation. Two separately
preserved offline fits of the new policy's own successes are not deployed or
scored. They are not counted as successful endpoints or additional PPO training.

## Actual native-road hazards

The bridge now supports registered route-relative parked vehicles and crossing
walkers on native driving surfaces. CARLA owns every actor's motion. Ego still
uses VehicleControl through PPO/planner/MPC or the explicitly labelled B3;
there is no execution teleport, direct velocity setting or autopilot. The
bridge records actual actor identities and per-frame pedestrian positions and
velocities. Raw audits reject missing or altered native dynamic measurements.

| B3 development phase | Registered cases | Result/status |
|---|---:|---|
| Initial static parked vehicle, old search | 4 | 0/4;3,600 audited controls |
| Skip blocked endpoints before spending fallback searches | 4 | 0/4;3,600 audited controls |
| Original walker offset−3.5m | 4 | Three completed successes; fourth fails during actor spawn |
| Final-goal candidate pair, static | 4 | 0/4;3,600 audited controls |
| Uniform walker offset−2.5m | 4 | Running; no complete score yet |
| Final-goal pair plus reference reuse, static | 4 | Prepared; not yet scored |

The original walker batch is explicitly incomplete. Its three completed cases
have1,340 frame-verified controls and92 reset ticks; the fourth has no policy
controls. The prefix audit never converts the missing case into a success or
writes a fabricated completed summary. Its failure, guard exit1 and normal
owned-server cleanup remain available.

A spawn-only infrastructure preflight tests every road at each of four fixed
lateral offsets. Offset−3.5m fails on road3;−2.5m,+2.5m,+3.5m create actors on all
four roads. The next registered walker task uses−2.5m uniformly, chosen from
spawn feasibility before policy execution, not from successful driving. Every
preflight actor is destroyed. A first probe's immediate Actor.destroy() hit a
stale client cache; its failed output/log and exact owned-ID cleanup receipt
remain. The corrected probe uses server-side DestroyActor without issuing ticks.

## Candidate coverage and timed-reference retention

On the original measured static scene0, full-goal Hybrid A* finds a667-sample,
192-node swept-body-free path within the same7,200-node limit. Local greedy
choices can enter a state with no recovered path. Blocked endpoints also wasted
the strong comparator's three fallback attempts. Skipping these necessary
endpoint failures strengthens B3 but does not solve the physical task.

A new explicit `route_goal_candidate` replaces only actions25/26 (the furthest
centre speed pair) with the already visible final task pose at.75/1.5m/s. It
changes the checkpoint action/candidate signature, requires new models, and is
shared with direct-policy observations. The same31 choices remain available;
PPO selects its own action. On actual initial scenes0/2, this selects complete
paths in192/82 nodes. Scene1 still has no recovered path; scene3 needs two
searches/12,709 total nodes. Offline coverage is not actual task completion.

Physical execution with this candidate alone still fails on the first three
roads: scene0 drives1.48m then waits after later searches lose connectivity.
The separate `reference_reuse` experiment therefore retains an existing timed
reference only when the high level explicitly chooses the same pose and speed.
It preserves time and path arrays, checks current full-body freedom, remaining
reference geometry, tracking limits(.25m/.2rad and.75m timed-position error),
and current measured actors' constant-velocity predictions. Changed actions,
explicit wait, expiry, new obstacles or failed tracking invalidate reuse.
There is no state projection, automatic target replacement or hidden rule
decision inside PPO. The flag changes the controller signature and is shared
by H/H0/B3; any future learning comparison must retrain relevant methods.

A separate geometry diagnostic finds that an actual rotated parked car's
single axis-aligned box covers24.17m², while a conservative12-strip cover uses
12.30m². The same bounded full-goal search still exhausts7,200 nodes on scene3.
This prototype is not adopted or presented as a feasibility proof. The initial
read-only query had an uninitialized frame0 cache and is invalid; its preserved
replacement waits for a snapshot and reproduces the settled scene box exactly.

## Reproduction and limits

Current CPU regression: **347 passed,1 skipped,4 CARLA exclusions**,283.51s.
Synthetic tests exercise reference lifecycle, unchanged path/time, explicit
action selection, changing static/dynamic hazards, missing-case accounting and
cadence initialization; they are not simulator success evidence.

```powershell
python -m pytest -q -m "not carla"
python scripts/audit_native_rule.py --batch runs/20261026T133002Z_native_rule --allow-completed-prefix --output runs/prefix_recheck.json
python scripts/probe_native_candidate_coverage.py --batch runs/20261026T131154Z_native_rule --final-goal-candidate --output runs/candidate_recheck.json
python scripts/evaluate_native_rule.py --prepare runs/20261025T085433Z_native_route_learning --source-method H --wall-seconds 3600 --catalog route_frame_terminal --planner-nodes 7200 --planner-heuristic dubins_v1 --road-collision-backend batch_sat_v1 --hazard-kind static_vehicle --rpc-port 2200 --final-goal-candidate --reference-reuse
```

Preparation does not launch a server or worker. Use the existing owned-foreground
launcher, empty-server probe and identified-worker guard from the native-run
documentation. Never run two workers against one server or restart an active
batch because a tool observation times out. New artifacts in this report are
local development artifacts until a separate public-download receipt exists.
Five independent native strategies, new paired final conditions, healthy full
B1, static/dynamic PPO learning, formal hierarchy/RL contribution and realtime
acceptance remain open; this report lowers none of their original gates.
