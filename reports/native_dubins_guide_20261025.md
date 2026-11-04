# Goal-heading search guide — 2026-10-25

This is a low-level engineering hypothesis. PPO contribution, healthy full-task
B1, five independent training seeds and fresh final confirmation remain open.

## Completed expanded-budget control

`20261025T074939Z_native_rule`, frozen1d3c9a9, completes3/4 with1,919applied
controls,123reset ticks and2,042worldticks. Walltime2,036.99s. Normal guard exit0
and owned-server cleanup are recorded. Raw audit SHA256:
`3bf224632244c0db501a2ca7951c093ea024bad8e4e17068d07f772b922ceba7`.
Roads0/1/2 succeed in341/361/317controls. Road3 times out after900controls,
with21reference and69wait decisions, progress12.7341m and finalgoal distance
7.9898m. Recorded plan/MPC failures and interventions remain zero; failed rule
candidate searches are separately charged and must not disappear behind wait.
Full decision P95 is2.78/11.97/3.72/28.39s; road3max30.59s. This is not realtime.
Some offline CPU work overlaps the later trial; walltimes are descriptive,
whereas search acceptance remains a deterministic node budget.

The4/4coverage gate fails; no new PPO model/pointer is created. Larger budget
alone does not fix the fourth road. The earlier30-minute interrupted run is
retained separately and is not retroactively counted as a completed trial.

## New offline evidence and implementation

The obstacle-free Dubins guide evaluates six forward bounded-curvature path
families using the vehicle's measured minimum turning radius. It only changes
Hybrid A* queue priority. Collision checking, primitives, cost, node limit,
analytic connector, goal tolerances and time parameterization remain unchanged.
Goal acceptance still uses the original geometric tolerance test. Because the
guide targets an exact pose while acceptance allows tolerances and search is
weighted/discretized, no admissibility, optimality or completeness is claimed.

Formulas are adapted from [OMPL's primary implementation](https://github.com/ompl/ompl/blob/main/src/ompl/base/spaces/src/DubinsStateSpace.cpp),
Mark Moll/Rice University,BSD-3-Clause. Full notices are retained both in the
module and `third_party/ompl/LICENSE`. This established method is not claimed as
an original algorithm. No OMPL runtime or third-party path sampler is added.

At the first interrupted-run wait state, original1,800-node search times out;
the new guide succeeds in147expansions at the same cap and produces a checked
81-point stopping reference. At the second, whole-run-audited wait state
(frame7649,decision21 of074939Z), original7,200-cap search reports
`open_set_exhausted` after2,637expansions. The guide succeeds in170expansions
with cost8.3025 and a checked68-point,5.635s reference ending at speed0.
Thus even open-set exhaustion in this discretized search is not proof of
physical infeasibility. Queue order changes which continuous representatives
survive discrete-key pruning. Neither offline path has been executed here.

`benchmark_dubins_guide.py` rebinds measured frames and freezes analysis sources.
Artifacts are `dubins_guide_probe_v2.json` under070545Z and
`dubins_guide_probe.json` under074939Z, with adjacent source archives. A62kB
development-state fixture preserves the second case for CPU regression; it is
not an unseen evaluation condition. Tests integrate every returned family with
independent bicycle arcs, check all six families, known lengths, rigid/scaling
invariance, original goal gates and the measured-state body-safe reference.

Default `heuristic_mode=euclidean` preserves historical config serialization and
model signatures. Opt-in `dubins_v1` changes the signature and requires fresh
H/H0/B1 policies. Staging carries both planner budget and heuristic to all three;
B1 remains direct actuation. H0 retains the existing witness/outcome ablation.
Full regression260passed/5conditional skips/2existing warnings in185.86s;
after adding the measured-state fixture,11targeted tests pass in9.71s.

## Prospective real-execution contrast

Before launch fix `heuristic_mode=dubins_v1` with the same7,200-node cap as the
completed expanded-budget run. Keep the exact FFcalibration,4705shared input,
continuation rule score/three-target-search limit,native-score speed policy,
four training roads,45s deadline,reward,vehicle limits,trajectory and MPC.
Only search-order guidance changes in the low-level behavior. Compute costs and
all failures remain recorded. No rule becomes a PPO action override.

Use a new immutable batch with60-minute wallguard,12GiBdisk/4GiBRAM minimum,
85°C GPU stop,no-progress300s,zero retries and owned cleanup. Only new
completed/raw-audited4/4coverage can start fresh H0/H/B1seed11,20,480controls
each initially. No existing model/normalizer is reused and no task gate changes.

```powershell
python scripts/benchmark_dubins_guide.py --batch runs/20261025T074939Z_native_rule --output runs/guide_probe_rebuilt.json
python -m pytest tests/test_dubins_guide.py tests/test_native_route_preparation.py
python scripts/evaluate_native_rule.py --prepare runs/20261024T075110Z_native_rollout_pair --source-method H_rollout32 --catalog route_frame_terminal --continuation-feedback --calibration runs/20261022T122427Z_native_curve_control/calibration/calibration.json --planner-nodes 7200 --planner-heuristic dubins_v1 --wall-seconds 3600
```

The optional read-only `probe_carla37.py --ego-snapshot` waits for the existing
owner's next frame. A fresh synchronous client can initially have a stale empty
actor cache; that is not evidence that the worker stopped. This observer never
ticks, spawns or controls actors and is not the evaluation-outcome source.

## Actual coverage and new learning

Frozenb495e71 B3 batch `20261025T084117Z_native_rule` completes **4/4**:
341/350/317/311controls,1,319total+123reset=1,442worldticks,521.6754swall.
All controls and four terminal outcomes pass raw audit, SHA256
`f34ec951297f51aef9aef061085e74273927aa96d776cb89672adc3b3d1535fb`.
Owned guard exits0/reasonnull and empty-server cleanup completes. Road3 parks
in15.55simseconds with zero recorded plan/MPCfailures/interventions. This is
known-road development coverage, not unseen confirmation or learned success.

`planning_comparison.json` verifies matching task/config/calibration and11control
sources; every measured-start delta is zero. It keeps allfour conditions.
Road3 waits fall69to1,while road1 candidate-search expansions rise6,314to13,634.
New full-decision P95 is4.26/17.45/4.28/6.27s byroad;P99 is
4.40/24.52/4.83/6.60s,max4.44/26.99/5.05/6.68s. These remain far above0.5s;
no hard realtime improvement or population effect is claimed. Candidate-search
node counts exclude any extra environment search and are not total planner work.

Independent checkout b495e71 passes22targetedtests in18.15s. The prior7200 raw
audit is byte-identical; the second-state benchmark reproduces state,signatures,
paths and nodecounts exactly. Source text matches after universal-newline
decoding; rawsource hashes can differ with GitCRLF normalization. The standalone
first-wait PNG also reproduces byte-identically. Guided raw/whole-comparison
reproduction follows the analysis-script publication.

The original4/4gate now passes. Fresh CARLA PPO batch
`20261025T085433Z_native_route_learning` actually starts under a six-hour guard:
H0/H/B1seed11,20,480controls each,32decisions/rollout,2epochs, identical frozen
FFadapter/7200nodes/guide and shared4705-input observations. No inherited
weights,normalizers or teacher actions. At17:06 H0 reports2,799physical controls;
its last completed optimization has16optimizer steps andactorL2change.0128093.
H/B1 are queued and have no new result yet. Training streams remain live and
have not undergone final raw audit; these are not converged or five-seed results.
See handoff for actual identities and22:54:33 resource deadline.

```powershell
python scripts/audit_native_rule.py --batch runs/20261025T084117Z_native_rule --output runs/guided_raw_rebuilt.json
python scripts/compare_native_rule_planning.py --baseline runs/20261025T074939Z_native_rule --candidate runs/20261025T084117Z_native_rule --output runs/planning_comparison_rebuilt.json
```

At17:16 the independent checkout at1fd311d has reproduced both the guided4/4raw
audit and the fullfour-case planning comparison byte-identically. Comparison
SHA256 `5da9a5e1cd7bc83bd5b66b0682a5f9523462ea2cca9d39d4fabf28fff15361a8`;
receipt is `runs/guided4_reproduction.json` in the independent checkout. No new
physical controls or optimizer updates were used for this reconstruction.
Live H0 training now reports4,233controls and13optimization phases; its endpoint
and H/B1 outcomes remain pending. The22:54:33 guard remains healthy.
