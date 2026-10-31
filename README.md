# Executability-aware hierarchical driving research

Latest (31 October): H trained on the explicitly versioned v4 geometry
completes **12/12 known development conditions, zero unsafe episodes**, after
20,563 real CARLA training controls and 650 Adam steps. Its matched H0 endpoint
is **11/12, zero unsafe**, after 20,562 training controls/650 Adam steps.
All 50,348 training/evaluation controls and 24 measured-start actions verify;
the owned server has been cleaned up. The one-case gap is not significant
(exact discordance p=1.0).
[Complete paired evidence](reports/native_v4_trained_seed11_feedback_pair_20261031.json).

The second trained H seed also completes12/12,zero unsafe;all24,858controls
verify. Its H0sibling is recovering after a recorded GPU-telemetry guard timeout,
with all lost interactions charged. This is not a second closed/matched pair.
[Recovery evidence](reports/native_seed22_GPU_timeout_recovery_20261031.json).

The separate **frozen-weight** v4 migration scores H 9/12 versus H0 10/12,
with no unsafe episodes. This reverses the earlier v3 gap and establishes no
feedback benefit. B1 trained on v4 scores **7/12, zero unsafe, five deadlines**;
its [public raw/model capsule](reports/B1_v4_PPO_public_verification_20261031.json)
independently reproduces all 27,737 controls and the figure. Matched actor-only
fits now score **replay8/12 with3bounds versus approach10/12 with2bounds**.
All10,193controls/24start actions verify and both public capsules independently
reproduce. The added training-state labels repair two conditions but safety
remains unqualified. Both initializers have matched20,480-control PPO queued;
these new PPO phases have not started.
[Complete fit comparison](reports/native_B1_v4_approach_pair_complete_20261031.json).
[Diagnosis](reports/native_B1_v4_stop_diagnosis_20261031.md).
These are development conditions; five trained seeds, a healthy full-task B1,
new final conditions and research acceptance remain unmet.

[Runtime and feedback audit](reports/native_v4_runtime_and_feedback_20261031.md):
H's environment decision P95 is348ms but its maximum is4.22s; full-runner
latency is not logged. B1's shared feedback computation alone has220ms P95
against a50ms period. Fixed-weight feedback hiding changes10/439Hactions and
zero H0actions; this demonstrates sensitivity, not a performance benefit.

Earlier execution snapshot:

31 October: [completed native diagnostics](reports/completed_native_diagnostics_20261031.md): fixed-reward-unit B1 finishes **2/12**,with6unsafe episodes after20,480real training controls/208Adam steps. It is rejected as healthy. The actual fixed-reference CARLA pair completes **MPC4/4,Pursuit/PI4/4**,with3811RPC-bound controls;Pursuit has lower tracking error and jerk here. Neither result establishes research acceptance. Both raw-data capsules are public. Shared-mask Hfrozen reevaluation finishes9/12,all core families3/4,zero unsafe;the fresh B1rollout1280study is now actually training. Subsequent bounded training remains queued.

A [numerically checked geometry optimization](reports/exact_corner_compute_20261031.md) reduces one recorded7200-node search median6.440→3.668s while preserving all549151collision decisions and the same failed search result. Full local regression passes668tests;active/queued frozen experiments retain their original code.
The [completed twelve-condition development comparison](reports/native_twelve_case_completed_20261027.json) uses four known roads across curve/parking, static obstacles and crossing walkers. Rule-only B3 succeeds **12/12**. With separately charged teacher supervision and **zero PPO updates**, H and H0 each succeed **11/12**; direct B1 succeeds **4/12**, with six collision/bounds episodes. B1 remains unqualified, and these results do not establish PPO value, hierarchy advantage or feedback benefit.

The [real CARLA PPO pair](reports/native_twelve_case_PPO_complete_20261027.json) is complete: **H11/12, H0 10/12**, with20,541/20,623training controls,650Adam steps each and **51,429raw controls audited**. H retains its supervised score; H0 loses one condition. The single-seed gap does not establish feedback benefit or PPO value over the12/12strong rule. The [ordered-candidate B1](reports/native_twelve_case_ordered_B1_endpoint_20261027.json) completes **3/12**; a [different fit split](reports/native_twelve_case_phase4_B1_endpoint_20261027.json) reaches **6/12**. Both failed the original full-task initialization gate. The corrective actor-only endpoint completed7/12with four bounds failures and remains unsuitable. A separately disclosed basic-motion learning entry allowed real B1 PPO: [20,480training controls and640Adam steps](reports/native_B1_basic_motion_PPO_endpoint_20261027.json), but the frozen result degraded to4/12. It is not a healthy baseline; a [matched KL-stop diagnostic](reports/native_B1_KL_PPO_endpoint_20261028.json) also completes4/12. The [102,400-control endpoint](reports/native_B1_long_PPO_endpoint_20261028.json) is now complete and degrades to **2/12**, with seven bounds failures and one collision. It is rejected as a healthy baseline. A [fresh critic-rate diagnostic](reports/native_B1_critic_rate_endpoint_20261028.json) also ends4/12with seven unsafe episodes. A separate [fixed learner-reward-unit profile](reports/native_fixed_reward_units_20261028.md) finishes2/12with six unsafe episodes and is rejected;see the31October report.

The [completed smaller-exploration H/H0 PPO comparison](reports/native_shared_small_exploration_complete_20261027.json) scores **2/4 each**, with **46,493** training/evaluation controls audited against CARLA. There is no aggregate feedback gap. The [smaller-exploration B1 PPO](reports/native_B1_small_exploration_20261027.json) also remains 2/4 on its known static task; its [public model/raw-data capsule](docs/model_card.md#public-smaller-exploration-b1) has been downloaded and independently reproduced, including all 23,025 raw controls.

[Full independent reconstruction of the earlier mixed teacher dataset](reports/native_mixed_measured_full_reconstruction_20261027.json) now reproduces all 3,617 frames, five arrays and the complete NPZ bytes exactly. This adds zero driving, fitting or PPO updates. The earlier 600-second failed reconstruction remains preserved; the completed attempt used an upfront 900-second budget and took 809.75 seconds. [Dynamic hierarchical view verification](reports/native_hierarchical_dynamic_views_20261027.json).

The [body-spline B2 deployment diagnostic](reports/native_B2_body_spline_endpoint_20261028.json) completes **7/12**, versus legacy postprocessing4/12; it has no new training and does not establish equal planner compute. The [new common-mask H/H0 initialization diagnostic](reports/native_shared_mask_seed11_BC_endpoint_20261028.json) completes **11/12 each**, with zero PPO updates. Both methods now share the physical endpoint mask; only learned actor/critic feedback differs. [All15initializers across five fixed seeds](reports/native_fixed_seed_initializations_20261028.json) exist, but this is not five trained CARLA policies.

A [new geometry-separated development inventory](reports/native_training_validation_partition_20261028.md) registers36training/24validation conditions and runtime audits keep their resets separate. It has no measured validation result yet. All existing twelve-case scores remain development results. Five independent trained seeds, 120 new paired final conditions, a healthy fully trained B1, complete ablations, PPO value versus strong rules and unseen transfer remain open. The new static obstacle distance is 13m; this does not erase the earlier 10m failures. Hosted source345b1ac passes **598CPU tests**,15skips/4CARLA exclusions;32targeted checks also pass in the independent locked environment. CPU verification is separate from CARLA results.

[Execution and queue commands](docs/native_execution.md) · [Current handoff](docs/handoff.md) · [Chinese README](docs/README.zh-CN.md)

The [real-PPO model/raw-data capsule](docs/model_card.md#public-twelve-condition-ppo)
is actually uploaded and publicly downloaded: an independent checkout reproduces
all51,429controls,24policy start actions and its byte-identical packaged figure.
[Three rule/initializer capsules](docs/model_card.md#public-twelve-condition-development-evidence)
separately reproduce18,591controls,36start actions and two repository figures.

Earlier development snapshots follow.

Latest [supervised-PPO retention test](reports/native_supervised_retention_20261027.md): the3/4supervised direct model drops to0/4after20,480real PPO controls. All23,427frames verify and observation statistics stay fixed. A matched-initialization shared reward-profile diagnostic is running; no PPO improvement is claimed.

Latest [measured direct-control diagnostic](reports/native_measured_bc_20261027.md):
extra teacher supervision yields a continuous direct network at **3/4 known
full static roads**, with zero PPO updates;1,976frozen frames verify. Actual
PPO continuation is now running and has a separate result. The candidate-arc
PPO version was rejected (H/H0/B1all0/4); all69,514frames are retained.
These single-seed development results do not establish formal acceptance.

Latest [measured body geometry and shared-reward learning](reports/native_static_body_learning_20261027.md):
conservative measured-body strips improve rule B3 static performance from2/4to3/4
on known roads. The endpoint-filter PPO follow-up remains0/4after audited frozen
recovery. New guarded-reward H completes20,761training controls and2/4known-road
endpoints; recovered H0 is0/4after20,480lineage controls. B1 is still training.
These are incomplete single-seed development comparisons; formal RL value and
a healthy full-task B1 remain unestablished. The earlier negative-result
[public model/data capsules](docs/model_card.md#public-hazard-evidence-2026-10-27)
were actually downloaded and independently reproduced, including raw frames.

Latest [native-hazard and cadence development](reports/native_hazards_cadence_20261026.md):
fresh10Hz direct PPO uses20,765 real training controls but remains0/4 on known6m
goals. Native parked-car/walker tasks and frame evidence are implemented; B3
static failures and a fourth-road walker spawn failure are retained. Reference
reuse gives B3 static2/4; the corrected walker task gives B3 4/4. Fresh static
H/H0/B1 each remain0/4; an isolated necessary-endpoint filter is being tested.
Research acceptance remains open.

Latest [matched terminal-margin development](reports/native_terminal_margin_20261026.md):
own-success replay gives3/4on known4.5m goals using all positive episodes versus
**4/4** using47qualified episodes, with matched initialization and unchanged task
gates. Frozen6m transfer and three real-PPO6m adaptation variants remain3/4;
speed exploration improves but the last failed curve exits bounds. This is not
a healthy full-task baseline, independent confirmation or formal acceptance.

Latest [skill-retention development](reports/native_retention_20261026.md): real
PPO plus separately counted own-success replay reaches **4/4 on three-metre
known-road goals**, but frozen six-metre transfer and4.5m continuation remain
**3/4**. These single-seed curriculum results do not establish a healthy full-task
baseline, ordinary-PPO performance or research acceptance. H local filtering and
progress-reward trials do not improve the earlier unshared-head endpoint.

Latest [matched full-road decision-head development pair](reports/native_actor_head_20261026.md): shared **1/4**, unshared **3/4**, one seed/four known roads. All13,259controls and500frozen decisions verify. Search failure and missed decision periods remain; this is not formal superiority or a healthy-B1/realtime result.

[Current status and next research steps](docs/current_status_20261026.md) · [Chinese continuation prompt](docs/continuation_prompt.zh-CN.md). Complete native model/data asset is published and public-download reproduction passes; formal research acceptance remains open.

[Hosted CPU regression passed](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40310891811):318passed/3platform-or-opt-in skips/4CARLA deselections on Ubuntu24.04 at source5a300e9; both local and independent fixed Windows environments pass320/1skip/4deselections. The [3m training and6m frozen-transfer assets](docs/model_card.md) are publicly downloaded/reproduced:794files,5,894rawcontrols,8initial decisions and2byte-identical figures. CPU/offline verification is separate from new CARLA driving or research acceptance. See [CI scope and reproduction](docs/cpu_ci.md).

An optional [exact road SAT batching backend](reports/road_collision_batch_20261025.md) preserves collision flags and complete search paths in the recorded development probe, while reducing search cost. Scalar remains the default; the completed frozen CARLA PPO batch used it. Offline timings are not full-cycle real-time evidence.

The [initial/trained actor-weight contrast](docs/native_weight_contrast.md) now has a frozen-source CARLA evaluator and complete frame/inference auditor. Local CPU regression passes295/1skip/4CARLA deselections; actual H0 initialization and save/reload match its training log. The actual weight contrast **started but is incomplete**, with no whole-batch completion/cleanup proof; its initial arm scores0/4 and the trained arm is partial. Both weight arms will share the trained frozen normalizer, so this is a conditional weight contribution test, not a fresh-normalizer untrained-system score.

The [complete fresh native-route development batch](reports/native_route_complete_20261025.md) scores **H0 3/4, H 2/4, B1 0/4** on four known training roads, after20,702/20,665/20,621actual CARLA training controls and130optimizer steps each. All **67,289training/evaluation controls** are frame-audited and all534deterministic decisions reconstruct from saved models. H has policy-selected waiting failures; B1's four probes all exit bounds and it is not a healthy final baseline. All training failures, including H's one and B1's39bounds episodes, remain. The original worker/guard complete and the owned empty server is stopped. These single-seed development results do not establish feedback benefit, PPO contribution, unseen generalization or hierarchy superiority.

The [goal-heading search guide](reports/native_dubins_guide_20261025.md) separately completes **4/4 known roads** under rule-selected B3 execution, with1,319controls raw-audited. The earlier batching implementation's CPU regression passed267/five conditional skips, followed by29targeted read-only-tool/audit tests in both environments; current CPU counts are linked above. Research acceptance remains incomplete; see [handoff](docs/handoff.md) for live identities and evidence limits.

As of 2026-10-24 18:38 local, the [complete 32/256-decision native rollout comparison](reports/native_rollout_horizon_20261024.md) scores **0/4 in all six arms**. H0 mean progress is **1.84/1.15m**, H **14.99/14.94m**, and B1 **2.20/6.64m**. All **639,009** segment training/evaluation controls are audited; all outcomes, rollout overshoot and inherited costs are retained. The subsequent [actual native rule diagnostic](reports/native_rule_protocol_20261024.md) completes **3/4**, using the same H low level and measured starts, with **1,977** audited controls and no policy updates. Its fourth case waits after reaching a state with no moving candidate reference. This supports investigating learned selection and terminal candidate coverage, not a hierarchy or PPO-value claim. Both batches completed with owned-server cleanup; these are reused training roads and correlated seed-11 PPO forks, not independent confirmation. The earlier [clipping study](reports/native_clipping_pair_20261024.md) also scores0/4 throughout; independent actor/critic clipping was not adopted.

Full CPU regression at source `ac59741`: **203 passed, five conditional skips**; 17 recovery/rollout tests also pass in the independent checkout. Three offline action-availability tests at `530d168` pass in both checkouts. The H0 pair, 226 physical episode returns and 16 recorded-state reference queries reproduce byte-for-byte independently. Those queries find available local references despite all eight final H0 decisions selecting failed plans; they do not establish successful execution. The earlier [100k native model/data asset](docs/model_card.md) is published and independently downloaded/reproduced: all 895 files, three raw audits, twelve initial actions and the trajectory figure match. Its H0/H/B1 retained training controls are **100,160 / 100,160 / 100,050**, all at **0/4**. Publication is not research acceptance.

The following paragraphs are historical development snapshots; their then-live states do not override the current run identities in [handoff](docs/handoff.md).

Latest [recovery-state supervision diagnostic](reports/recovery_aggregation_20261023.md): ordinary PPO after supervised aggregation regresses from **4/10 to 0/10**. Reducing initial exploration alone also ends at **0/10**, with six collisions. Freezing observation statistics during otherwise matched reduced-noise PPO instead ends at **4/10 and parking 8/8** after 102,400 controls; all static, narrow and unseen cases still fail. Every declared endpoint, raw control and supervision cost is retained. This supports further retention testing, not a healthy complete-task baseline, a pure-RL comparison or hierarchy superiority. Local and independent-checkout regression at source `eee6c8a`: **139 passed, five conditional skips**.

The preceding current-state label contrast scores **1/10 versus its 2/10 actor-only control** after matching initialization and19,400updates. All32,538relabel controls and10,397new evaluation controls are verified; changing labels alone did not repair competence. The earlier teacher-mixture study's supervised endpoints were4/10 and1/10, followed by4/10 after102,400PPO controls. Every endpoint and supervision cost remains recorded.

Latest [recovery and stability evidence](reports/recovery_stability_20261023.md): the unbounded, tanh and softsign B1 branches all finish at 6/10 development successes; static and unseen tasks still fail. Native CARLA H recovery adds 70,080 controls and B1 completes 100,270 controls; both score 0/4 on training-road probes. All 176,167 training/evaluation controls match raw CARLA frames, with lost recovery costs explicitly charged. Independent-checkout live v3 verification adds 1,592 verified controls. Full CPU regression: **133 passed, five conditional skips** in both worktrees. Research acceptance remains incomplete.

The separate [demonstration-assisted diagnostic](reports/demonstration_diagnostic_20261023.md) records 32 successful B3/direct-replay pairs, but its learned BC-only policy and subsequent 25,600/102,400-control PPO endpoints all score 0/10. Teacher success is not learned-policy success; additional supervision and the changed control frequency are explicitly disclosed.

Latest [native geometry and PPO evidence](reports/indexed_native_geometry_20261022.md): explicit v3 lane-union geometry repairs recorded seam/junction false rejections. Fixed-plan Hybrid A* and B2 each complete4/4development roads, but real paired native PPO at~10k controls remains0/4for both methods. All26,470training/evaluation frames match raw CARLA records. Lightweight B1 learns23m parking at intermediate checkpoints and later regresses; the final5M full-task result remains0/10. CPU regression:117passed/5conditional skips. Research acceptance remains incomplete.

A low-speed research prototype: **PPO → local pose/speed → forward Hybrid A* → time-parameterized trajectory → MPC → physical execution and feedback**.

**Research acceptance is not yet met.** Real CARLA 0.9.15 training and validation have run, but current evidence does not establish that the learned high-level policy improves on a strong rule-based high level. This is not a road-safety or production-driving claim.

See the [acceptance ledger](docs/research_acceptance.md), [results](docs/research_results.md), [failure analysis](docs/failure_analysis.md), [method](docs/design.md), and [Chinese guide](docs/README.zh-CN.md).

## Measured historical results

| Experiment | H | Flat PPO B1 | Limitation |
|---|---:|---:|---|
| Lightweight, completed 1M paired seeds 11/33/44 | 30/40 each | 0/40 each | Only 3 complete H seeds; B3 also 30/40; B1 learning inadequate |
| CARLA, approximately 50k real world-step adaptation, seed 11 | 9/30 | 0/30 | H successes only dynamic |
| Same CARLA budget, seed 33 | 10/30 | 0/30 | Static and curve/parking remain unsuccessful |

These evaluated scenarios are now **historical**, not unseen confirmation data for subsequent development. Failed runs and incomplete seeds are retained. New development has not earned an architecture-superiority claim.

## Current development (2026-10-22)

| Completed development version | Result | Evidence boundary |
|---|---|---|
| v8 H/H0,1M physical steps |8/10 vs9/10; strong B3 10/10|No learned-high-level or feedback advantage|
| v12 candidate H/H0,1M |8/10 vs6/10; CARLA100k adaptation4/6 vs3/6|One seed; H regresses from5/6at25k|
| v13–v17 flat learning diagnoses |Full-task B1 endpoints0/10|Frequency, architecture, training span and reward alternatives retain their failures|
| v18 distance stages,5M |0/10; earlier23m parking capability lost|Normalization swaps do not rescue the5M policy|
| v19 rehearsal,5M endpoint |Full task0/10; intermediate23m parking later regresses|Completed; intermediate policies are diagnostic, not selected final scores|

Full source/model provenance, traces and limitations are in the [retention/feasibility report](reports/rehearsal_feasibility_20261022.md), [new B2 report](reports/b2_rehearsal_20261022.md), [flat diagnosis](reports/flat_diagnosis_20261022.md), and [native-road learning report](reports/native_road_learning_20261022.md). The shared measured64x64 road grid reaches policies, planning, risk and termination. Actual native-road H/B1 PPO updates and exact reloads have run; those short runs did not complete a task. All historical evaluated conditions remain development data.

Historical fixed-reference actual CARLA experiments exposed severe stopping/braking spikes. Feedforward initially remained disabled for training until its internal state was observable; the current version exposes all four actuator states and trains fresh policies with that calibrated adapter, as recorded in the [actuator-state report](reports/native_actuator_state_20261025.md). Earlier searched paths with legacy actuation produced2successes/2sampled-footprint departures. Offline grid feasibility is not continuous road-membership truth or a real-time claim. Those failures remain in the [published diagnostic evidence](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261022).

Actual CARLA RGB videos illustrate one PPO success and one timeout, each linked to its model, scene and sensor/world frames. See the [camera reproduction instructions](docs/CARLA_bridge.md#real-camera-development-evidence-2026-10-22). Videos are illustrative development evidence, not an aggregate success estimate. Live process identities must be rechecked against the OS; historical snapshots in the [handoff](docs/handoff.md) are not permanent status.

Rebuild completed development milestones, checking frozen source and checkpoint hashes:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_paired_development.py --batch runs/20261022T050532Z_hierarchy_v8_development
.\.venv\Scripts\python.exe scripts/summarize_development.py --batch runs/20261022T042237Z_development_v6
```

## Local environment and tests

Python 3.11; install only into a project virtual environment. CARLA uses a separate existing Python 3.7 client matching server 0.9.15.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

CPU tests exercise physical dynamics, planning, QP control, PPO updates/checkpoints, statistical accounting and failure recovery. Live CARLA tests require explicit opt-in and an exclusively owned server; skipped tests are not evidence of live validation.

Historical local and independent-checkout regression at source `6d351d5`: **137 passed, 5 conditional skips**; this count applies to that source only. The completed pure-learner aggregation reproduces a byte-identical summary and figure in the independent pinned environment; both models reload with repeatable predictions and frozen statistics. The audit covers28,586collection controls and21,644evaluation controls. Earlier label-contrast and five-model recovery checks remain documented in their source-specific receipts. These newer diagnostic artifacts are local; they have not been uploaded as release assets.

## Bounded development and recovery

The candidate experiment is recorded in `runs/candidate_development.json`; the separate flat baseline uses `runs/matched_flat_development.json`. Each `protocol.json` records a two-hour execution window, source hashes, job budgets and resource limits. These are development diagnostics, not the final experiment protocol. Pointer files are local outputs. To create a new candidate experiment from the documented task on a fresh checkout, run the initializer once; it needs no CARLA installation or historical local run. Omit initialization when resuming an existing batch.

```powershell
.\.venv\Scripts\python.exe scripts/initialize_candidate_development.py
$batch = (Get-Content runs/candidate_development.json -Raw | ConvertFrom-Json).batch
.\.venv\Scripts\python.exe scripts/development_watchdog.py --batch $batch
```

The supervisor refuses concurrent ownership, skips completed jobs, preserves checkpoints and refuses automatic retry of failed jobs. A STOP file is not cleared automatically. Inspect `supervisor.json`, `supervisor_final.json`, each `worker_status.json`, `latest.json`, and raw logs. Do not use old six-hour scripts to overwrite historical protocols.

For a successfully completed local batch, extend the same policies in a new bounded window with `scripts/continue_development.py --parent <batch> --milestones <larger-cumulative-budgets> --label <unique-label> --reason <evidence-based-reason>`. It requires released parent ownership, successful exits, matching counters/configuration and verified frozen source; it copies the resumable checkpoint without resetting optimizer, RNG, normalization or interaction costs. The new pointer is `runs/<unique-label>.json`; start its watchdog with the command above. Preparing a continuation alone does not start training. The currently launched reward continuation uses `runs/progress_v16_continuation.json`.

For existing locally saved historical data, rebuild the historical report with:

```powershell
.\.venv\Scripts\python.exe scripts/publication_report.py --batch runs/20261016T015226Z_research_batch_3377580f
```

## Reproduce the development asset

The development pre-release asset is `development_v8_seed11.zip` (SHA256 `ad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c`). It contains both200k H/H0 checkpoints, frozen source/normalization/configuration, optimizer and training-episode summaries, and all20 development evaluation traces. Read the [model card](docs/model_card.md) before interpreting the results. This is a bounded development asset, not the full final experiment archive.

Download the asset from the [development pre-release](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261022):

```powershell
Expand-Archive -LiteralPath runs/development_v8_seed11.zip -DestinationPath runs
.\.venv\Scripts\python.exe scripts/reproduce_development_asset.py --asset runs/development_v8_seed11 --output runs/reproduced --full
```

The command verifies hashes, reloads both models with frozen normalization, repeats all20 cases and reconstructs the endpoint figure. Use a new output directory on each replay. The local complete replay matched all recorded outcomes; fresh-checkout verification and publication status are tracked separately in the [release record](docs/release_status.md). No CI success is implied.

Environment directories, CARLA binaries and bulk raw videos do not enter Git history. The [MIT license](LICENSE), original attribution and [third-party notices](docs/THIRD_PARTY_NOTICES.md) are preserved. Research acceptance, five-seed confirmation, complete CARLA adaptation and final release remain pending.
