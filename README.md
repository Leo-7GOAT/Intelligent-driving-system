# Executability-aware hierarchical driving research

The [structured native-road experiment](reports/road_candidate_development_20261023.md) has completed H0/H/B1 endpoints at **100,160 / 100,160 / 100,050 retained training controls; all remain 0/4** on four training-road probes. H0/H have four deadlines each; B1 has three deadlines and one bounds failure. All **250,019** segment training/evaluation controls match raw CARLA frames. B1's interrupted costs raise its charged training lower bound to 109,595, with an unknown ancestor tail retained. H failures include waiting short of the goal and near-goal heading/planning failure. Exact rollout replay identifies critic-dominated global gradient clipping; a six-arm, same-parent comparison of global versus independent actor/critic clipping is now running in real CARLA. This is one-seed development evidence, not established improvement. Independent-checkout full regression at source `2040d3f`: **183 passed, five conditional skips**. The [normalization continuation](reports/recovery_aggregation_20261023.md) is complete at 302,080 cumulative controls per branch: updating statistics **4/10**, frozen statistics **3/10**. The earlier frozen-statistics advantage did not persist. The [complete native model/data asset](docs/model_card.md) is published and downloaded into a separate checkout: all 895 files, three raw audits, twelve initial actions and the trajectory figure reproduce. Publication is not research acceptance.

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

Fixed-reference actual CARLA experiments have exposed severe stopping/braking spikes. Experimental feedforward reduces speed error and jerk on four references but remains disabled for training because its internal state is not represented in the policy observation. Four searched Hybrid A* paths executed with unchanged legacy actuation produce2successes/2sampled-footprint departures. Offline grid feasibility is not continuous road-membership truth or a real-time claim. All failures remain in the [published diagnostic evidence](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261022).

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
