# Executability-aware hierarchical driving research

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

Latest [native-road/terminal-pose evidence](reports/native_pose_20261022.md): v16 B1 completes3M at0/10; v13 B1 completes1M at0/10. Fresh heading-PBRS v17 is running (1/10at1M). Three actual CARLA fixed-reference actuator variants each complete4/4native routes, but severe braking/jerk remains; experimental variants stay opt-in. Native-road PPO scene integration and formal research acceptance remain incomplete.

B1 passes the 6m empty-route parking sanity check (8/8 after approximately 1M physical steps), but the same model fails the 23m task (0/8). H/H0 feedback ablation reaches8/10 versus6/10 at200k,8/10 each at300k, and8/10 versus9/10 at1M on development conditions. Strong rule B3 completes10/10 on the identical cases. Both frozen200k policies score2/6 in actual CARLA transfer; real CARLA adaptation for approximately100k world ticks each raises both to3/6. These single-seed diagnostics establish neither feedback benefit nor PPO superiority.

Development v12 tests shared candidate scoring and explicit geometry, with an equal-network H/H0 feedback ablation and no action masks or teacher. At approximately 1M physical steps, H completes 8/10 and H0 6/10 development cases. The separate 450-input mixed-task B1 completes 0/10 after 1M steps and is not a healthy, matched final comparator. Actual CARLA adaptation of the v12 100k models reaches 4/6 for H and 3/6 for H0 at approximately 100k world ticks each; H regresses from 5/6 at 25k. These failures remain in the record.

The v13 experiment gives H/H0/B1 the same full-length empty-task pretraining stage, counted inside their total interaction budget, before mixed-task training. It is developmental, not independent confirmation. B1 remains 0/10 at 300k and fails the full-length empty-task inference diagnostic with and without exploration. The [camera/curriculum report](reports/camera_staged_20261022.md) retains earlier milestone evidence. The published original development checkout passed 66 tests with 5 conditional skips.

Later endpoints and diagnosis: v13 H completes 1M at 8/10; the hold10 B1 frequency trial also completes 1M but reaches 0/10 with zero progress. Both isolated B1 trials (candidate-set encoder and signed-progress base reward) have now completed approximately 1M steps at 0/10. The progress-reward model moves forward but stops about 2.7m short in both empty cases. Its unchanged checkpoint is continuing to cumulative 2M/3M to test training span, retaining all prior failures and costs. This changes neither seed nor objective and is not independent confirmation. The reward differs from the old H objective and requires matched retraining before an architecture claim. See the [flat-PPO diagnosis](reports/flat_diagnosis_20261022.md).

Actual CARLA RGB videos now illustrate one PPO success and one timeout, each linked to its model, scene and sensor/world frames. See the [camera reproduction instructions](docs/CARLA_bridge.md#real-camera-development-evidence-2026-10-22). Videos are illustrative development evidence, not an aggregate success estimate.

A separate [actual steering/native-geometry audit](reports/steering_native_20261022.md) completed22actuator phases/3,520CARLAcontrol frames across normalized steering±1.0. The existing mapping stays unchanged; sampled error within the policy domain is at most0.01346rad. Native-map surveying supplies334possibly overlapping curve candidates, without claiming native-road driving validation. The B1 progress-reward continuation has reached2,000,379steps and still0/10; it continues toward the predeclared3M endpoint with all failures retained.

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

Latest local full regression: **102 passed, 5 conditional skips**. Earlier release checkout test counts describe their own frozen versions.

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
