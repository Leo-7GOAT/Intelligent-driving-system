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

B1 passes the 6m empty-route parking sanity check (8/8 after approximately 1M physical steps), but the same model fails the 23m task (0/8). H/H0 feedback ablation reaches 8/10 versus 6/10 at200k, then8/10 each at300k on development conditions. Both frozen200k policies score2/6 in a real CARLA transfer diagnostic. These are single-seed diagnostics, not independent confirmation. Current CPU regression: 66 passed, 5 conditional skips.

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

## Bounded development and recovery

The current development batch is recorded in `runs/active_development.json`. Its `protocol.json` records a two-hour execution window, source hashes, job budgets and resource limits. It is a development diagnostic, not the final experiment protocol.

```powershell
$batch = (Get-Content runs/active_development.json -Raw | ConvertFrom-Json).batch
.\.venv\Scripts\python.exe scripts/development_watchdog.py --batch $batch
```

The supervisor refuses concurrent ownership, skips completed jobs, preserves checkpoints and refuses automatic retry of failed jobs. A STOP file is not cleared automatically. Inspect `supervisor.json`, `supervisor_final.json`, each `worker_status.json`, `latest.json`, and raw logs. Do not use old six-hour scripts to overwrite historical protocols.

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
