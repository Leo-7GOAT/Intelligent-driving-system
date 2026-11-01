# Executability-aware hierarchical driving research

[中文说明](docs/README.zh-CN.md) · [Acceptance ledger](docs/research_acceptance.md) · [Method](docs/design.md) · [Releases](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases)

A known-map, low-speed research system combining **PPO high-level decisions → local pose/speed goals → Hybrid A* → timed trajectories → MPC → actual CARLA state and reward feedback**. Flat PPO, grid-planner and strong-rule comparisons share the declared physical constraints and safety capabilities.

**Development prerelease; research acceptance remains unmet.** Real CARLA training, control-frame audits and public model reproduction have been executed. The current evidence does not establish hierarchical superiority, a stable added-PPO benefit, or road-safety certification.

## Verified development evidence

The following studies have different training histories and declared interfaces. They are separate results, not a single budget-matched comparison.

| Study | Measured result | Evidence and limitation |
|---|---|---|
| Five independent H/H0 pairs, 36 TRAIN / 24 known validation conditions | H 120/120; H0 119/120; no collision/bounds | [Paired report](reports/native_matched_broader_feedback_cohort_20261101.json). Mean +0.83 percentage points, 95% interval [−1.48, +3.15]. This is 24 unique conditions, not 120 new conditions. |
| Added PPO after teacher initialization | Trained H 120/120; teacher-only 117/120; strong B3 24/24 | [Attribution report](reports/native_broader_PPO_teacher_rule_attribution_20261101.json). Paired mean +2.50 points, 95% interval [−2.13, +7.13]; stable PPO benefit is unproven. |
| Shared terminal parking aid, frozen B1/H/B3 transfers | Each 4/4 on the same four known TRAIN roads | [Engineering audit](reports/native_shared_parking_engineering_20261101.json). B1/H/B3 overrides: 6/1,409, 8/1,287, 8/1,286 controls. Deployment transfer, not new learning. |
| Single B1 PPO course with the aid fixed throughout training | 10,240 new actual controls; 318 Adam steps; basic endpoint 4/4 | [Learning audit](reports/native_B1_shared_parking_PPO_20261101.json). Initial model already passed 4/4; this proves retention, not an added-PPO gain. |
| Same frozen B1, twelve original known conditions | 12/12; 4,848 verified controls | [Three-class probe](reports/native_B1_shared_parking_full_probe_20261101.json). No new learning or final conditions. |
| Same B1, 24 different registered development roads | 14/24: curve/parking 5/8, static 3/8, walkers 6/8; eight bounds and two collisions | [Complete record](reports/native_B1_shared_parking_validation_complete_20261101.json). 10,210 controls verified. A spawn failure and seven-case continuation use explicitly disclosed mixed initialization versions; descriptive development evidence only. |

A separate, uniformly registered five-seed B1 basic course is in progress. Completed seeds 11/22/33/44 score **3/4, 4/4, 4/4, 4/4**, respectively, after **10,240 new actual PPO controls and 320 Adam steps each**. Seed 11 has one bounds failure and fails the basic safety gate. [Seed 11](reports/native_B1_shared_parking_cohort_seed11_20261101.json), [seed 22](reports/native_B1_shared_parking_cohort_seed22_20261101.json), [seed 33](reports/native_B1_shared_parking_cohort_seed33_20261101.json), [seed 44](reports/native_B1_shared_parking_cohort_seed44_20261101.json). Pending results are not counted as successes. The single-seed broader 36 TRAIN / 24 validation course is registered and queued, not a completed five-seed baseline. [Registration](reports/native_B1_broader_shared_parking_registration_20261101.json).

The parking aid requests physical braking only inside the unchanged goal pose gates while overspeed; it changes neither steering nor vehicle state. It releases before success can be accepted. H/H0/B1/B2/B3 can share the same version. Every override is recorded. [Implementation and experiments](docs/shared_parking.md).

## Install and run CPU checks

Use Python 3.11 and an isolated environment. Existing installations need not be recreated.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q -m "not carla"
```

[GitHub CPU workflow](.github/workflows/cpu-regression.yml) exercises dynamics, planning, MPC, actual PPO/checkpoint updates and recovery. Live simulator tests are excluded or opt-in; a CPU pass does not certify CARLA performance. Source-specific counts and exclusions are recorded in verification receipts.

CARLA uses the existing **0.9.15 server and its separate compatible Python 3.7 client**. Do not install CARLA binaries in this repository or attach another tick owner to an active world. [Bridge, calibration and live verification](docs/CARLA_bridge.md).

## Reproduce a published trained model and raw results

The [shared-wrapper PPO prerelease](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101-parking-ppo) includes models, normalization, frozen source, configuration and raw controls. Its 572-file PPO capsule was publicly downloaded and independently rebuilt: all 11,639 controls, four measured-start actions, the figure and saved Adam/RMS proof match. [Executed receipt](reports/native_B1_shared_parking_PPO_public_verification_20261101.json).

From the repository root, use fresh output paths:

```powershell
$recipe = Get-Content protocols/public_native_B1_shared_parking_PPO_20261101.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe scripts/download_native_asset.py --manifest-url $recipe.manifest_url --sha256 $recipe.archive_sha256 --output runs/my_parking_PPO_download
.\.venv\Scripts\python.exe scripts/reproduce_native_asset.py --archive (Join-Path runs/my_parking_PPO_download $recipe.archive_name) --sha256 $recipe.archive_sha256 --output runs/my_parking_PPO_rebuild
.\.venv\Scripts\python.exe scripts/analyze_native_shared_parking_ppo.py --batch runs/my_parking_PPO_rebuild/asset/20261101T112758733226Z_native_B1_shared_parking_PPO --output runs/my_parking_PPO_learning_audit.json
```

This restores the policy and reconstructs existing evidence; it performs zero new driving or learning. [Additional capsules, whole-action audits and a real replay of a publicly downloaded model](docs/shared_parking.md). The public five-seed H/H0 release independently reconstructs 287,811 recorded controls and its paired statistics. [Verification](reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json).

## Training, recovery and remaining acceptance work

Each native batch freezes its source hashes, model, normalization, physical-interaction budget and watchdog limits before execution. Completed jobs are preserved, failed gates remain failures, and new low-level versions are distinguished from frozen-policy transfers. [Exact local handoff and queue identities](docs/handoff.md) and [current experiment details](docs/shared_parking.md) describe the already launched courses; status files require verification against live process identity before any recovery. Do not launch a duplicate of an active queue.

The original requirements remain:

- At least five independently trained policies and a healthy full-task flat baseline, with fair H/H0/B1/B2/B3, frequency, safety and fixed-reference comparisons.
- At least 120 **new, unique, paired final conditions**, including at least 30 per core class and unseen layouts or vehicle parameters. Existing 12- and 24-condition development sets are not final confirmation.
- Lightweight safe completion ≥85% overall / ≥75% per core; CARLA ≥80% / ≥70%, with collision-or-bounds episodes ≤5%.
- H minus healthy B1 ≥10 percentage points with a paired 95% lower bound above zero, or a prospectively frozen efficiency alternative; separately demonstrated PPO value against strong B3 and interpretable collaboration benefit.
- Reproducible models, raw results, all failures/costs, honest conclusions and public delivery. Current releases fulfill engineering reproduction milestones, not the full research acceptance.

[Acceptance ledger](docs/research_acceptance.md) · [Failure analysis](docs/failure_analysis.md) · [Evidence-bounded Chinese/English project descriptions](docs/resume_project.md) · [Archived development snapshots](docs/README.development-history.md).

Decision-cycle deadline overruns remain; synchronous simulator execution is not real-time validation. This work does not establish high-speed, production or real-vehicle capability. [MIT license](LICENSE) and [third-party notices](docs/THIRD_PARTY_NOTICES.md). Environment directories, simulator binaries, credentials and bulk raw videos are excluded from Git history.
