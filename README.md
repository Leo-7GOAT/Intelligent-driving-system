# Executability-aware hierarchical driving research

[中文说明](docs/README.zh-CN.md) · [Acceptance ledger](docs/research_acceptance.md) · [Method](docs/design.md) · [Releases](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases)

A known-map, low-speed research system combining **PPO high-level decisions → local pose/speed goals → Hybrid A* → timed trajectories → MPC → actual CARLA state and reward feedback**. Flat PPO, grid-planner and strong-rule comparisons share the declared physical constraints and safety capabilities.

**Development prerelease; research acceptance remains unmet.** Real CARLA training, control-frame audits and public model reproduction have been executed. The current evidence does not establish hierarchical superiority, a stable added-PPO benefit, or road-safety certification.


Latest: the [five-seed broader B1 prerelease](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261102-broader-b1) is public. Its 69 assets were downloaded without authentication and reconstructed: 151,660 controls, 120 initial actions and five byte-identical figures. [Public receipt](reports/native_B1_broader_public_download_verification_20261102.json).

Recent baseline repairs remain unsuccessful. Additional TRAIN geometry scored 3/24; a learned route-error residual scored 1/24 on the same known development inventory. Actual 5,120-control PPO courses then scored 5/24 at the retained learning rate and 4/24 in a matched rate-only contrast. All raw controls, actual Adam counts and failures are retained. [Low-rate result](reports/native_B1_route_feedback_PPO5120_complete_validation_20261102.json), [rate contrast](reports/native_B1_route_feedback_rate_contrast_complete_20261102.json). These are single-seed development results, not a healthy baseline or independent final confirmation.

The route-error branch was rejected. A matched TRAIN rehearsal course from the earlier CARLA-trained canonical seed22 policy retained all model/Adam/RMS/RNG initialization bytes and used 10,240 new actual controls with zero new supervision. It passed the basic roads **4/4** and scored **15/24** on known development conditions (curve/static/walker: 6/3/6 successes out of eight each; seven bounds and two collisions). The same-budget course without familiar-road rehearsal failed one basic road. This is one-seed development evidence, not five-seed health or final confirmation. [Training, schedule and raw-control evidence](reports/native_B1_matched_rehearsal_complete_20261102.json).

The canonical H/H0 adaptation completed with **24/24 known development successes each and zero unsafe cases**. Retained new learning controls are 10,467/10,453, with 330 retained Adam updates each. H0 recovery lost at least 1,643 consumed controls and 50 Adam steps; its unknown tail remains disclosed. Equal scores do not establish a feedback benefit. [Complete evidence](reports/native_H_H0_diverse_recovered_complete_20261103.json). Earlier interrupted diagnostics and all failures remain retained.

For the old twelve static TRAIN conditions, H scored **3/12** (8 deadlines,1 bounds failure) and H0 now has a complete recovered **4/12** (7 deadlines,1 bounds failure). H0 combines only disjoint authenticated cases with identical frozen model and normalization bytes;all torn data and costs remain retained. This one-seed comparison does not establish a feedback benefit. The completed zero-PPO, BC-initialized teacher transfer scored **5/12** (6 deadlines,1 bounds failure;7,339 controls). This is a neural imitation policy, not the direct B3 rule controller. [Authenticated three-policy comparison](reports/native_static_H_H0_teacher_complete_20261103.json). [Complete recovered pair](reports/native_static_H_H0_recovered_complete_20261103.json). The shared bounds case had identical measured states and commands despite stable goal actions;offline lane queries and read-only engine surface rays reveal a raster/lane/surface disagreement. Original failures and scores are unchanged. [Geometry evidence and limits](reports/native_static_shared_road_disagreement_20261103.json). The reference-context H/H0 course has completed **21,024 actual training controls**, with 330 Adam updates per learner, unchanged RMS and full recorded-frame/control-delivery verification. Its first endpoint attempt stopped before a policy control because an evaluation container omitted the 31 context inputs; the trained weights are preserved. The container repair passes actual saved-model inference on a recorded CARLA begin and 1,310 CPU tests. Corrected endpoint evaluation started on the actual CARLA server at 17:31 local time; the H policy has executed more than 60 decisions with the registered 4,736-input architecture. The new H static endpoint has now finished **3/12** (8 deadlines,1 bounds failure;8,304 measured controls verified), so its success count did not improve. H0 and known-validation results,whole-phase engine verification and complete NN replay remain pending. [Completed H endpoint](reports/native_reference_context_H_static_complete_20261103.json). A prospectively registered H-only scale100 PPO follow-up is waiting for that entire audited queue;it has not trained yet and retains the completed H0 control. [Queue state and gates](reports/native_reference_scale100_followup_queued_20261103.json). [Learning and failure costs](reports/native_reference_context_learning_complete_20261103.json), [schema validation](reports/native_reference_context_evaluation_schema_validation_20261103.json), [live-state handoff](docs/handoff.md).

The last complete local CPU regression passed **1,310 tests**, with one skip and four CARLA tests deselected; subsequent targeted checks cover recovery and raw-frame auditing. [Source-specific hosted CPU verification](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41590243732) passed. [Training-only raw verification](reports/native_training_only_raw_verification_20261103.json) accounts for 20,480 existing B1 controls and their reset/reward totals; it adds no learning or driving.

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
| Broader B1 course, canonical seed22 | 20,480 new actual controls; 630 Adam steps; 7/24 known VAL successes (5/8 curve, 0/8 static, 2/8 walker); 11 bounds, five collisions, one deadline | [Completed audit](reports/native_B1_broader_seed22_20261102.json). All 31,480 controls verified; RMS unchanged. Development screen fails. Different initialization from the older 14/24 model; no causal comparison is made. |
| Same broader B1 recipe, seed11 | 20,480 new actual controls; 640 Adam steps; 11/24 successes (4/8,4/8,3/8); five deadlines,five bounds,three collisions | [Completed audit](reports/native_B1_broader_seed11_20261102.json). Development screen also fails; allfive broader courses complete. |
| Same broader B1 recipe, seed33 | 20,480 new actual controls;619 Adam steps;11/24successes (6/8,0/8,5/8);seven bounds,six collisions | [Completed audit](reports/native_B1_broader_seed33_20261102.json). 29,621controls verified;development screen fails; allfive complete. |

| Same broader B1 recipe, recovered seed44 | 15/24 successes (6/8 curve,4/8 static,5/8 walker);four collisions,three bounds,two deadlines | [Complete segment audit](reports/native_B1_broader_seed44_recovered_20261102.json). Saved11,520 plus8,960 resumed controls retain20,480learning controls/640Adamsteps. The course fails the development screen;interruption retained. |

| Same broader B1 recipe, seed55 | 10/24successes (5/8curve,2/8static,3/8walker);seven collisions,seven bounds | [Completed audit](reports/native_B1_broader_seed55_20261102.json). 20,480newcontrols/619Adamsteps;28,841rawcontrols verified. Screen fails. |

All five registered B1 basic courses are complete: seeds 11/22/33/44/55 score **3/4, 4/4, 4/4, 4/4, 3/4**. The two failures are bounds violations, so only **three of five basic gates pass**. The courses retain 51,200 new actual learning controls and 1,599 Adam steps. These are four repeated known TRAIN roads, not 20 unique conditions or a healthy full-task baseline. [All five outcomes and costs](reports/native_B1_shared_parking_complete_basic_cohort_20261102.json).

Seed55 includes an explicit interruption recovery: 3,840 saved controls plus 6,400 resumed controls. Reported consumed training controls are at least 51,858, of which 51,837 have readable bridge acknowledgements; the remaining 21 trace-only controls and an unknown tail are disclosed. The broader seed22 course is now complete at 7/24 on known validation conditions. It is additional development learning, not final confirmation. [Recovery proof](reports/native_B1_seed55_recovery_registration_20261102.json), [broader registration](reports/native_B1_broader_shared_parking_registration_20261101.json).

A measured-input diagnosis confirms that feedback reaches H: hiding both feedback groups changes 11/446 sampled greedy actions; H0 remains exactly invariant in all 448 samples. The physical mask stays fixed. This local sensitivity supplies no new driving-performance claim. [Diagnostic and reproduction](docs/native_feedback_use.md).

Before seeing seed22's validation results, the other four seeds were registered for the same additional 20,480-control broader course. Allfive broader courses are complete:seeds11/22/33/44/55score11/7/11/15/10outof24. [Prospective four-course registration and limits](docs/broader_B1_cohort.md). The complete five-seed broader result is54/120successes (45%) and58collision-or-bounds cases;allfive development screens fail. These reuse24known conditions,not120new final conditions. [Complete cohort and costs](reports/native_B1_broader_complete_cohort_20261102.json). The seed22 audit also reconstructs field-for-field from an independent checkout using local data. [Verification receipt](reports/native_B1_broader_seed22_independent_verification_20261102.json).

Seed44 hit disk protection after 11,520 saved controls; it completed the remaining 8,960 from the exact model/Adam/RMS/RNG checkpoint. Completed old artifacts were hash-verified and relocated to D while retaining their original paths. The interruption and replacement downstream queues are explicit. The earlier four-seed partial44/96record is retained separately. [Historical partial cohort](reports/native_B1_broader_partial_cohort_20261102.json). [Recovery evidence](reports/native_B1_broader_seed44_interruption_20261102.json).

The parking aid requests physical braking only inside the unchanged goal pose gates while overspeed; it changes neither steering nor vehicle state. It releases before success can be accepted. H/H0/B1/B2/B3 can share the same version. Every override is recorded. [Implementation and experiments](docs/shared_parking.md).

The original [TRAIN-only teacher collection](docs/broader_TRAIN_teacher.md) and subsequent recovery/diverse-road collections are complete. Their extra supervision and unsuccessful policy probes are recorded separately from PPO learning. The newest 36-case teacher collection achieved 31/36; that teacher result is not flat-policy performance. [Collection evidence](reports/native_B3_diverse_TRAIN_teacher_collection_20261102.json).

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

- At least five independently trained policies and a flat baseline with demonstrated basic learning, sound implementation and adequate training, with fair H/H0/B1/B2/B3, frequency, safety and fixed-reference comparisons.
- At least 120 **new, unique, paired final conditions**, including at least 30 per core class and unseen layouts or vehicle parameters. Existing 12- and 24-condition development sets are not final confirmation.
- Lightweight safe completion ≥85% overall / ≥75% per core; CARLA ≥80% / ≥70%, with collision-or-bounds episodes ≤5%.
- H minus healthy B1 ≥10 percentage points with a paired 95% lower bound above zero, or a prospectively frozen efficiency alternative; separately demonstrated PPO value against strong B3 and interpretable collaboration benefit.
- Reproducible models, raw results, all failures/costs, honest conclusions and public delivery. Current releases fulfill engineering reproduction milestones, not the full research acceptance.

[Acceptance ledger](docs/research_acceptance.md) · [Failure analysis](docs/failure_analysis.md) · [Evidence-bounded Chinese/English project descriptions](docs/resume_project.md) · [Archived development snapshots](docs/README.development-history.md).

Decision-cycle deadline overruns remain; synchronous simulator execution is not real-time validation. This work does not establish high-speed, production or real-vehicle capability. [MIT license](LICENSE) and [third-party notices](docs/THIRD_PARTY_NOTICES.md). Environment directories, simulator binaries, credentials and bulk raw videos are excluded from Git history.
