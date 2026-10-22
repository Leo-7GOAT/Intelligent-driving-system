# Active research handoff — 2026-10-22 13:23 local

Goal remains active and incomplete. Read the current user objective attachment before continuing. Existing public repository `Leo-7GOAT/Intelligent-driving-system`, branch `feature/physical-hierarchical-ppo`, base8142224. Development commitbbaf303 pushed to the existing feature branch and remote SHA verified. Do not modify other projects, reset/reinitialize Git, erase failures, or reuse inspected final conditions as unseen confirmation. No subagents authorized. Historical handoff retained in `reports/handoff_before_v6_20261022.md`.

## Live work (revalidate OS processes before acting)

- H/H0 continuation: `20261022T051509Z_hierarchy_v8_continuation`. Pointer `runs/hierarchy_continuation.json`. Both continue their completed200k checkpoints using EXACT parent frozen source, RNG, optimizer, environment state and normalization. Cumulative targets300k/1M, two-hour window from13:15. Actual workers8676/20476 were alive and optimizing at230k+, phase90. Check current identities, never assume these PIDs remain valid.
- B1 distance diagnosis: `20261022T051509Z_distance_v9_development`. Pointer `runs/distance_development.json`. Fresh seed11 fixed23m versus uniformly sampled6–23m, empty training, common45s deadline/beta40 PBRS/gSDE settings, targets100k/300k/1M. Actual workers1892/24656 were alive and optimizing at76k+, phase29/30. Four CPU workers total across both pairs, each Torch single-thread, no GPU training;12 logical CPUs, approximately43GiB RAM and72GiB disk available at launch. Independent supervisor leases, no automatic failure retry.
- CARLA transfer `20261022T051716Z_carla_v8_transfer` is COMPLETE, both learned H/H0 frozen200k policies score2/6 (one curve7302 and one dynamic7302), four deadlines each, zero observed collision/bounds, zero infrastructure failures. H4793 world ticks includingreset/4608control; H0 4861/4676. Normalization frozen. NO CARLA PPO updates in this transfer. Server launcher37724/shipping11520 was confirmed empty and stopped; receipt `infrastructure/server_stop.json`. Exec session11430 exited0, no remaining bridge.
- At13:21 H/H0 continuation both300k evaluations8/10; actual training313–325k and progressing. B1 full23 at300k gets2/8 pooled distance probes, uniform6–23 gets0/8. Inspect per-distance data before calling baseline health passed. Full1M budgets still running.

## Completed evidence

- v6 B1 two reward variants each1,000,960 physical steps: both8/8 on6m parking; PBRS40 model0/8 on23m/45s. Full-task baseline health NOT established. Retain PBRS40; explicit task-progress reward not needed by current evidence.
- v7 H/H0 gracefully superseded at12,790/10,225 physical steps. Partial logs/checkpoints retained, no performance conclusion.
- v8 original `20261022T050532Z_hierarchy_v8_development`: both supervisor exits0 and all milestones complete. At100k both6/10; at200k H8/10, H0 6/10. H completes empty/static/narrow/dynamic, H0 fails static; both fail unseen. H dynamic intervention~.145 vsH0~.025, so do not equate higher completion with uniformly better safety/efficiency. One seed, two near-identical conditions per family; no confidence interval or independent confirmation claim. Source/model verified48grouped rows, plot and report in `analysis/`.
- Actual CARLA B3 low-level diagnostic in v7 batch:3/6 before terminal envelope,4/6 after, with degraded static7302 retained. These are rules, not learned PPO. Server was confirmed empty and stopped before the new transfer.
- Regression:66 passed5 skipped in `reports/v9_transfer_regression_20261022.xml`, including calibration-preserves-policy-config regression. Raw XML ignored because local personal paths. Old failed tests retained.

## Implemented this turn

- Optional training-only `training_route_range`, deterministic seeded distance sampling; validation/test remain full-length unless explicit diagnostic override. Mutually exclusive with fixed-route/curriculum. Applies equally to architectures if used in later fair comparisons.
- Generic development evaluator handles B1 continuous actions and explicit route cases; closes episode envs; per-job PPO options. Running v8 continuation uses old frozen source, unaffected.
- `initialize_distance_development.py`, `initialize_development_continuation.py`, and `summarize_paired_development.py`. Continuation is a new batch with copied final checkpoint and inherited-step provenance, not hidden warm start or an extra independent seed.
- Fixed CARLA evaluator/fine-tuner dropping new policy observation/action/planner settings. `calibrated_config(...,source_config=...)` now preserves all source fields except calibrated vehicle; regression covers H/B1/B2 and shape450. Old frozen historical experiments unchanged.
- `carla_feedback_transfer.py` prepares/provenance-checks and evaluates learnedH/H0, labels calibrated low-level frozen policy, no rule replacement.

## Next executable actions

1. Transfer completed and server stopped. Inspect `action_failure_audit.json` for H/H0: failed cases repeatedly select the same candidates, producing search_budget or start_or_goal_body_collision_or_bounds. This is a policy/candidate/domain-transfer failure, not evidence of collision safety or successful CARLA learning. Both policies need actual adaptation after full lightweight milestones; do not erase the frozen-transfer negatives.
2. Inspect both training supervisor files plus actual OS PIDs, current optimization logs and milestone results. Continue their bounded windows; no resetting completed jobs.
3. Rebuild either new pair with `python scripts/summarize_paired_development.py --batch <batch>` after completed evaluation markers appear. Parent H/H0 milestones remain in original batch; continuation counts include parent training.
4. Update acceptance/README/handoff with final CARLA transfer outcomes and latest learning. Resolve unseen failure and healthy B1 before final protocol; don't prematurely freeze/publicize superiority.
5. Remaining full goal: five independent policies and sufficient formal comparisons, healthy matchedB1, PPO-vsB3 interaction value, CARLA true adaptation and120unique final conditions, native-road/actuator validation, real success/failure videos, clean-checkout reproduction, release hygiene, truthful bilingual docs/CV, real GitHub commit/push/tag. Research acceptance remains NOT met.

Recovery command for a stopped nonfailed bounded pair: `python scripts/development_watchdog.py --batch <exactbatch>`. Existing live lease prevents duplicates; STOP/expired window/failures require diagnosis, never blindly remove them. A new continuation window must retain source and cumulative counters explicitly.

Latest verified OS process/heartbeat snapshot13:23: H378,749 physical steps (149optimization phases); H0 359,981 (141); B1 full23 491,293 (191); B1 uniform500,061 (195). All four workers live, both supervisors without STOP, GPU35C. Do not treat these numbers/PIDs as permanently current. Prior goal turn was a verified live-process status inspection; this turn made code/test/actual training and CARLA evidence progress. Detailed development report: `reports/development_20261022.md`.


## Subsequent continuation (2026-10-22, publication and real adaptation)

Previous turn is classified as progress. This turn also makes concrete code/test/training/publication progress. B1 distance jobs both completed1,000,960 physical steps: fixed23m2/2 on23m but0/6 at shorter probe lengths; uniform6–23m0/8. Keep failures. H/H0 lightweight continuation is still running toward1M; inspect OS PIDs and current markers.

CARLA adaptation batch `20261022T052635Z_carla_v10_adaptation`, pointer `runs/carla_adaptation_development.json`: source-frozen paired H/H0 from the SAME200k models used in the transfer diagnostic. Interleaved cumulative25k/100k actual world-step budgets, reset cost included; seeded training conditions0..9; six existing development evaluation conditions. Both25k milestones complete at26,598/26,558 ticks and2/6 each, no improvement yet. H continued beyond69k with real optimizer updates; H0 waits its turn afterH100k. Actual coordinator46440, supervisor52864, bridge34208 at last verification; recheck identities. Server launcher10328 owned by this batch. Frozen bridge uses NAVSTACK_CARLA_PYTHON to reference the existing isolated runtime, while its Python source stays frozen. Worker finally attempts identity-checked empty-server shutdown; inspect cleanup receipt rather than assuming success. Do not launch another CARLA owner while this one is live.

Development asset `runs/prerelease_20261022/development_v8_seed11.zip`, SHA256ad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c.152files, two models, full20development evaluations, frozen source, optimizer/training-episode summaries. Verified all20replays both locally and in a fresh clone/newvenv. Fresh clone is `runs/prerelease_20261022/clean_checkout`; independent pip installation/pipcheck/66tests5skips succeeded. PureEOL issue in originaldialog_parser.py index is normalized in follow-up commit, with no semanticdiff. Original12snapshotfiles retain exactSHA256 through .gitattributes exceptions. No secrets/private user paths in publication candidates or asset; generic scanner path literals reviewed as expected code.

Git push needed per-command `-c http.proxy=http://127.0.0.1:7890`, matching enabled Windows proxy; directGit timed out, ghAPI worked. No global config changed. Authentication remainsLeo-7GOAT. Pre-release intendedtagv0.2.0-dev.20261022; inspect realGitHub status before claiming it exists. Research acceptance remains incomplete even after publication.


## Latest authoritative update (2026-10-22 13:51 local)

Publication is real: GitHub development pre-releasev0.2.0-dev.20261022 exists at9bcaa79d34a6985a3d12764d3bf01383c7f06d83,3220328-byte model/data ZIP uploaded, downloadedhash matchesad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c. Fresh tag clone `runs/prerelease_20261022/release_checkout` is clean;66tests5skips and20replays allpass using the freshly installed pinned environment from the first clone. NoCIclaim. Firstclone retainedwithEOLdirtyfile; newtagcloneverifiedcorrect. Pre-release doesNOTsatisfyresearchgates.

Live batches:
- H/H0 continuation (`runs/hierarchy_continuation.json`) at~897k/916k respectively, still training to1M. ActualPIDs8676/20476; revalidate.
- CARLA adaptation (`runs/carla_adaptation_development.json`): H finished100,890world ticks,38optimization phases,3/6dev (bothstaticfail;curve7301fail;dynamicbothsuccess;curve7302success). H0~79k,30phases,stilltraining; coordinator46440, latestbridge26504. Inspect actualprocesses and finalcleanupreceipt; neverstartanotherCARLAowner whilealive.
- B1 mixed feedback baseline (`runs/matched_flat_development.json`): batch20261022T054701Z_matched_flat_v11_development, actualworker14136, supervisor35164; >23kphysical steps,9realoptimization phases. SameHtask/reward/vehicle450obs/candidatefeedback; directcontrols, noMPCactionreplacement. Onefromscratchseed11; physical1Mtarget/twohourwindow. Geometryfeatures atflatfrequencyareadditionalcompute andmaylimitbudget; reportactualsavedsteps,don'thidecensoring.
- B1 distance pair terminal at1,000,960each; fixed23m2/2full-distance,0/6shorter; uniform0/8. NotfairtomixedHuntilthenewmixedbaselinefinishes.

New offline audit `scripts/audit_recorded_carla_candidates.py` completed16recordedinitial/finalstates withfrozen sources. Everyfailedfinalstatehas3–12spatiallyplannableunique candidate targets. This isnotCARLAexecution or dynamic-safety evidence. It rulesout universal lackof candidate paths assolefailurecause. PPOselection/representation/sparselearningremainissues; H staticCARLAtraininghad0successesinrecent18episodes evenafter~69k. Current100kpair willgivefurtherfailure evidence beforechangingobjectiveorpolicyarchitecture. Rewardprogress_weight alreadyexists and isexplicitlynonPBRS; anyfuturetrial must beversioned,sharedwithrelevantbaselines andnotmixedwitholdcheckpoints/results.

Next: inspect/finishrunningmilestones, rebuildtables; confirmCARLAcleanup; continuefailure-drivenlearningandfairbaselines, independentlyseededformalexperiments, native-road/controlvalidation, videos, completeCARLArelease replay andfinalresearchacceptance. Do notmistake uploadedpre-release forgoalcompletion. No external login blocker.
