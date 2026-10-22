# Active research handoff — 2026-10-22 13:23 local

Goal remains active and incomplete. Read the current user objective attachment before continuing. Existing public repository `Leo-7GOAT/Intelligent-driving-system`, branch `feature/physical-hierarchical-ppo`, base8142224. No new commit/push yet. Do not modify other projects, reset/reinitialize Git, erase failures, or reuse inspected final conditions as unseen confirmation. No subagents authorized. Historical handoff retained in `reports/handoff_before_v6_20261022.md`.

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
