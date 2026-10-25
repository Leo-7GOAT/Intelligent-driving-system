# Research acceptance ledger

Updated 2026-10-25. This is an implementation and evidence ledger, not a claim that the research goal has been met. The user's latest objective supersedes the earlier six-hour-only and no-push instructions. Each execution window remains bounded and checkpointed. The complete project goal remains active across windows.

Latest development: audited rule-only B3 now completes4/4 known training roads;
fresh seed11 H0/H/B1 CARLA learning retains its original frozen source.
H's completed20,665-control endpoint scores2/4known training roads; H0's
20,702-control endpoint scores3/4. All23,135H/22,589H0training/evaluation
controls are frame-verified, with independent byte-exact raw audits and the
all-eight-case partial H0/H figure. All248H/190H0deterministic decisions are
reconstructed. H has policy-selected waiting failures and one training bounds
episode; H0's failed road is fully stationary/intervened. B1 now completes
20,621training controls and0/4bounds failures in evaluation, with21,565total
controls audited and all96decisions reconstructed. All3registered arms are
complete, total67,289audited controls/534reconstructed decisions; original
owned-server cleanup verified. See the [complete report](../reports/native_route_complete_20261025.md).
This comparison does not support a feedback/PPO-value improvement.
This partial development result does not close any independent-confirmation gate.
No new independent-seed, unseen-confirmation or learned-advantage result exists.
The [initial/trained weight contrast](native_weight_contrast.md) evaluator and
whole-frame/inference auditor are now implemented:295CPU passes/1skip/4CARLA
deselections locally, with actual H0 seeded initialization and parameter-exact
save/reload verified. The actual paired CARLA evaluation remains unexecuted
pending original queue cleanup; no learned-weight benefit is established.
Optional exact road-SAT batching has independent CPU reproduction and a
[published development evidence asset](../reports/road_collision_batch_20261025.md);
it remains disabled in the current training batch. These low-level engineering
and publication advances do not close the capability or research gates below.

## Evidence isolation

Hosted CPU CI is now actually executed at sourcec77314f:Ubuntu24.04/CPython3.11.17
installs the lock and passes276tests,3platform/opt-in skips,4CARLA deselections.
The corresponding local Windows selection passes278/1skip/4deselections. See
[actual CI run and scope](cpu_ci.md). This closes a CPU CI delivery gap, not a
CARLA, policy performance, five-seed or final-confirmation gate.

`runs/20261016T015226Z_research_batch_3377580f` and its frozen source/model/config/result files are retained. All previously inspected primary, generalization and CARLA test manifests are now **historical evaluation sets**. They cannot be reused as unseen confirmation after development. Formal v4 and earlier additionally retain their documented RNG defect. No historical failure or infrastructure exception is removed.

New development batches live in separate directories. Architecture or reward changes require separate versions and compatible checkpoints. A new confirmation protocol must freeze code, model selection, budgets, scene generation, statistical endpoints and multiplicity treatment before any confirmation results are inspected.

## Required gates and current evidence

| Gate | Required evidence | Current status |
|---|---|---|
| Real learned hierarchy | PPO-selected behavior → Hybrid A* → timed reference → MPC → executed state/reward; no execution teleport/autopilot/set-speed | Real learned CARLA transfer/adaptation and frame-linked camera replays exist; full capability gates remain unmet |
| Lightweight capability | ≥85% overall, ≥75% each core class, independent test | Not met: historical H 75%, generalization 4/12 |
| CARLA capability | ≥80% overall, ≥70% static/curve-parking/dynamic; collision or bounds ≤5% | Not met: historical H 9/30 and 10/30, all successes dynamic |
| Productive safety | Progress, time, stops, interventions, collision, bounds, planning/control failures jointly reported | Historical data available; intervention dependence unresolved |
| Hierarchy main hypothesis | ≥10 percentage point mean paired gain over healthy trained flat PPO, 95% interval lower >0, ≥5 independent seeds | Unproven; short-route B1 sanity passed but full-route health unresolved; only 3 complete historical pairs |
| PPO high-level value | Predefined interaction task and endpoint vs strong B3 sharing low level/safety | Unproven: historical B3 matches H's 30/40 exactly |
| Executability feedback | Necessary measured improvement and H/H0 ablation | Equal-network ablations complete: v8 at1M gives8/10vs9/10, v12/v13 give8/10vs6/10; no stable independent gain established |
| Baseline health | Empty/short corridor learning, justified tuning and sufficient comparable budget | Short6m sanity8/8; old fixed23m model2/2 at23m but0/6 shorter. Matched mixed baseline/frequency/pooled actor/reward trials finish1M at0/10; unchanged reward trial now continues to cumulative2M/3M |
| Required comparisons | H, H0, B1, B2, B3, frequency, common safety, fixed-reference controller | Algorithms partly exist; complete independently trained evidence missing |
| CARLA independent training | ≥5 independently trained/adapted policies and real PPO updates | Historical2paired seeds; new v12 seed11H/H0 each receives approximately100kactual world ticks with real updates; five independent current-version policies still missing |
| CARLA final conditions | ≥120 unique paired conditions, ≥30 each core family, unseen layouts/vehicle parameters | Missing; old 30 conditions are historical |
| Feasible task design | Feasibility diagnostics independent of policy performance; retain all failures | Historical witnesses prove 5 failed layouts traversable; remaining cases unresolved |
| Fairness/accounting | Common task/information/rewards/budget/limits/termination; physical/decision/world/reset steps, wall/sim time, networks, updates, takeover | Existing log fields; new comparisons still require audit |
| Timing | P95/P99/max full decision/control cycle and overrun fractions; offline deterministic node budget separate | Historical component timing exists; complete deployment timing not established |
| CARLA lifecycle | Exclusive ownership, exceptions, crash cleanup, checkpoint recovery; only owned actors/processes | Kernel lease and empty-world crash recovery verified; crash with live actors not yet recovered automatically |
| CARLA calibration | Geometry, steering/actuation, coordinates, frames, sensor truth; transfer vs retraining separate | Geometry/frames measured;22phase±1normalized-steer sweep now completed3,520control frames. Native curve geometry surveyed, actual native-road tracking remains unverified |
| Videos | Real success and failure clips tied to case/model/seed/run | Published H25k success/deadline videos with exact model/seed/scene/world-camera manifests, full decoding and fresh-checkout replay; illustrative only |
| Statistics | All seeds, paired intervals, variance, censoring, layered sampling if justified; no pseudoreplication | Historical partial analyses retained; fresh confirmation pending |
| Reproducibility | Clean checkout install/tests/model load/representative evaluation/raw figure rebuild; CARLA replay when available | Fresh clone/newvenv, downloaded models/all20CPUreplays, actual CARLA/control-camera replay verified; updated independent checkout92passed5skipped at e0b2a0a. Final research archive still incomplete |
| GitHub | Correct repository with preserved visibility/history, real commit/push, remote SHA verification and tag/release | Existing public Leo-7GOAT/Intelligent-driving-system verified via matching source/license blob hashes; origin fetched; development pre-release v0.2.0-dev.20261022 published at9bcaa79; remote/tag/asset-download hash verified; research acceptance remains incomplete |
| Release hygiene | Secrets/private path/data scan, third-party licensing/citations, bounded assets and SHA manifest | Development publication scans pass; three prerelease assets have download-verified hashes; final archive pending |
| Documentation | English README, Chinese guide, method/protocol/results/ablations/failures/model card/licenses, bilingual factual CV | Development publication set exists and discloses limits; final formal-results documentation remains incomplete |

No single upload, completed batch, smoke test, subset metric or passing unit suite satisfies the complete objective. If a new confirmation fails, retain it, open a new development version, and use new unseen conditions for any later confirmation.

## Development decisions so far

1. PID-only lock files can persist after forced exit. Use a held operating-system lease that is released by process death; preserve stale PID text as evidence. Respect live legacy owners and reject unknown metadata. Still require an empty CARLA world before controlling it.
2. First matched sanity trial: original beta=1 versus beta=40, same B1 configuration, seed 11, fixed 6m empty task with real parking requirement, development seeds 7301–7308. Both 0/8 after 102,400 physical steps. These nearly identical straight conditions are a learning sanity check, not diverse independent tasks.
3. At failed fixed-horizon termination the terminal-zero potential telescopes away the distance-progress difference. A second explicit task-progress reward trial is separate from PBRS. Both variants receive a from-scratch 1M physical-step budget, with 100k/300k/1M development assessments. If retained, the changed task reward must be shared by H and relevant controls. No rule acts on B1's behalf.
4. Goal-frame candidates are being checked against the existing planner as a separate geometry diagnostic. They are not integrated into PPO or claimed to improve it without evidence.
5. The target diagnostic completed 180 checks. The proposed goal-frame candidates had lower valid-reference counts on these probes, so they remain experimental and are not integrated. Near-goal candidates contain repeated poses; these counts are action availability, not independent scene successes.
6. At 302,080 training steps both reward variants again scored 0/8 in deterministic evaluation. For the same progress-reward checkpoint, a paired inference audit produced deterministic 0/8 versus stochastic gSDE 5/8, identically in train and validation splits (frozen normalization). This is a development clue about inference, not a reason to relabel historical deterministic results or select favorable stochastic replays. Any future inference-mode choice must be frozen before new confirmation.
7. The old generic checkpoint selector ranked any reduction in collision rate before completion, potentially preferring permanent waiting to a productive policy within a declared safety tolerance. The new development-only selector applies an explicit empirical safety constraint, then maximizes completion and progress before minimizing intervention/time. A regression covers the permanent-waiting counterexample. Existing frozen experiments and current running diagnostic jobs are unchanged.

## Update at 2026-10-22 13:13 local

- v6 batch `20261022T042237Z_development_v6`: both fresh B1 reward variants completed 1,000,960 physical steps and deterministic 8/8 on the 6m parking sanity task. The PBRS40 model scored 0/8 at 23m/45s. The short-route task does not establish full-task baseline health. PBRS40 retained; no evidence that explicit task-progress reward is necessary. Rebuilt eight milestone rows with 117 frozen source hashes and model hashes checked.
- New goal-frame candidates, terminal return envelope, analytic Hybrid A* connection and vectorized full-body collision checking are opt-in, versioned implementations. Earlier candidate-count diagnostic was inconclusive; subsequent physical B3 tests justified testing the revised interface, not claiming learned superiority.
- v7 H/H0 runs stopped gracefully after 12,790 / 10,225 physical steps to version the terminal-envelope and vectorization changes. Both exited zero; partial evidence and supersession reason retained. No statistical inference from the censored pair.
- v8 batch `20261022T050532Z_hierarchy_v8_development`: H and H0 use 450-input networks and the same low level. H gets candidate geometric witnesses plus previous execution outcomes; H0 receives zeros in these 99 slots. No action masking or rule replacement. At 100k milestone both score 6/10: H succeeds on static, H0 on narrow, both fail unseen. One seed and development cases only. Run final status must be inspected.
- Actual CARLA B3 diagnostics: v7 3/6, terminal-envelope variant 4/6, with one degraded static condition retained. These are NOT PPO results and do not satisfy a CARLA acceptance gate. All raw frames/RPC and frozen sources retained under v7 batch. Owned empty CARLA server stopped with verified identity, receipt `infrastructure/server_stop.json`.
- Full current CPU regression: 65 passed, 5 conditional skips (`reports/v9_regression_20261022.xml`). Raw XML remains local because it contains personal paths.
- Distance-development protocol compares from-scratch B1 on fixed23m versus uniformly sampled6–23m, common45s deadline/PBRS40/PPO settings and1M budget. Validation distance overrides are explicit; ordinary validation/test remain full length. This is single-seed diagnosis, not an architecture comparison.

## Completed200k pair and launched continuation (13:18 local)

Original v8 pair finished with both exits0: H8/10, H0 6/10. H succeeds on static/narrow/empty/dynamic, both fail unseen. H has higher dynamic intervention fraction (~0.145 versus~0.025), retained as a tradeoff. This is one-seed development only. Both policies now continue unchanged to cumulative1M in a separate provenance-linked batch; inherited interactions are included. Fresh B1 fixed23m versus uniform6–23m training is also live. New learned-policy CARLA transfer is running, distinct from earlier B3 diagnostics. CARLA transfer configuration now preserves learned observation/action semantics and planner flags; a focused regression passed. No new formal acceptance gate is declared passed.

## Learned CARLA transfer completed (13:22 local)

Batch `20261022T051716Z_carla_v8_transfer`: both H/H0 frozen200k policies complete2/6 (curve7302 anddynamic7302), four deadlines each, no observed collision/bounds and no infrastructure faults. H4793 world ticks including resets, H0 4861; these are EVALUATION interactions, not PPO training. Both normalization states frozen. Failed conditions repeatedly select unplannable candidates; action/failure audits retained. No executability-feedback advantage transfers in this small diagnostic. Owned server confirmed empty and stopped. Current300k lightweight development is8/10 for both H/H0, so the200k difference is not stable evidence of feedback superiority.

Latest full regression after transfer configuration changes:66passed,5conditional skips (`reports/v9_transfer_regression_20261022.xml`). Detailed case-level report is `reports/development_20261022.md`. All research and publication gates remain incomplete.

## Publication progress

Development commitbbaf303 was actually pushed to the correct existing public repository; remote SHA verified. A fresh GitHub clone/new isolated environment installed pinned dependencies, passed66tests (5conditional skips), and hash-verified/replayed all20 packaged development episodes. The bounded asset is a pre-release and is not complete final experimental evidence. Original12 archived file hashes are retained byte-for-byte. No hosted CI claim.

## Additional failure evidence and next paired baseline

Offline spatial counterfactuals at every failed frozen-CARLA state found3–12plannable distinct candidate targets at the final stalled states. These are geometry checks only, not actual tracking/dynamic-safety success. Thus absent candidates alone do not explain the repeatedly selected unplannable targets. Raw16state audits are preserved in the transfer batch.

B1 fixed23m and uniform6–23m jobs completed1,000,960physical steps each. Fixed23m succeeds2/2 at23m and0/6 at shorter distances; uniform0/8. A new from-scratch common mixed-task B1 with the same450observations and candidate feedback asH is running in `20261022T054701Z_matched_flat_v11_development`. Its commands remain direct flat control; no planning/MPC control replacement. Candidate feature compute cost and native action frequency are disclosed.

CARLA H actual adaptation reached100,890world ticks,38optimization phases, and3/6development completion; static remains0/2. H0 matched100k adaptation is still running. The improvement from2/6 is not a superiority or convergence claim.
