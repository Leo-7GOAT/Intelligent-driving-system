# Development evidence — 2026-10-22

These are development results, not independent confirmation. No hierarchy-superiority or road-safety claim.

| Experiment | H | H0 / alternative | Limitation |
|---|---:|---:|---|
| Lightweight feedback, 100k | 6/10 | 6/10 | One seed |
| Lightweight feedback, 200k | 8/10 | 6/10 | Both fail unseen layouts |
| Lightweight feedback, 300k | 8/10 | 8/10 | 200k difference not stable |
| Actual CARLA transfer of200k policies | 2/6 | 2/6 | Four deadlines each, no CARLA optimizer updates |

B1 PBRS40 and explicit-progress variants each completed1,000,960 physical steps and8/8 on6m parking. The PBRS model failed23m/45s transfer0/8. Fresh full23m training reaches2/2 on23m at302,080 steps but0/6 on6/12/18m probes; fresh uniformly sampled6–23m training is0/8 at the same milestone. Both continue to1M; these near-identical straight conditions only diagnose learning and distance generalization.

## CARLA case results

| Policy | Scenario | Condition | Success | Progress m | Intervention fraction |
|---|---|---:|---:|---:|---:|
| H | static | 7301 | 0 | 2.79 | 0.900 |
| H | curve | 7301 | 0 | 7.55 | 0.667 |
| H | dynamic | 7301 | 0 | 2.88 | 0.922 |
| H | static | 7302 | 0 | 0.31 | 0.967 |
| H | curve | 7302 | 1 | 22.60 | 0.039 |
| H | dynamic | 7302 | 1 | 22.50 | 0.020 |
| H0 | static | 7301 | 0 | 2.92 | 0.833 |
| H0 | curve | 7301 | 0 | 4.07 | 0.900 |
| H0 | dynamic | 7301 | 0 | 5.52 | 0.778 |
| H0 | static | 7302 | 0 | 0.00 | 1.000 |
| H0 | curve | 7302 | 1 | 22.66 | 0.000 |
| H0 | dynamic | 7302 | 1 | 22.42 | 0.164 |

The failed learned-policy cases repeatedly request unplannable candidates (search-budget exhaustion or endpoint body collision/bounds). Low collision counts coexist with poor progress and repeated braking; they are not evidence of effective safety. H uses4793 actual world ticks including resets (4608 control), H0 uses4861 (4676 control). Each policy has six completed cases, no infrastructure-unresolved cases, frozen normalization. Both succeed oncurve7302 anddynamic7302. The curve task uses a curved obstacle route on a straight roadbed, not a native curved-road claim.

## Provenance and reproduction

- Lightweight200k: `runs/20261022T050532Z_hierarchy_v8_development`, frozen132source files; model hashes checked; raw optimizer/episode/action/frame data retained.
- Lightweight continuation: `runs/20261022T051509Z_hierarchy_v8_continuation`; parent snapshot/RNG/optimizer/environment restored, inherited interactions included in cumulative counts.
- B1 distance: `runs/20261022T051509Z_distance_v9_development`;135frozen source files; no hidden warm start.
- CARLA transfer: `runs/20261022T051716Z_carla_v8_transfer`; frozen sources/model hashes verified, actual RPC traces and scene metadata retained; controlled server shutdown receipt available.
- Rebuild paired learning tables/figures with `scripts/summarize_paired_development.py --batch <batch>`. Rebuild6m sanity with `scripts/summarize_development.py --batch runs/20261022T042237Z_development_v6`.

This report does not replace the acceptance ledger. Five-seed formal comparisons, independent confirmation, complete CARLA acceptance and videos remain incomplete. Development publication and representative clean-checkout reproduction are now verified below.

## Completed milestones and stronger negative controls (14:10 local)

| Development experiment | Actual interactions | Safe completion |
|---|---:|---:|
| H lightweight continuation | 1,000,551 physical steps | 8/10 |
| H0 lightweight continuation | 1,001,774 physical steps | 9/10 |
| H CARLA adaptation from200k policy | 100,890 world ticks,96,955 control | 3/6 |
| H0 CARLA adaptation from200k policy | 100,985 world ticks,96,895 control | 3/6 |
| Strong B3 on identical v8 lightweight cases | Zero PPO training; actual evaluation | 10/10 |

Both lightweight continuations complete396 optimization phases. Both CARLA runs complete38 phases/1,520 optimizer steps and exact model reload; each has three deadlines, no observed collision/bounds episodes and no unresolved evaluation infrastructure fault. The CARLA source models are the same200k pair used for frozen transfer, not the later1M pair. The B1 distance jobs finish1,000,960 physical steps each without fixing distance generalization: fixed23m remains2/8 pooled probes (2/2 at23m,0/6 shorter), uniform6–23m remains0/8.

B3 evidence: `runs/20261022T060649Z_candidate_v12_development/reference_v8_B3`, using all132 verified frozen v8 source files, identical cases and H configuration except method/split, with rule-selection time, search diagnostics and all transitions saved. Its10/10 includes both unseen cases failed by H. Thus these simple tasks provide no evidence of PPO's advantage over the strong rule high level. No confidence interval is justified from this single training seed and nearly duplicated scene conditions.

## Candidate representation experiment v12

Repeated unplannable policy choices coexist with3–12 spatially plannable candidate targets at each inspected failed CARLA terminal state. v12 tests whether explicit candidate geometry and shared scoring improve learned associations. A605-input observation contains the original351 values,31x8 candidate entries and six execution outcomes. Candidate entries are ego x/y scaled by15m, sine/cosine of target yaw, speed scaled by3m/s, endpoint clearance flag, analytic witness flag and estimated travel time scaled by45s. The last three entries and previous outcomes are zero in H0; both get identical geometry and network shape. Candidate scales bypass per-slot observation normalization so identical candidates retain identical meaning.

The actor encodes common state and scores each candidate with the same learned weights. Standard SB3 PPO samples all31 actions; there is no masking, teacher or rule replacement. Reward, actions, planners, controllers and training scenes remain the v8 versions. H/H0 start from scratch at seed11 with milestones0/25k/100k/300k/1M in a two-hour supervised window. Frozen batch: `runs/20261022T060649Z_candidate_v12_development`; the separate v11 B1 continues its own frozen450-input protocol. v12 versus v8 changes geometry exposure and network architecture together; only the within-v12 H/H0 comparison isolates feedback. Formal B1 must receive the new common information before any claim against v12.

Tests include common information across H/B1/B2/B3, correct feedback ablation, candidate-order permutation of action probabilities, all actions retaining nonzero probability, fixed candidate normalization, actual PPO updates, cumulative physical-state resume and custom model/normalizer reload. The first full regression after this change passed69 tests with5 conditional skips; an additional605-input CARLA calibration-preservation test follows.

Publication: GitHub development pre-release `v0.2.0-dev.20261022` exists at9bcaa79; the branch subsequently advanced to2bc2750. Clean tag checkout passed66 tests/5 skips and20 downloaded-model replays, and reproduced one actual CARLA success plus one expected failure. See `docs/release_status.md` for asset hashes and evidence. Publication does not close research acceptance.

At the fixed100k milestone, v12 H/H0 complete8/10 and6/10 respectively (101,923/101,965 actual training steps). Frozen transfer of these exact models to calibrated CARLA completes4/6 versus3/6, retaining two/three deadlines. H succeeds onstatic7302,curve7302 andbothdynamic cases; H0 succeeds oncurve7302 andbothdynamic cases. Neither records a collision/bounds episode; small samples and braking remain limitations. This transfer uses3,459/4,007 actual world ticks including reset, no PPO updates, frozen normalization, no unresolved episode infrastructure failure. Evidence: `runs/20261022T061420Z_carla_feedback_transfer`.

Paired real CARLA adaptation of this100k pair is actually running in `runs/20261022T061639Z_carla_feedback_adaptation`, with equal25k/100k world-step milestones including reset. H has already completed actual optimizer updates; H0 waits its interleaved turn. It must not be confused with the completed v10 adaptation of the old200k pair. Full regression after all policy/calibration tests:70passed5skipped. The new initializer can create the documented candidate task without historical data; equality to v8 task/cases was checked.
