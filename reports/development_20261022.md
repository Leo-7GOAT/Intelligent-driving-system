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

This report does not replace the acceptance ledger. Five-seed formal comparisons, independent confirmation, current CARLA adaptation, videos, clean-checkout reproduction and GitHub release remain incomplete.
