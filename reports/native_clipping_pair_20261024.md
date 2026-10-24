# Completed native PPO gradient clipping comparison

Batch `20261024T064940Z_native_recovery` completed normally on 2026-10-24.
Six same-parent seed-11 forks compare global gradient clipping with separate
actor/critic clipping. These are development runs on four training roads, not
six independent seeds or held-out confirmation. No clipping change is adopted.

| Method | Global: new controls | Independent: new controls | Global mean progress (m) | Independent mean progress (m) | Global/independent safe completion |
|---|---:|---:|---:|---:|---|
| H0 | 20,160 | 20,160 | 10.3751 | 1.4325 | 0/4, 0/4 |
| H | 20,160 | 20,160 | 16.7472 | 14.8990 | 0/4, 0/4 |
| B1 | 20,087 | 20,119 | 9.8687 | 0.7841 | 0/4, 0/4 |

All 24 evaluation outcomes are deadline failures. H0 mean intervention rises
from 78.33% to 95%; H and B1 have zero evaluation intervention in both arms.
Zero collision/bounds outcomes here cannot be treated as competent navigation:
none completes and several policies stop near their starting positions.

The intended optimization mechanism changed. Actor post-clipping gradient
medians are 0.016289/0.062073 for H0, 0.019069/0.093473 for H, and
0.060546/0.500000 for B1 (global/independent). B1 median approximate KL rises
from 0.00004792 to 0.00061105, while driving progress falls. This rejects
promoting the intervention on a gradient/KL criterion. It does not establish
that separate clipping can never help a different task or training schedule.

Each arm performs 63 optimization phases and 126 optimizer steps. H0/H each
retain 120,320 cumulative trained controls. B1 retains 120,137/120,169, with
charged lower bounds 129,682/129,714 plus an unknown unflushed ancestor tail.
The earlier failed interpreter launch and all lost ancestor interactions remain
in their original logs; neither is erased from accounting. B1 global training
contains one success, 14 bounds failures and 20 deadlines; independent contains
10 bounds failures and 20 deadlines. Training successes are not substituted for
the unsuccessful frozen evaluations.

The six raw audits bind 142,446 applied training/evaluation controls to measured
CARLA frames. Model and normalization hashes, identical inherited checkpoint
bytes within pairs, physical accounting, task/road identity and measured start
poses are checked by `scripts/analyze_native_gradient_pair.py`. All measured
paired start-pose differences are zero. Each arm also has an exact offline
four-start-action inference reproduction with frozen normalization and an
action/terminal-pose diagnostic. These are not new vehicle replays.

Evidence under the batch directory:

- `{job}/endpoint_raw_audit.json`: six raw frame audits.
- `{job}/initial_inference_audit.json` and `action_learning_diagnostic.json`.
- `analysis/complete_pair.json`: full three-method comparison,
  SHA256 `e3200867495c4d3c4bcee317f0b039f97b151d012fe68f8f57f8061d88c2ce4c`.
- `analysis/all_endpoints.png` and its JSON manifest: all 24 measured trajectories.
- `guard_final.json`: exit code zero, no guard stop reason.
- `infrastructure/server_stop.json`: owned empty server stopped, TCP/UDP ports released.

The independent checkout at source `92eee2b` rebuilds the complete comparison
byte-for-byte; receipt:
`runs/prerelease_20261022/native_road_checkout/runs/native_complete_clipping_pair_reproduction.json`.
That receipt uses local raw evidence. The separate downloaded reproduction is
recorded below.

## Published asset and independent downloaded reproduction

The completed batch is now published as
[native_clipping_seed11_20261024.zip](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_clipping_seed11_20261024.zip):
110,064,724bytes,639manifestfiles, SHA256
`bbea000c380e202c1801770679413dcf616b59dfc593423a5e57a31005f2bcfc`.
GitHub reports uploaded state and matching size/digest. It was exported from
clean source `d35572bba8f4ce320e6838d32c0f66d1d13bd79e`; the historical prerelease
tag and all previous assets remain unchanged. Twelve auxiliary metadata files
have explicit original/export hashes for path redactions. Model, normalization,
checkpoint and raw audit bytes are unchanged. All six arms and failed outcomes
are included; upstream raw traces are not duplicated, but their costs remain.

A separate actual GitHub download verified all639files before model loading.
The independent checkout at the same source rebuilds all six raw audits,
24measured-start actions, the complete paired report and the trajectory figure.
The figure SHA256 is
`847e27ff7bf5a5e7136117f2e83eb71ce38b43f7284811ce8a471ef701b00014`.
No new driving or optimization occurred. Reproduction receipts are
`runs/prerelease_20261022/native_road_checkout/runs/native_clipping_downloaded_reproduction/reproduction.json`
and `pair_reproduction.json` in that same output folder.

The next declared hypothesis concerns
[on-policy rollout length](native_rollout_horizon_20261024.md). It retains global
clipping for all three methods and does not redefine the original research
acceptance criteria.
