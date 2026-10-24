# Structured road policy development — 2026-10-23

## Gradient-clipping mechanism diagnostic — 2026-10-24

The first actual continuation rollout of H0 and H was reconstructed from saved
RNG, measured CARLA states, prior commands, feedback and sequential observation
normalization. Each reconstruction exactly matches all 32 sampled actions (320
controls), and reproduces the baseline actor change and logged PPO metrics.
No simulator connection, plant integration or new environment interaction occurs.
The current-code receipts are `H0/rollout_update_audit_v4.json` and
`H/rollout_update_audit_v4.json` in `runs/20261024T050724Z_native_continuation`.

| First-rollout measurement | H0 | H |
|---|---:|---:|
| Actor gradient L2 before clipping | 0.06462 | 0.07251 |
| Critic gradient L2 before clipping | 12.15327 | 11.77603 |
| Global gradient multiplier | 0.04114 | 0.04246 |
| Baseline update exact distribution KL | 2.1603e-9 | 1.4008e-8 |
| Independently clipped update KL | 6.4071e-7 | 1.7676e-6 |

All alternative optimizer branches are discarded in-memory diagnostics on the
same rollout. Increasing epochs or changing Adam epsilon were also measured and
retained in the receipts. The selected next hypothesis changes only the clipping
groups: separate disjoint actor and critic parameters, each with the existing
0.5 bound. The PPO surrogate, epochs, learning rate, Adam state, reward,
observations and low-level execution remain unchanged. This suggests critic
domination of the shared clipping factor; it does not establish the cause of
all training failure. Component magnitudes relative to Adam epsilon are logged,
but Adam's running moments prevent interpreting that comparison alone as an
effective learning-rate measurement. Larger KL is not evidence of better driving.

`navstack/independent_clip_ppo.py` calls the installed SB3 PPO update and applies
per-group clipping through an optimizer pre-step hook. It rejects shared or
unclassified trainable parameters, removes hooks even on failure, and saves an
explicit algorithm marker. Loading an existing checkpoint preserves its mode;
changing mode requires an explicit registered intervention. No SB3 training-loop
source is copied. Tests exercise real PPO/plant learning, both structured policy
types, exact save/reload, and actor-update invariance when only critic targets
change. Full regression is **173 passed, five conditional skips**, with two
existing SB3 warnings (`runs/native_gradient_final_regression.xml`).

Source commit `c5f82852653a91bc677cb0a5a8259a8f77dd9879` was actually pushed and
matched against the correct remote branch. A clean independent checkout with
its separate pinned environment also passes **173 tests, five skips** and
rebuilds both v4 rollout receipts byte-for-byte. Receipt and executable recipe:
`runs/prerelease_20261022/native_road_checkout/runs/native_gradient_reproduction.json`
and `verify_native_gradient_reproduction.py`. This reuses local raw artifacts and
the original frozen observation source, not downloaded release data or another
CARLA drive. Both replay outputs account for zero new environment controls.

`scripts/prepare_native_gradient_pair.py` is implemented and tested, but the
six-arm CARLA study is **not prepared or started** at this snapshot. After the
current H0/H/B1 batch completes and its endpoints and owned-server cleanup pass
audit, it will branch each endpoint into global/independent clipping arms with
20,000 additional controls each. Both arms inherit identical checkpoint bytes,
Adam/RNG/normalization state and all prior lost-interaction costs. The preparer
refuses an active continuation, atomically claims all three parent endpoints,
preserves per-job targets for recovery and checks copied checkpoint hashes before
registration. These are correlated seed-11 forks, not six independent seeds.
The decision measure is completion/progress jointly with safety and intervention
on all four training roads; no model will be selected solely by KL. Both arms
record gradients after clipping to test whether the proposed mechanism persists.

Reproduce the diagnostic using the original frozen observation code:

```powershell
$env:NAVSTACK_AUDIT_SOURCE_ROOT=(Resolve-Path runs/20261024T050724Z_native_continuation/source).Path
python scripts/audit_native_rollout_update.py --batch runs/20261024T050724Z_native_continuation --method H --output runs/NEW_H_rollout_audit.json
Remove-Item Env:NAVSTACK_AUDIT_SOURCE_ROOT
```

The new output path must not exist. Preparation after complete endpoint audits:

```powershell
python scripts/prepare_native_gradient_pair.py --parent runs/20261024T050724Z_native_continuation --additional-controls 20000 --wall-seconds 21600
```

Preparation alone launches no process. The current baseline batch continues from
its immutable source; this intervention has not changed its learning semantics.

## Updated endpoints and live continuation — 2026-10-24

**First 100k endpoint (13:22 local):** H0 has finished this continuation with
80,000 new controls, 82,758 training world ticks and 500 new optimizer steps;
the lineage and charged training total are both 100,160, with 626 cumulative
optimizer steps. Reload is exact. All 88 newly completed training episodes are
deadlines. Frozen evaluation remains **0/4**, all deadlines. Progress on the four
roads is 20.8904/1.5977/20.1850/10.8140 m; intervention fractions are
0.6222/0.9333/0.6000/0.7333. The measured-start action probabilities remain
near uniform (maximum 0.03252–0.03307). Larger progress on some roads does not
meet task completion or establish a learned-policy advantage.

The completed H0 segment's raw audit verifies all **83,600** training/evaluation
controls, with audit SHA256
`b2b3d2d5c80ad5aae96609afa399317b42938b96f8d2d3a64e0ef5a29976abe6`.
It also verifies 3,724 evaluation world ticks and the 626-step inherited optimizer
accounting. H is now training in the same registered worker; B1 is queued.
Neither is claimed complete at this snapshot.

The original structured-policy H0 and H endpoints each complete 20,160 training
controls, 20,872 world ticks and 126 optimizer steps. B1's completed recovery
`20261024T044910Z_native_recovery` adds 9,870 controls, 10,427 world ticks and
62 optimizer steps, reaching 20,061 model-lineage controls and 126 retained
optimizer steps. Its charged training interaction lower bound is 29,606: the
9,545 controls lost since the inherited checkpoint remain charged, and the
interrupted ancestor has an unknown unflushed tail. It is not a cost-matched
20k comparison simply because the model lineage is approximately 20k.

All three frozen endpoint probes score **0/4** on the four training roads:
H0 and H time out; B1 leaves the road in all four cases. H0/H actor L2 changes
are 0.21155748/0.39653084. H0/H initial action probabilities remain nearly uniform
(maximum approximately 0.0326/0.0340); actual parameter updates do not yet imply
useful decisions. The measured-start inference audit uses frozen evaluation
normalization and zero additional CARLA ticks.

Per-method `endpoint_raw_audit.json` verifies 23,760 controls for H0, 23,760 for H
and 10,500 for the completed B1 recovery, including evaluation. Every applied
control is matched to a bridge frame, state, command and IMU timestamp; reset
ticks, terminal fences, optimization counts and inherited costs are checked.
The combined manifest `runs/native_combined_endpoints_20k_to_100k_verified_v2.json`
also checks model/optimizer/normalization/RNG compatibility and retains the
interruption chain. Manifest ID:
`e9fadedaa04acb8716113842530cd052c67f2d204f6b06b414c07410f26a4b00`.

Batch `20261024T050724Z_native_continuation` was actually launched at 13:08 local
on 2026-10-24, with H0/H/B1 additional budgets 79,840/79,840/79,939, a six-hour
wall limit, disk/RAM/GPU guards and no automatic retries. Its pointer is
`runs/native_continuation.json`. Learning semantics and low-level code are frozen;
the continuation starts new episodes while preserving learning state. These are
continued seed-11 policies, not new independent seeds. At launch H0 produced
real new CARLA controls and optimization events; final results remain pending.

CARLA startup on port 2000 had failed inside the RPC socket acceptor. The matching
packaged PDB and Windows port ownership supported a bind conflict with an
unrelated local application. Port 2100 (span 2100–2102) was free and the installed
0.9.15 server then started successfully. The earlier recovery completed and its
owned empty server was stopped with no remaining listeners. The new launcher
records atomic process-identity/exit receipts; cleanup verifies the target
process, bridge drain and empty matching world before stopping it.

The earlier source regression was **163 passed, five conditional skips**, with
two existing SB3 deprecation warnings. This does not replace live frame audits.
The new continuation is not complete; no performance or research gate is met.

```powershell
python scripts/audit_native_method_endpoint.py --batch COMPLETED_BATCH --method H --output NEW_AUDIT.json
python scripts/combine_native_endpoints.py verify --manifest runs/native_combined_endpoints_20k_to_100k_verified_v2.json
```

The sections below retain the historical registration and earlier observations.

Source `4b7db609c965696a6eeab2ebb794cbf0588052fe` is pushed and remote-matched.
The clean independent checkout, using its separate pinned environment, passes
163 tests with five conditional skips and two existing warnings. All three
completed parent endpoint audits rebuild byte-for-byte, covering 58,020 actual
training/evaluation controls. Receipt:
`runs/prerelease_20261022/native_road_checkout/runs/native_continuation_reproduction.json`.
This reuses local raw artifacts and does not claim release-asset download or
another live-drive replication. A later whitespace cleanup briefly broke a test's
indentation in the intermediate commit; `4b7db60` fixes it and is the tested head.

`scripts/diagnose_native_action_learning.py` groups actual observed actions and
their rewards/progress by action ID (or direct-control bins). It verifies the
transition hashes against each endpoint audit. At the 20k H0/H endpoints all
44 completed training episodes are deadlines. Their median PPO approximate KL
values are 7.45e-9/3.91e-8, with zero clipping and near-maximum categorical entropy.
These metrics support weak policy-distribution change, not failed parameter
updates. Conditional action outcomes are confounded by visited state and cannot
be interpreted as causal action values. No new controls or teacher queries were
used for this diagnostic; outputs are each method's `action_learning_diagnostic.json`.

The native-road MLP experiments remain negative evidence: H0 and the recovered
H/B1 approximately 100k-control endpoints complete none of the four training-road
probes. Fixed-plan Hybrid A* and 2D A* diagnostics each complete all four with the
same indexed road geometry; they are controller diagnostics, not learned policies.
This motivates testing the policy representation before adding training budget.

## Registered change

Batch `runs/20261023T062141Z_native_road_learning` freezes source, calibration,
four measured roads, seed 11, and a two-hour window ending 16:21:41 local time.
H0, H and B1 each receive a nominal 20,000 actual control steps, with rollout
overshoot and world/reset ticks reported separately. Checkpoints are saved at
10k and 20k. All four training roads are evaluated deterministically after each
method finishes. These probes are neither held-out conditions nor final testing.

The new optional `--road-candidate-policy` uses 4,701 common inputs: 605 existing
state/candidate/outcome features plus the same 64x64 binary road raster. Separate
actor and critic CNNs encode the raster. H/H0 use shared candidate scores; B1
pools the same candidate features and directly predicts two actuator commands.
Candidate geometry and binary road cells keep their fixed scales. Other state
features retain training normalization. H0 zeros only execution feedback.

This is a representation bundle relative to the old flat MLP, so a comparison
with that version cannot isolate CNN architecture from feature encoding and
normalization. H versus H0 within this batch isolates the feedback under the
same architecture. B1 receives the same information, with parameter counts
recorded rather than assumed equal. H/H0 have 207,522 parameters each.

Action catalog (`ego_arcs`), reward, vehicle, controller, road geometry, 0.05 s
control period, 10-control action hold, 45 s deadline, PPO learning rate, 32-action
rollouts and two update epochs are unchanged. There is no new teacher, action
mask, route prior or rule replacement. These are fresh incompatible policies;
no old model is silently loaded.

## Actual execution and evidence limits

Client and server were verified as CARLA 0.9.15 with an empty world before launch.
Worker 18088 (creation epoch 1792736844.863) and guard 12856 were OS-confirmed.
The first inspected H0 log records 4,800 actual controls, 15 optimization phases,
30 optimizer steps and actor L2 change 0.03812. This proves learning activity,
not task competence. All completed and failed endpoints must be retained.
Revalidate process identity and current markers instead of treating this snapshot
as permanent liveness. Launch resources and identity are saved in the batch.

The frozen guard enforces disk, RAM, temperature, deadline and progress limits.
It attempts shutdown only for the recorded owned server after bridge exit and a
matching-version, empty-world probe. Inspect both `server_cleanup.json` and
`infrastructure/server_stop.json`; attempted cleanup is not proof of shutdown.
A later working-source patch additionally records cleanup subprocess timeouts
and launch failures. It does not change the running frozen source.

## Verification and reproduction

Tests use the real lightweight plant, Hybrid A* and PPO to check identical inputs
across methods, exact feedback ablation, candidate permutation behavior, road
input gradients, actor CNN updates, normalizer round trips and exact model reload.
They are not substitutes for CARLA frame verification. Fault-injection cleanup
tests check durable error receipts and no retry, not actual server shutdown.

```powershell
python -m pytest -q
python scripts/native_road_learning.py --prepare SURVEY_BATCH --road-geometry-version carla_driving_indexed_union_64_v3 --controlled-budget 20000 --wall-seconds 7200 --episode-seconds 45 --evaluate --feedback-ablation --road-candidate-policy
# Preparation creates a new batch; the current batch is already running.
# After all methods and bridge logs are closed:
python scripts/analyze_native_road_learning.py --batch COMPLETED_BATCH
```

The analyzer checks frozen source hashes and matches training/evaluation control
states, frame sequences and IMU timestamps against raw bridge responses. It
requires all methods and does not certify an incomplete batch. New artifacts
remain local until an explicit publication receipt is recorded. No five-seed,
generalization, hierarchy-advantage or final acceptance claim is made.

## Closed H0 endpoint and independent verification

Source `3c9e6f2b0f48fc15755b479d9010c6e13f0c0364` is pushed and remote-matched.
Both root and the clean independent checkout, using its pinned environment and
working directory, pass **146 tests, five conditional skips**, with two existing
SB3 deprecation warnings. The earlier invocation from the main working directory
is not used as independent-import evidence; the subsequent checkout-local run is.

H0 completes 20,160 controls, 20,872 training world ticks, 63 optimization phases
and 126 optimizer steps. Actor L2 change is 0.21155748 and model reload is exact.
All 22 completed training episodes are deadlines; one unfinished episode remains.
Its frozen four-road evaluation is **0/4**, with four deadlines and 346 rejected
endpoint plans. Evaluation uses 3,600 controls and 3,724 world ticks. Independent
checkout verification matches all **23,760** H0 training/evaluation controls to
raw CARLA state/frame/IMU records; it does not audit the still-active H or B1.
Receipt: `runs/prerelease_20261022/native_road_checkout/runs/road_candidate_h0_frame_audit.json`.
Raw response archive SHA256:
`e0d48cbbea47d30550c024bfff12bd3bce1b2ae9ed53d8f9e989ca886db557f9`.

The same checkout loads the actual H0 model and frozen normalizer, reconstructs
observations at all four measured evaluation starts and reproduces actions
29/30/21/29 exactly. No vehicle step is simulated in this offline check. Action
probabilities are nearly uniform: maximum probability 0.03249–0.03263, entropy
3.43398 (uniform 31-way entropy approximately 3.43399). Thus optimizer activity
has not yet produced strong action discrimination; deterministic selection of a
slightly larger probability can still repeatedly choose an invalid goal. This
does not prove the cause or predict the H result. Receipt and executable recipe:
`runs/prerelease_20261022/native_road_checkout/runs/road_candidate_model_reproduction.json`
and its `.py` companion. Model SHA256:
`9c4ee2d94eafcf50e9c5eb00ee6e0618ef0a5806aa459a914b226d0b94daab65`.

At 14:38 local, the original identity-checked worker/guard are still live; H is
past 9,600 controls and B1 is pending. Existing lightweight normalization
continuation also remains live. These are snapshots, not final paired outcomes.
