# Structured road policy development — 2026-10-23

## Updated endpoints and live continuation — 2026-10-24

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

Current working-source regression is **163 passed, five conditional skips**, with
two existing SB3 deprecation warnings. This does not replace live frame audits.
The new continuation is not complete; no performance or research gate is met.

```powershell
python scripts/audit_native_method_endpoint.py --batch COMPLETED_BATCH --method H --output NEW_AUDIT.json
python scripts/combine_native_endpoints.py verify --manifest runs/native_combined_endpoints_20k_to_100k_verified_v2.json
```

The sections below retain the historical registration and earlier observations.

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
