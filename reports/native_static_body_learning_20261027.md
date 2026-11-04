# Measured static-body geometry and shared progress learning

Status: rule-only B3 improves from2/4 to3/4 on the four known full native roads.
The original failed conditions remain. This is a geometry/control engineering
result, not a PPO contribution, independent confirmation or final acceptance.
PPO H/H0/B1 development continues under a new declared shared reward profile.

## Actual geometry contrast

`20261027T003800Z_native_rule` replaces the parked car's single axis-aligned
box with12closed x-strip rectangles enclosing the convex projection of its
eight actual CARLA bounding-box vertices. Every hull edge is among the point
pairs used to calculate strip extrema. A1e-7m outward tolerance prevents seams;
the construction covers the measured body and does not shrink an obstacle.
CARLA still supplies collision truth. All methods receive the same rectangle
information and driving-surface raster; no future actor script is exposed.

Routes, actor spawn, measured vehicle,7,200node limit per search, Dubins guide,
trajectory/MPC, reference-reuse logic and success gates match the preceding
`20261026T140943Z_native_rule` batch. Only static representation/source version
changes. All four conditions execute, not just the previous successful ones.

| Known road | Result | Actual controls | Full cycle P95 /P99 /max(s) | Over0.5s |
|---|---|---:|---:|---:|
| 0 | success | 361 | .335 /.373 /.381 | 0% |
| 1 | deadline | 900 | .622 /.677 /.779 | 32.2% |
| 2 | success | 317 | .312 /.354 /.372 | 0% |
| 3 | success | 348 | 6.065 /8.574 /8.602 | 22.9% |

All1,926applied controls and123reset ticks audit against the bridge. Guard exits0
and the owned empty server stops normally. Scene3's improvement supports keeping
the tighter conservative body representation for further development. Search
latency still violates the decision period in some conditions: no realtime or
safety certification is claimed.

Road1still stalls. On this same measured scene, the original forward lattice
exhausts its open set after112nodes; a finer.2m/.075rad/.4m-primitive lattice
exhausts after970nodes at both7,200and28,800node limits. These bounded search
failures are not a proof of continuous physical infeasibility and do not justify
deleting the condition or attributing its failure to PPO alone.

## Completed policy filter and process-loss recovery

The prior endpoint-filter model completes4,160real training controls/130Adam
steps. All30initial tensors and postinitialization RNG match its unfiltered
control. It has317plan_failed/20reference/79wait decisions, versus392/12/12;
the mask retains unknown connections and excludes only known blocked endpoints.

Its evaluation is interrupted by process loss of unknown cause. Fresh OS checks
verify worker/guard/server absence and free2200..2202, without inventing an exit
code or normal-cleanup receipt. The complete4,160training frames, all frozen
source hashes, checkpoint bytes and optimizer count verify. Old readable
evaluation controls have a1,541lower bound; unclosed gzip tails stay unmodified.

`20261027T002654Z_native_evaluation_recovery` loads the exact saved model and
normalizer, performs **zero** new training/optimizer updates, and evaluates all
four original conditions. Result: **0/4**, all deadlines,3,600new controls fully
audited. Guard exits0 and owned cleanup completes. This repeats development
conditions and does not add an independent seed. The filter alone is insufficient.

## New shared physical-step reward experiment

The terminal-zero potential term telescopes on failed finite-horizon episodes;
it does not reward partial progress when two episodes end at the same deadline.
Earlier H training saw no success and overwhelmingly failed plans, while its
frozen policy learned prolonged waiting. A new explicit task-reward profile
addresses this exploration signal; no policy-invariance claim is made.

`guarded_progress_v1` adds.5times the measured reduction of distance to the
global goal per physical step. Collision and bounds events both become−90.
Success/deadline gates, physical-time discount.99/s, time/risk/comfort weights,
potential and all low-level modules remain unchanged. All selected learning
methods receive exactly the same reward definition. There is no local-target,
planner-success or teacher-action reward.

The profile is restricted to initial goal distance<=40m and episode time<=45s.
Its discounted progress contribution is at most20 and terminal-zero PBRS at
most1. An unsafe return is therefore at most21−90*.99^45≈−36.3 when other costs
are ignored; a waiting return is at least−(.1+.4+.04)*45−10=−34.3 under the
declared component bounds. This prevents the progress term from making a
terminal unsafe shortcut preferable to waiting under these bounded assumptions.
It does **not** guarantee what PPO learns or establish actual driving safety.

Actual batch `20261027T004817Z_native_hazard_learning` trains fresh H/H0/B1,
seed11, each to a nominal20,480new physical controls, ten PPO epochs per nominal
320-control rollout, with all overshoot recorded. It uses the measured strip
geometry and necessary-endpoint filter; H0 ablates feedback and unknown feedback
remains unmasked, while B1 directly selects speed/steering. All four original
full roads remain. No inherited model, normalization, replay or teacher is used.
H now completes20,761training controls/2,080decisions/65PPO phases/650Adam
updates in1,446.545s. Training ends:8successes and19deadlines. Frozen known-road
endpoint is2/4: roads0/3succeed, roads1/2deadline. Road2has900interventions and
90planner failures; the free selected endpoint still times out at7,200nodes.
All23,273training/evaluation control frames audit. This does not establish PPO
value over B3, whose result is3/4, or a completed H/H0/B1 comparison.

H0 reached10,240controls/320Adam updates, then directory publication raised
WinError5. The complete staged checkpoint hashes validate; original partial
files remain immutable. Bounded publication retry is now implemented without
rewriting bytes or overwriting targets. Recovery014402retains model, normalizer,
optimizer and RNG; a prior recovery013823failed before any interactions because
its launcher omitted the client path, and that failure is retained separately.
H0 adds10,240nominal controls, then B1 starts fresh for20,480. The parent's
unknown buffered tail remains explicitly charged as unknown.

The optional kinematic-arc candidate prototype addresses valid endpoints whose
heading/position mismatch consumes the planner budget. On the measured initial
road2scene,14/14clear local targets connect versus10/14old targets. The repeatedly
selected old action10times out at7,200nodes; the new pose connects in5nodes.
This is an offline initial-scene diagnostic, not a CARLA execution result.
The catalog has a new signature and requires fresh models; live recovery keeps
the old catalog. Both retain the full-body planner, timed trajectory and MPC.

## Verification and publication scope

Source7ae55abis actually pushed and hosted CI40427033435passes366tests/3skips/
4CARLA exclusions in164.33s. Independent Windows checkout passes368/1/4in137.03s.
Subsequent geometry/reward code passes380/1/4locally in124.76s;27geometry/audit
and20reward/registration targeted checks pass. The bridge geometry module also
compiles under the actual isolated Python3.7 CARLA client.

```powershell
python -m pytest -q -m "not carla"
python scripts/audit_native_rule.py --batch runs/20261027T003800Z_native_rule --output runs/body_recheck.json
python scripts/prepare_native_hazard_learning.py --parent runs/20261027T003800Z_native_rule --controls 20480 --wall-seconds 7200 --seed 11 --methods H H0 B1 --endpoint-filter --guarded-progress
```

Preparation is not execution. Already registered/started batches must not be
restarted. Use the existing owned-server launcher/probe and identified-worker
guard; fresh recovery needs current OS absence evidence. The latest
[handoff](../docs/handoff.md) and [measured ledger](native_hazards_cadence_summary_20261026.json)
distinguish live, interrupted and complete phases. All known roads remain
development data. Five independent strategies, healthy full B1, new paired
final conditions, formal RL value and original research acceptance remain open.


## Public negative-result capsule

The prior4k H/H0/B1 frozen0/4models and all seven B3 hazard phases are now
actually public and independently replayed. The PPO capsule verifies485files,
23,219raw controls,12measured-start actions and a byte-identical PNG; the rule
capsule verifies2,556files and18,238controls including failures. See the
[download/reproduction receipt](native_hazard_public_reproduction_20261027.json)
and [model card commands](../docs/model_card.md#public-hazard-evidence-2026-10-27).
The first PPO `_full.zip` figure replay failed because the reference PNG was
missing; the original asset and failed attempt remain preserved. Recommended
`_repro.zip` adds the raw-derived reference figure without changing models/raw.
New20k models remain local; publication is not formal research acceptance.


## Complete initial-scene candidate coverage

[All-four-condition receipt](native_candidate_geometry_20261027.json), with the
same measured body, road, vehicle and7,200-node search budget per unique pose:

| Known road | Offset clear / connected actions | Arc clear / connected actions |
|---|---|---|
| 0 | 14 / 10 | 14 / 14 |
| 1 | 6 / 0 | 12 / 8 |
| 2 | 14 / 10 | 14 / 14 |
| 3 | 8 / 6 | 12 / 10 |

Counts include two speed actions per pose. Better local connections do not
prove progress through the entire road, valid dynamic execution, or learned
benefit. Reproduce both catalogs using `scripts/probe_native_candidate_coverage.py`
with `--batch <complete-rule-batch> --final-goal-candidate --all-candidate-plans`
and `--candidate-geometry route_offsets_v1` or `kinematic_arcs_goal_v1`, each
with a new `--output` file. No CARLA server is contacted by this diagnostic.

Latest Windows CPU regression:388passed/1skipped/4CARLA exclusions in127.68s;
the subsequently added deduplication/no-execution inventory test also passes
(all4candidate tests). The checkpoint/recovery/hazard subset passes23tests.


Verified recovery endpoint update: H0 has completed10,240new controls and320new
Adam updates, reaching20,480lineage controls/640retained updates. Frozen result
is0/4,all deadlines; this segment's13,840training/evaluation controls pass the
raw CARLA frame audit. The interrupted ancestor's unknown tail remains flagged.
B1 is now the active method in014402; no complete paired comparison yet.


## Closed guarded-reward endpoints and the next isolated catalog

[Frame/model/cost ledger](native_guarded_results_20261027.json): H2/4,H0zero/4,
B1zero/4 on the four known full static roads. H/H0/B1lineage controls are
20,761/20,480/20,710; charged B1training is21,640including930lost checkpoint-tail
controls. Retained Adam updates are650/640/660. B1is not a healthy comparator:
its final outcomes are deadline, collision, bounds, bounds. There is no formal
seed interval, equal-cost superiority claim or test-set generalization claim.

B1training prefixes10,961and10,679and H0original10,240controls now also verify
against complete raw bridge frames, actuator evidence and hazard reset geometry.
The B1frozen recovery has zero updates and identical model/statistics. A metadata
bug omitted its inherited unknown-tail flag; the old receipt is retained, the
combined ledger keeps uncertainty, and future preparers now propagate it.

Actual arc-catalog B3batch015226completes3/4with1,919controls and normal cleanup.
Its only environment difference from prior3/4B3is the catalog geometry. Road1
makes1.035mprogress then waits; better initial endpoint connections did not
solve the full road. No failed case is dropped. Fresh paired batch020918is now
training H/H0/B1with the new signature, equal nominal20,480-control budgets and
unchanged measured vehicle, strip obstacle, planner, timed trajectory and MPC.
It has no inherited weights or rule teacher; endpoints are still pending.
