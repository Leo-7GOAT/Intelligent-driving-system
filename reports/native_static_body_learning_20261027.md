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
One successful full static training episode has appeared in live H logs; frozen
endpoint performance is not yet established. The7,200s resource guard and
checkpoints bound this development batch.

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
