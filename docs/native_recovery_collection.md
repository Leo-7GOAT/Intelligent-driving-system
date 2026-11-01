# Native TRAIN recovery collection and rejected model

This is development evidence. Four new TRAIN roads are preselected from
`experiments/native_development_train36_validation24_v1.json`,indices4..7.
The24known validation conditions and future final conditions are excluded.

The completed batch `20261101T073000482834Z_native_TRAIN_recovery_collection`
uses the fixed seed22replay model4f205708... and its frozen normalizer.
A seeded Bernoulli teacher probability.5selects each10-control block. The
teacher keeps a2Hzreference and20HzMPCworkspace reading actual CARLA states;
the direct B1environment alone executes commands. It never integrates a
substitute plant. All four outcomes and all controls are retained.

Actual collection:1,612controls,850teacher/762learner,1,735world ticks including
resets,13.957s teacher-query compute,84.579s collection walltime. Mixed outcomes
are3successes/1timeout,zero collision/bounds. These are not learned-policy scores.
Every observed input,learner proposal,source choice and control frame is audited.
Policy tensors,PPOoptimizer,counters and RMS stay unchanged;zero PPO/BCupdates.

Only the850valid **actually executed** teacher commands enter the fixed next
fit. The762teacher proposals on learner-controlled frames remain diagnostic and
are excluded by `fit_train_mask`. Original TRAINdata contributes3,467physically
executed teacher rows;the29historical counterfactual rows are excluded from
this new fit. Sampling gives each original/new episode equal total probability,
including the timed-out episode. There are512separate actor Adam proposals,
not PPOupdates. Prelocked per-basic-group ceilings and backtracking retain
teacher TRAINfit;critic,PPOstate and normalization are unchanged.

The final model2f7287e9... is evaluated separately with direct `model.predict`,
no teacher or runtime override,on all four original basic TRAINroads:

| Case | Outcome | Actual controls |
|---|---|---:|
| 0 | success | 350 |
| 1 | deadline | 900 |
| 2 | success | 318 |
| 3 | out_of_bounds | 89 |

All1,657controls and four measured-start actions are verified. The unchanged
gate requires≥3/4success **and** zero collision/bounds. It fails. This model is
not promoted;the five-seed full-task gate remains closed. Lower TRAIN MSEdoes
not establish driving benefit. Total new cost3,269physical controls and512actor
Adam proposals,zero new PPO. The inherited PPOlineage is20,480controls.

[Portable result and hashes](../reports/native_B1_executed_recovery_collection_and_probe_20261101.json).
Raw assets and model checkpoints for this attempt are local;there is no claim
of public-download reconstruction. Both owned simulator lifecycles are terminal
and cleaned up normally. Do not repeat these completed physical controls.

The existing local evidence is rebuilt without a simulator:

```powershell
.venv/Scripts/python.exe scripts/analyze_native_executed_recovery.py --fit runs/B1_seed22_executed_native_recovery_fit_20261101 --output runs/rebuilt_native_recovery_report.json
```

The actual collection audit command used was:

```powershell
.venv/Scripts/python.exe scripts/audit_native_recovery_collection.py --batch runs/20261101T073000482834Z_native_TRAIN_recovery_collection --output runs/rebuilt_collection_audit.json
```

This reconstructs every recorded input and source assignment without a plant
step or new fit. Preserve old audit files;outputs refuse overwrite. New live
experiments require a separately registered protocol and bounded owned lifecycle,
rather than deleting registration pointers or rerunning completed batches.

```powershell
.venv/Scripts/python.exe -m pytest -q -m 'not carla'
```

The local full regression after this change passes1,091tests with1skip and
4CARLA exclusions in156.11s. CPU tests do not qualify the research result.
