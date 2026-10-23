# Native recovery and flat-policy stability — development evidence

Research acceptance remains incomplete. These are one-seed diagnoses on previously inspected development conditions, not independent confirmation or evidence of hierarchical superiority.

## Interrupted CARLA training and charged costs

The previous native batch `20261022T152120Z_native_road_learning` stopped without a final H model or B1 start. H has 94,329 readable control frames and a truncated gzip footer/tail. Its last complete optimization is at 94,079 controls; its latest retained model is at 30,080. The original logs are preserved without repair. The process-death cause is unknown; the machine's last boot predates this interruption.

H0 completed 100,160 controls / 103,631 training world ticks / 626 optimizer steps and exact reload; all four training-road probes time out. It is retained and not rerun.

`scripts/audit_interrupted_native.py` records readable-prefix counts, errors and checkpoint hashes. `scripts/prepare_native_recovery.py` registers one recovery child per parent and refuses duplicate preparation. Batch `20261023T010623Z_native_recovery` resumes H's learning state from 30,080, starts a new measured CARLA episode, and budgets 69,920 additional controls before a fresh B1 100k run. The old checkpoint lacks RNG state, so seed 11 is explicitly restarted; this is not seamless simulation continuation.

Every new 10k threshold now saves model, normalization, optimizer-containing SB3 state and RNG state in a partial directory before publishing a complete directory and latest pointer. Saved files have hashes. Retained model lineage and charged interaction costs are separate: H inherits at least 94,329 consumed controls, including 64,249 not represented in the restored model. The unflushed tail is unknown. This recovery is **not budget-matched superiority evidence**.

The new server and bridge use the existing CARLA 0.9.15 installation/client. Server ownership belongs to this recovery batch; never apply the previous batch's PID cleanup. At 09:25 local H had 49,580 new controls / 154 optimization phases and was OS-confirmed live. These are a timestamped snapshot, not final totals. The batch deadline is 11:06 local.

H subsequently completed 70,080 new controls /72,497training world ticks /219optimization phases /438optimizer steps, actor L2 change1.176118 and exact reload. Its retained model lineage is100,160controls, while charged cost is at least164,409controls. Four frozen training-road evaluations are all deadlines, with progress7.594/10.161/16.987/9.447metres. All73,680training-plus-evaluation control frames were matched to raw CARLA states/IMU/frame/time with no unaccounted controls (`H/frame_verification.json`). Evaluation adds3,600controls/3,724world ticks. B1 then actually started its fresh100kcontrols; do not call the full pair complete yet.

The recovery batch is now complete. Fresh B1 training executes **100,270
controls / 105,663 world ticks / 315 optimization phases / 630 optimizer
steps**, with exact reload and actor L2 change 1.879024. Training retains
111 out-of-bounds episodes, 61 deadlines, one success and one unfinished
episode. Its four deterministic training-road evaluations are **0/4**: two
out-of-bounds and two deadlines, consuming 2,217 controls / 2,341 world ticks.
The full analyzer matches **176,167 H/B1 training-plus-evaluation controls**
to raw CARLA states, frames, IMU timestamps and simulation clocks, with none
unaccounted. The figure was visually inspected. One stochastic training success
does not establish learned competence; recovery costs remain unequal.

Independent checkout at `d9f48cd28b892928f69822f3131c7263489db0a9` also passes
132 CPU tests with five conditional skips. Its live v3 batch
`20261023T023532Z_native_road_learning` executes H 640 controls/701 world ticks/
four optimizer steps and B1 952 controls/1,044 world ticks/six optimizer steps.
All **1,592 controls** are frame-verified; both models reload exactly. H has one
deadline and B1 two bounds failures, plus one unfinished episode each. This
checks independent source/runtime operation, not performance or another
independent training seed. The registered 640-control minimum can overshoot
when collecting full action rollouts; actual counts are retained. After both
bridges exited, an empty-server probe and process identity checks permitted
stopping the recovery batch's owned CARLA server (receipt in its infrastructure
directory). No CARLA server remains expected from these completed jobs.

## Completed paired learning-rate experiment

Batch `20261023T011009Z_update_stability` starts both branches from the same v19 checkpoint at 4,898,060 controls. Model, normalization, optimizer, environment and RNG states are identical; only the constant learning rate changes. Original-rate initial optimization records reproduce the parent continuation. All four endpoints were registered before execution. Counts below include actual rollout overshoot.

| Additional controls, original / lower | Original 3e-4 full task | Lower 1e-4 full task | 23m parking probes, original / lower |
|---|---:|---:|---:|
| 50,125 / 50,106 | 4/10 | 4/10 | 2/2 / 2/2 |
| 100,525 / 100,182 | 1/10 | 6/10 | 0/2 / 2/2 |
| 151,243 / 150,420 | 4/10 | 5/10 | 2/2 / 2/2 |
| 201,464 / 200,516 | 4/10 | 5/10 | 2/2 / 2/2 |

All 6m/12m/18m probes are 2/2 at every endpoint in both branches. Static-obstacle evaluation remains 0/2 at all eight endpoints. Approximate KL median/P95/max is 0.04109/0.07312/0.10506 versus 0.02800/0.04868/0.11167. Lower learning rate reduces typical observed changes, but does not bound KL or establish robust performance. The early-stop target is not a strict KL constraint.

The final lower-rate full-task result consists of both empty cases, both dynamic cases and one narrow case. The two static cases, both unseen cases and the other narrow case collide. All failures remain. During additional lower-rate training all 190 static episodes collide; original-rate static training has 161 collisions and one deadline. Parking improvement has not solved obstacle negotiation.

Rebuild every endpoint and verify source, initial checkpoint, evaluated model and case-list hashes:

```powershell
python scripts/analyze_update_stability.py --batch runs/20261023T011009Z_update_stability
```

Outputs are `analysis/summary.json` and `analysis/update_stability.png`; the plot has been visually inspected. This is one seed with correlated branches, not a seed-level confidence interval or retrospective final-model selection.

## Action-mean saturation diagnosis and registered intervention

The final lower-rate checkpoint's six static/narrow/unseen replays use frozen normalization. Removing obstacles from an unexecuted observation probe changes steering commands, ruling out complete input insensitivity. In all five collision episodes, however, the Gaussian longitudinal mean is outside the actuator range at every decision. On the two static cases its raw values range from 3.81 to 5.18 with mean standard deviation about 0.51; actual normalized acceleration is clipped to +1 throughout. Collisions occur at the configured 3 m/s speed limit. This supports an exploration/saturation hypothesis, not proof of causality.

`scripts/audit_obstacle_response.py` retains actual executed traces, raw Gaussian means/stds and unexecuted obstacle-removal probes. It restores the world before every real step and checks frozen normalization. Output: `obstacle_response_lower/results.json` within the completed rate experiment.

New batch `20261023T012208Z_bounded_mean_pair` branches from the **fixed final** lower-rate endpoint at 5,098,576 controls. It keeps all tensors, optimizer/RNG/environment state, reward, task distribution, action hold, variance and learning rate identical. One policy replaces the Gaussian mean mu by tanh(mu). This bounds the mean, not Gaussian samples: SB3 likelihoods, gSDE and PPO remain unchanged, and normal actuator clipping still applies. No planner, teacher, action mask or safety rule controls B1.

Milestones are zero, +100k, +300k and +600k controls in each branch, with all ten complete tasks and eight distance probes. The zero-update outcomes are 5/10 and 6/10: this immediate parameterization effect is explicitly **not learning gain**. Both retain all eight parking probes. Static cases are the primary diagnostic; no best endpoint will replace the registered endpoint sequence. At +100k controls the branches score 3/10 and 6/10; both static and unseen cases still collide, so there is not yet support for the primary obstacle-learning hypothesis. The worker ran under a one-hour deadline, resource checks and an identity-verified process-handle watchdog. All ancestral and discarded sibling costs remain in parent evidence; this is not from-scratch training.

The pair is now complete. At +300k both branches score5/10; at +600k both score6/10 (actual additions601,766and602,070controls). All eight parking probes pass at every endpoint. Static0/2and unseen0/2persist, so the main obstacle-learning hypothesis is **not supported**. Original/bounded KL median/P95/max is0.02873/0.05145/0.07702 versus0.01765/0.04771/0.06134. The analyzer verifies every checkpoint/case/source and the full figure was visually inspected; worker and guard exited normally. This optional policy is not adopted as a proven improvement.

Frozen static inference of the bounded300k model with gSDE resampling every1/4/10decisions gives0/8successes for both train-layout and validation-layout stochastic probes in every setting, all collisions at3m/s. Deterministic probes also all fail. These different inference modes of one model are not independent policies or completed training interventions.

Final bounded-policy replays show no clipping of the first obstacle's normalized coordinates. Gaussian acceleration means nevertheless remain0.9992–0.9999 on static cases, with preactivations3.94–5.11. Thus bounding alone did not restore anticipatory braking, and tanh's small tail derivative remains a specific concern. New batch `20261023T015449Z_softsign_mean_development` starts from the **same original5,098,576-control checkpoint** and changes only the mean transform to softsign. Original/tanh comparators are reused from the completed batch; neither is retrained. All tensors, optimizer/RNG/environment state, normalization, task/reward/PPO and learning rate are verified equal before the transform. Zero/+100k/+300k/+600k endpoints and the same18cases are registered, under a one-hour guard. Zero-update full task3/10and23m1/2are preserved as an immediate negative transfer effect. The worker is actually updating PPO (not just prepared); it had40,473newcontrols by09:56. Source, ancestor costs and comparator protocol hashes are saved. This remains a one-seed development experiment.

An audit bug was fixed before these explicit static probes: inherited route stages could silently replace the requested training-layout static scene with empty parking. Explicit diagnostic scenarios now disable reset curricula and assert the actual initialized scenario name. Default historical empty-distance probes are unaffected.

The softsign run subsequently completed all four endpoints: full-task success
**3/4/5/6 out of 10** at zero/+100k/+300k/+600k. Actual final addition is
600,276 controls (5,698,852 including its ancestor). Static and unseen cases
remain 0/2 each; the final six successes are empty, narrow and dynamic cases.
All eight distance probes pass after the zero-update endpoint, where 23m was
1/2. KL median/P95/max is 0.014302/0.036010/0.051638. The source/model/case
checks pass and the plot was visually inspected. Neither tanh nor softsign
solves the obstacle-learning failure; all three final branches reach 6/10.
The one-branch plot title now reflects its actual branch count, while preserving
the existing two-branch published figure output.

The subsequent, separately labeled demonstration-assisted diagnostic is in
[demonstration_diagnostic_20261023.md](demonstration_diagnostic_20261023.md).
It adds supervision and changes the direct-control frequency, so it is not a
single-factor continuation of these PPO-only branches.

## Offline native proposal coverage

Recovery rotates training-route order, so candidate audits now match recorded geometry IDs to evaluation protocol order instead of taking the first four training scenes by index. Regression tests cover the rotated order and missing geometry.

At the eight recorded initial/terminal states, a new **offline-only** route-tangent catalog proposes the same31wait/pose/speed choices from known centreline arc length and heading. It projects the state onto the reference only to choose targets; measured state remains untouched. It is not enabled in an environment or any training job. Direct analytic witness counts at initial states are18/4/22/12, versus10/4/14/10 for fixed-final-heading candidates and24/14/26/22 for original ego arcs. Terminal counts are0/8/0/0 for route and fixed-final-heading catalogs, versus16/14/16/14for ego arcs. A witness to an arbitrary ego arc does not mean that it approaches or can subsequently park at the task goal. Analytic rejection does not prove no Hybrid A* path.

This candidate proposal alone does not repair the failed terminal states and is not adopted as a proven improvement. Any later route-conditioned experiment must expose comparable route information to H/H0/B1/B2/B3 and disclose the map/route prior. Exact proposals, geometry IDs and recorded states are in `H/route_candidate_comparison.json`; no CARLA tick or Python vehicle integration occurs during this audit.

## Verification and remaining limits

Full CPU regression after all policy/diagnostic changes: **130 passed / 5 conditional skips**, including the dedicated Windows retained-handle test. Tests cover true optimizer updates, Gaussian/gSDE log probabilities at out-of-range sampled actions, exact parameter/optimizer transfer, save/reload, native checkpoint RNG/hashes, duplicate-recovery prevention, explicit scene identity and route-proposal invariants. Optional live tests skipped here are not counted as CARLA validation.

The completed learning-rate evidence is now published separately as `paired_ppo_rate_stability.zip` (67,837,038 bytes,256manifestfiles,SHA256`b284991976fe7ad87405666a548f837b6561bdcc84b3f674e276a7213bc0fb71`). A fresh download verifies every file; the independent checkout loads all eight endpoint models with frozen normalization and rebuilds a byte-identical plot and identical endpoint results. Sourcee81b378is pushed/remote-verified. The older tag is unchanged; use this newer source for the new analysis command. The bounded-mean and native-recovery models are not in that asset.

Two additional assets are now actually uploaded to the same immutable-tag
prerelease, freshly downloaded and verified:

| Asset | Bytes | Verified manifest files | SHA256 |
|---|---:|---:|---|
| bounded_mean_stability.zip | 185,786,166 | 255 | `e3dba9ce5f6346a8c4710236a2dbe24900e1554109263eb9faced7e659d4628a` |
| softsign_mean_stability.zip | 92,312,679 | 218 | `cb57defabc169472ad36de1fc37b89c6a5f319154829a95179a20be0b5eb948f` |

Independent checkout at `10fff2c48689e90ed376275d6b3e59f18fe8bfd3` reloads all
12 downloaded endpoint models with frozen normalization and reproduces identical
scientific rows and byte-identical figures. Receipts are
`runs/bounded_download_reproduction.json` and
`runs/softsign_download_reproduction.json` inside that checkout. Inference model
and normalizer bytes are unchanged; portable JSON metadata and omissions have
provenance. These are not portable training-resume bundles. All endpoints and
failures are retained. The native recovery and demonstration models remain
outside these assets. The release notes identify the required newer source;
the historical tag has not been moved.

The independent-checkout live v3 check has completed as recorded above. Five independent policies, healthy full-task B1, native static/dynamic competence, feedback and strong-B3 contributions, comfort improvements and fresh 120-condition confirmation remain required.
