# Independent final audit and LIVE matched continuation — 2026-10-23 14:06 local

This goal turn is progress: optional frozen training normalization implemented,
tested with actual PPO,100kcontrast finished, allrawdataaudited, independently
reproduced and sourcepublished; matched200k/300kcontinuation actually started.
No hard blocker. Full objective remains active/incomplete. No agents/CARLA or
other projects were used. Read the complete objective attachment on continuation.

Source917d29e pushed. Clean independent pinned checkout at917d29e passes139tests,
5skips; rebuilds complete frozenaudit, both terminalaudits, comparisonJSON/PNG
byte-for-byte. Receipt clone/runs/frozen_normalization_complete_reproduction.json.
AuditSHA0dff2aa898279121261ebb95e366eccd0469e7e93cba6609e82c757378555777,
figureSHA0257082c2c8dddc4690a73df0bd69a8fa3bae5a0ec3d302d5f4162d57363f399.
New assets remain local, not uploaded. Earlier source-specific receipts retained.

NEW LIVE `runs/20261023T060219Z_normalization_retention_pair`, pointer
`runs/normalization_retention_pair.json`. Worker50724 creation14:02:36.755,
launcher36860; guard55448 creation14:02:39.059, launcher45680. Both OS-confirmed
at14:06. Deadline15:32:19, one CPU worker,12GiBdisk/4GiBRAM/85C guards,0retries.
Initial43.28GBdisk/38.05GBRAM/GPU38C. Same-checkpoint resumes of BOTH completed
reduced-noise branches: updating_stats then frozen_stats at200k, then sameorder
at300k. Inherited102400PPOcontrolsperbranch; weights/optimizer/RNG/env/statistics
files byteidentical to their respective100kendpoints. Frozen branch retains
freeze_observation_stats=true; updating branch retainsfalse/default. Same task,
cases,reward,LR,seed,lowlevel. No newteacher/BC. Updatingbranchat146227controls,
lastphase57/3531totaloptimizersteps (17newphases/824newsteps). Frozenbranch has
not started newcontrols yet. Do not interpret this snapshot as permanent status.

Scripts: continue_normalization_pair.py prepares only after the complete independent
receipt; paired_update_stability.py now supports branch-specific PPO options.
New verify_normalization_continuation.py --batch <batch> --partial verifies only
completed endpoints/closed transition chunks, starting counters/episodeclock from
the inherited snapshot. At launch noendpointcomplete, so its auditedNEWcontrols
are0 despite live training progress. Omit--partial after full completion and
normalguardexit. It checks rawactionmapping, clocks, terminalmetrics, actorupdates,
optimizerexposure and frozen/updated normalization. Generic analyzer available:
scripts/analyze_update_stability.py --batch <batch> (complete only, plus plots).
New integrationtest uses actual resumedPPO/plant/evaluation and rejects altered
executedcommands. Rootfullregression completed140passed5skips2existingwarnings
(runs/normalization_retention_pair_regression.xml). Latest changes will have a
later Gitcommit; inspectstatus.

Next: preserve live worker/frozen source; inspect endpoint200k, auditrawdata and
continue matched300k. Final100kresult4/10parking8/8 is retention evidence only;
static/narrow/unseenremain0/2each. No architecture/five-seed/CARLA-final gate met.
Native-road candidate architecture was only inspected this turn, not modified or
trained. Do not claim new CARLA work or an unstarted architecture trial.

# Completed frozen-statistics result — 2026-10-23 13:57 local

`20261023T054347Z_frozen_training_normalization` COMPLETE, normal guard exit0;
worker24572/guard49316 are absent. Full audit checks102,400training controls,
33,455eval controls,40phases/2,332optimizersteps; all13complete checkpoints retain
exact means/variances/count8113.0001. Endpoints full4/10,0/10,4/10; parking5/8,
2/8,8/8. Final empty/dynamicboth succeed; static/narrow/unseenall deadlines.
Training152success46deadline37collision retained; no final-eval collision/bounds.
ActorL2 from branchinitial .665943; finalmodelSHA
9f7b8105f21e6d20206d4d60ea47d94070cbf2a639ecea8f9b57fb7ec508519f.
This restores overall initial4/10 while improving parking, not complete baseline
competence. The updating-statistics reduced-noise comparator remains0/10,0/8.

New terminal audits preserve all18cases: at25k allnine deadlines are stationary
and braking for last5s; at100k remaining failures end8.68–19.04m away. Existing
audit_terminal_failures.py now handles branch protocols and --output. New
plot_normalization_contrast.py produces all9rows and a visually inspected figure
in batch/comparison; no confidence intervals for one correlated lineage.
Analyzer checks allcomplete periodic normalization checkpoints in addition to
endpoint rawcontrols. Historical partialaudit replaced by complete derivative.

Sourceeee6c8a pushed/remoteverified. Clone sourceeee6c8a is clean and passes
139tests5skips2warnings; completed reduced-noiseaudit and fixed438visitprobe
byte-identical. Receipt clone/runs/frozen_normalization_initial_reproduction.json.
Final frozenaudit/figure independent reconstruction is next, followed by a paired
unchanged continuation of BOTH completed reduced-noise branches to200k/300k.
No continuation has been prepared or started at this paragraph's timestamp.
Use current OS/protocol files for later state. Full goal still active/incomplete.

# Frozen training normalization contrast — 2026-10-23 13:47 local

Previous user-requested summary turn was a status-only turn, not development
progress. This continuation makes code/test/actual-training progress. Full goal
remains active and incomplete, without a hard blocker. Read the full objective
attachment before continuing. No subagents, other projects or CARLA were used.

Reduced exploration `runs/20261023T052413Z_reduced_exploration` is COMPLETE,
normal guard exit0, old worker45768/guard33204 absent. Full raw audit:
102,400 training controls, 2,707 optimizer steps, 40 phases; endpoints4/10,
2/10,0/10; final6collisions4deadlines; all parking fails;33,709eval controls.
Corrected final PPO-only actorL2=.741360; inherited ledger22.607050 includes
manual log_std shift. `exploration_verification.json` is authoritative.

NEW LIVE batch `runs/20261023T054347Z_frozen_training_normalization`, pointer
`runs/frozen_training_normalization.json`. Actual worker24572 creation13:44:09.501,
launcher39616; guard49316 creation13:44:11.782, launcher19760. Revalidate OS
identity before any action. Frozen source runs paired_update_stability.py.
Deadline14:43:48; one CPU worker,12GiB disk/4GiB RAM/85C guards,zero retries.
All initial checkpoint files identical to reduced-exploration INITIAL checkpoint,
not its learned endpoint; only option freeze_observation_stats=true is added.
Same log_std=-4, LR1e-4, seed, reward, cases,0/25k/100k milestones. Inherited
supervision/control costs retained. Training still updates actor and critic.
Zero4/10 and10,882control/state pairs match the completed control EXACTLY.
Partial normalization_verification.json passes. First two phases give5,120
controls/80optimizersteps; do not use this historical count as live status.

Fixed438train-visit inference is now reproducible through
scripts/measure_normalization_sensitivity.py. Statistics-only longitudinal
action medianchange.389393,P951.341785;87.21%change>.1. This supports testing
training-time freezing, not claiming a cause. It is not an on-policy sample.
Output `runs/20261023T050438Z_aggregated_ppo/verified_normalization_sensitivity.json`.

Code: optional freeze in navstack/research.py rejects fresh/unfitted runs and
reward normalization, preserving default behavior. New actual-PPO tests prove
unchanged moments, actor updates and exact split/uninterrupted resume. Full
root regression139passed5skips2existingwarnings. Preparation saves all source,
training_source.diff and exact initialization hashes. Analyzer now supports both
contrasts, exact frozen moments/zero-trace equality and --output for independent
derivatives. Old exploration outputs remain compatible.

Next: inspect live process/guard/progress; audit completed endpoints (use --partial
until completed.json AND normal guard_final); independently reproduce final
exploration audit and normalization probe, test source in pinned clean checkout,
update outcomes and publish source after preflight. Do not launch a duplicate or
edit frozen sources. Full research gates/five seeds/fresh CARLA confirmation are
still unmet; these diagnostics are not final architecture evidence.

# Verified source and partial exploration result — 2026-10-23 13:31 local

Code25df90c6d5297d5f860bf9c173bb60fb35adfa5e pushed/remotematched. Root/clone
137passed/5skips/2existingwarnings. Cloneclean/detached25df90c usingpinnedsibling
environment. Receipt runs/exploration_diagnosis_reproduction.json: completedPPO
summary/figure, factorialverification and438-visitnoiseprobe allbyteidentical.
PPOsummarySHA22529b83ac5dfba7e547ab01c026ec3509acf90ddf56f0ff976eda8baa4baba3;
factorialSHA5a1eb1969fd96e075e12d813bc50f75b589cb7b47eff811c09524f587bda6d22.
Newartifacts localonly. No repeatedbaselinetraining or diagonal evaluation.

Same reducedexplorationworker45768/guard33204 livehealthy, at13:31heartbeat
58,457controls/22phases. New25kmilestone:25,600controls/744optimizersteps,
2/10(emptyboth),5collisions3deadlines;parking0/0/0/2,11,041evalcontrols.
Originalnoise25k0/10 with3collisions7deadlines. Do not claimoverallimprovement.
New scripts/analyze_exploration_branch.py --batch runs/20261023T052413Z_reduced_exploration --partial
passes zero/25krawcommands/clocks/terminal/model/source and parameterledger checks.
Oncompletion omitpartial to additionallyverifyallrawtrainingcontrols/updates.

IMPORTANT accounting: actor_vector includeslog_std. Copiedresume.pkl keeps
originalpre-interventioninitial_actor forbyteidenticalRNG/envledger preservation.
Its actor_l2_change=22.627417at0 and22.576128at25k includesmanualnoisechange.
Do NOT callthose PPOlearnedchanges. Newanalyzer compareschangedinitialcheckpoint:
PPO-onlyL2=0at0,.410725at25k; verifiesexactrelationshiptoinheritedledger. This is
anaccountingclarification, notachange to runningtraining; preserve frozenfiles.
Partial_endpoint_verification.json retainsfirstrawcheck; authoritative corrected
namedmetrics arein exploration_verification.json. No final100kresultyet.
Newanalyzer/reportsource publicationfollows; fullgoalACTIVE/incomplete/nohardblock.

# Audited collapse and exploration contrast — 2026-10-23 13:27 local

Goal ACTIVE/incomplete, nohardblock; previous turn substantialprogress. This turn
completed originalPPO, actual72episode factorial, allrawverification and started
the single-variable exploration contrast. No agents/otherprojects/CARLA started.

Original `20261023T050438Z_aggregated_ppo` COMPLETE normalexit0 at13:22local;
worker27336/guard40952 gone. Final102,400controls/40phases/16,000optimizersteps,
actorL2change3.253203, exactreload. Fulltask0/10 alldeadlines; allparking0/2.
Training102deadlines25collisions4success2bounds pluspartial; all102,400train and
41,325evalcontrols checked including policyaction->requestedcommand mapping.
Finalplotvisuallyinspected. Zero4/10 and25k0/10 retained.

`normalization_25k_factorial` under thatbatch COMPLETE normalexit0; worker42900
andguard15228 gone. Actual47,507controls/72episodes,0updates/queries. Oldweights
withold/newstats both4/10; newweights withold/newstats both0/10. Parking vectors
0/2/1/2,2/2/2/2,0/0/0/0,0/0/0/0. Allrawchecked by new
scripts/verify_normalization_factorial.py; unchangedpairs25,125actions/states
exactly reproduce originalevaluation. Sourcecopy/hash in audit_source.py and
protocol; verification.json. Counterfactual swaps not promoted as models.

New scripts/measure_exploration_scale.py produces verified_exploration_scale.json:
438fixed train-mask visits, actual initialconditionalstd median1.02429 (not the
logged noiseparameterexp(-2)=.1353), boundsprobability36.18%/32.98%. At25kmedian
.9700/.9715. Noextra controls/queries/updates. Diagnosticnoton-policyfrequency.
Initial quickread used locale decoding and failed checkpointlookup before any
interaction; corrected with UTF8. Structured reproducible script output is the
authoritative measurement, not the first ad-hoc copy.

NEW LIVE `runs/20261023T052413Z_reduced_exploration`, pointer reduced_exploration.json.
Actualworker45768 created13:24:44.344local, launcher25928; guard33204 created
13:24:46.558, launcher53676. OSidentities/healthyguard verified. Deadline14:24:16,
oneCPUworker,12GiBdisk/4GiBRAM/85Cstop,0automaticretries. Runfrozen
source/scripts/paired_update_stability.py --batch (never editfrozen source).
Only initiallog_std tensor-2->-4, remains trainable; exactotherparam/optimizer/
RNG/env/normalization checkpoint preservation. Uses ORIGINALzero-update model,
not the failed100k endpoint. Parentbaselinecost102400 andallancestralsupervision
charged. Sourceprep refuses incompleteparent/factorial audits. SameLR1e-4/KL.03/
reward/cases; same0/25k/100kmilestones. Zeroeval4/10 parking0/2/1/2; independent
reload and rawcomparison in zero_update_exact_verification.json confirms10,882
actions/states exactlyidentical, onlylog_std differs. At13:26training7680controls,
363optimizersteps/3phases (KLearlystopping); actualcosts must be reported, not
assumed equalupdates. No learning endpoint yet. Continue existingworker.

Rootregression137passed/5skips/2existingwarnings in exploration_branch_regression.xml.
Preflight327files/no findings. CorrectGitHuborigin/accountverified. Publication
andindependentchecksfollow; inspectlatestGit/receipts. Scripts changed:
audit_normalization_transfer.py supportsnewprotocol/all18cases/resourcebudget;
verify_normalization_factorial.py raw/diagonal audit; measure_exploration_scale.py;
prepare_exploration_branch.py; paired_update_stability zeroresultguard;
watch_existing_pair allowlistaudit; analyzer trainingaction checks and PPO-only plot.
Do not rerunbaseline/factorial; allresearchacceptancegates remain unmet.

# PPO regression retained — 2026-10-23 13:12 local

Same liveworker27336/guard40952 OSidentity-confirmed healthy. Currentheartbeat
36,094controls/14optimizationphases. Fixed25k milestone actually25,600controls,
4,000optimizersteps,actorL2change1.665761: fulltask0/10 (7deadlines,3collisions),
parking0/0/0/0,14,243evalcontrols. Zero-update4/10 retained. Partialanalyzer
passes both endpoints; do not substitute the stronger supervised checkpoint.
Same100ktarget/deadline14:04:38. No duplicate or newteacher collection.

Inspected scripts/audit_normalization_transfer.py for possible next diagnostic:
it already crosses policy/normalizer using frozen parent dynamics and records
actual physical traces, but currently accepts jobs or originalwarmstart only,
not this ppo_milestones protocol. It also lacks a bounded resource heartbeat.
Do not claim a factorial was run here; only the old warmstart factorial exists.
If newdiagnosis is needed, add explicit protocol support and bounds, then cross
zero/25k (or fixed100k after completion) without promoting swapped pairs as
evaluation models. Rootcode6d351d5 and doc03e6f34 verified pushed; latestdocs may
advance. This turn is substantialprogress, fullgoalACTIVE/incomplete/nohardblock.

# Verified publication/reproduction — 2026-10-23 13:10 local

Code6d351d586e43577bf44ef9c5e2d51c587515ded4 pushed/remotematched on correctorigin,
accountLeo-7GOAT verified. Root and clone137passed/5skips/2existingwarnings.
Clone is detached at6d351d5, clean, using siblingclean_checkout pinned .venv.
Its completed aggregation summary/figure are byteidentical; summarySHA
2a7f8039b1a60c78b09db33e4cd74b56e4e8f543611895ec9721ade4bcd9865b.
Both supervised models give64repeatable finite predictions with frozenstats.
Receipt runs/pure_learner_aggregation_reproduction.json in clone. Newassetslocal.

Same actualPPO worker27336/guard40952 OSverifiedalivehealthy at13:09.
25,600controls/10phases/4,000optimizerupdates completed; firstposttraining
evaluation inprogress, no result yet at this entry. The target0 rawanalyzer
also passed. Continue this live batch, no duplicate. Sourcefrozenatpreparation
and not altered. FullobjectiveACTIVE/incomplete, nohardblock.

# Completed aggregation and actual PPO continuation — 2026-10-23 13:07 local

Goal ACTIVE/incomplete; no hard blocker. Previous summary turn was verified wait.
This turn completes two-round raw auditing, strengthens independently reloaded
prediction checks, and starts actual PPO from the fixed final supervised model.
All main research acceptance gates remain unchanged and unmet.

Aggregation043924Z COMPLETE, worker52636/guard44820 exited normally at13:02.
28,586new controls/queries,0teacher executions,67,200supervised optimizer steps,
0PPO. Round1:18,669controls,10/32collection success,30,800updates; round2:
9,917controls,23/32collection success,36,400updates. Both development4/10;
round1threecollisions/threedeadlines, round2sixdeadlines with zero terminal speed.
Final model a22756645fe203b46c827d573063ff26dae41973b93f5e33ef30147e67dff318.
Full strengthened analyzer checks all28,586predictions against preceding models,
all21,644evalcommands/clocks, masks, frozenstats/critic/logstd, actualactorchanges
and normalguardexit. Larger batched prediction roundoff is resolved by exact
single-row inference, not looser tolerance. Figure visually inspected. Earlier
mixture batch also passes stronger checks in strengthened_analysis; its original
analysis summary/figure hashes restored exactly to source-specific receipts.

NEW LIVE `runs/20261023T050438Z_aggregated_ppo`, pointer runs/aggregated_ppo.json.
Actualworker27336 created13:05:08.306local, launcher54292; guard40952 created
13:05:10.628local, launcher51964. OS identities and healthyguard verified.
Deadline14:04:38local;1CPUworker,12GiBdisk/4GiBRAM/85Cstop, noautomaticretries.
Frozen source must not be edited. Same finalround2 model, originalmodelseed11
(parent supervised permutation seeds do not constitute independent policies).
No collection/BC/teacher in this batch. DeclaredPPO milestones0/25k/100k;
retain all endpoints. Allancestralsupervisioncosts charged. Zero-update endpoint
exactly matches parent4/10,parking0/2/1/2 and10,882controls. At13:07actualPPO
7,680controls,3optimizationphases,1,200optimizersteps with nonzeroactorchanges.
No new endpoint yet. No CARLA process started in this turn.

New scripts/train_aggregated_ppo.py requires complete parent audit and checks
inherited PPO hyperparameters/seed, freezes sources and registers this continuation.
Genericrunner now checks zero-update reproduction before any PPO interaction.
Analyzer supports --output for preserving old derivative versions and target0.
Root full regression137passed/5skips/2existingwarnings in
runs/aggregated_ppo_regression.xml; publication preflight324files/no findings.
Source publication and independent reconstruction follow; inspect git/receipts
for latest source and current worker before another action. Next: audit25k then
100k endpoints, update failure diagnosis from actual results. Never repeat parent
collection or silently select round1 instead of the final endpoint.

# Verified source/reproduction — 2026-10-23 12:47 local

Source f49ee75c0ac378e17cf5179f585bbc6a510f58c4 pushed/remotematched;
accountLeo-7GOAT and correctorigin verified, publication323files/no findings.
Root and independent pinned environment both137passed/5skips/2existingwarnings.
Clone receipt runs/actor_label_contrast_reproduction.json byte-identical verified
summary SHA9ae6d701643f9474fdf55209076d8c6738d10d877ef3198d8a946ec1276471c0,
both model hashes and18,541evalcontrols checked. Full clone test receipt
runs/current_state_aggregation_reproduction_tests.xml. New artifacts local only.

Live aggregation043924Z at12:46round1case17/control41; worker52636 and guard44820
OS-confirmed alive/healthy. First17cases complete, failures retained, no new
supervised update yet. Same14:39:25deadline. No other train/CARLA processes
expected. Root/clone clean before this documentation update. Continue this
actual batch; do not repeat relabel/label fitting. Full objective ACTIVE.

# Active learning continuation — 2026-10-23 12:43 local

Goal ACTIVE/incomplete, no hard blocker. Previous turn was progress; this turn
completed actual relabeling, fixed-label fitting/evaluation and started the next
coverage diagnostic. All main H/H0/B1/B2/B3, five-seed, CARLA and fresh-confirmation
acceptance gates remain unchanged. No agents or other-project work.

Relabel033822Z COMPLETE:32,538actualcontrols/queries,1,371.404queryseconds,
all96cases, every non-action array identical. Root and independent pinned
checkout verifier pass all raw/source/data/label/clock/cost checks. Worker21256,
guard36560 and helper19104 exited; no restart. First helper failure retained.

Current-state actor fit042840Z COMPLETE:200epochs/19,400supervisedupdates,
actorL2change28.81247, unchangedcritic/logstd/norm and0PPO. Pure development1/10
(one dynamic success,6deadlines,3collisions), parking1/2/0/0 out of2. Control
031926Z was2/10,parking0/0/1/1. All10,397new evaluation controls and8,144control
controls verified by scripts/analyze_actor_label_contrast.py. Source/data/init
matching checks passed. Worker11204/guard23784normalexit0; both stopped. This
negative label-only result is retained, not promoted as a solution.

NEW LIVE batch `20261023T043924Z_current_state_aggregation`, pointer
runs/current_state_aggregation.json. Worker52636 created12:39:52, launcher42992;
guard44820 created12:39:56, launcher42048. OS and actual collection heartbeat
verified. Deadline14:39:25local, one CPUworker,12GiBdisk/4GiBRAM/85Cstop,
zeroautomaticretries. Frozen source never edit. Two rounds of seeds8401..8408
x four scenes, first6fit/last2heldout; teacherexecution probability0 and fresh
current-state queries. Starts fixed newly fitted1/10model, no best checkpoint
substitution. Carries all32,538relabeled visits,24,831train/7,707heldout. Excludes
613-control diagnostic data. Each round200actor-onlyepochs, stats/critic/logstd
frozen, all18endpointcases retained. NO PPO stage in this batch. This is a
coverage-oriented continuation combining new visits and additional optimization,
not a single-factor causal comparison. Broader study costs retained separately.

Generic recovery runner/analyzer now support current-state labels, actor-only
fitting and an explicit empty PPO milestone list, with legacy defaults intact.
Full regression137passed/5skips/2existingwarnings in pure_learner_aggregation_regression.xml;
old two-round+102400PPO raw analyzer also passes. New actual batch preparation
asserts data/source/checkpoint hashes and original masks. Code/report publication
and independent contrast reproduction follow; inspect Git/receipts for final
state. No new model/data release asset uploaded. Next action is monitor this
specific live batch and verify all retained endpoints; use its results for the
next learning intervention rather than repeating completed label fitting.

# Live preparation transition — 2026-10-23 12:20 local

Goal ACTIVE/incomplete, no hard blocker. This continuation verifies two completed
relabel groups (21,688controls/queries,932.882teacherqueryseconds), strengthens
single-factor preparation, and registers a bounded follow-up. Prior turn was
progress: actual613-control diagnostic plus reproducible raw evidence.

Same relabel worker21256/guard36560 alive; at12:19total26,765controls,
round2case13/control233. No duplicate/restart. Original deadline13:38:23local.
Full verification and fitting have NOT occurred yet.

NEW LIVE helper19104, created12:19:19.589local, command is current PowerShell7
running batch/prepare_followup.ps1. It waits on identity-checked worker21256;
requires normal guard exit and completed.json, checks pinned source hashes,
then calls fit_recovery_actor.py --parent ...030731... --relabel ...033822...
--control ...031926... . That entry performs full raw verification before
preparing the fixed-label contrast. It DOES NOT LAUNCH TRAINING. Revalidate
helper OS identity; inspect fit_preparation_completed.json or failure.json
before manually preparing anything. Do not duplicate this pending preparation.
Do not edit scripts/fit_recovery_actor.py, scripts/verify_relabel_teacher.py or
navstack/imitation.py until helper exits; it rejects changed source hashes.

First helper49464 exited before waiting/fitting because Windows PowerShell
could not load Get-FileHash with inherited module paths. Failure and empty
marker preserved in followup_attempt1_failure. Corrected launch uses the same
PowerShell7 runtime as normal terminal commands; source hashes and live OS
process checked. This was a diagnosed helper repair, not a relabel retry.

When prepared, pointer runs/current_state_actor_fit.json identifies its frozen
batch. Launch source/scripts/fit_recovery_actor.py --batch under existing
watch_existing_pair.py identity guard, then verify actual updates and outcomes.
Budget30min starts at preparation; no retries or new PPO. Source guard now also
checks control parent/data hashes and identical fit_demonstrations AST. AST
comparison against actual control source passed and script compiles. Existing
137-test full regression remains the last full run; these guard assertions
will be exercised by actual full-data preparation. Source/attribution docs are
being published; inspect Git. DAgger idea now attributed to Ross et al.2011,
without importing its theoretical guarantees or calling supervision pure PPO.

# Active evidence snapshot — 2026-10-23 12:07 local

Goal remains ACTIVE/incomplete, no blocker. Previous turn was progress (source,
raw audit, independent reproduction); this turn adds actual pure-learner queries
and partial-data verification. No agents/CARLA/new policy fitting started.

Relabel worker21256/guard36560 remain live, created11:39:12/11:39:14. At12:06:
16,888actual controls, round1case17/control303. Original32cases complete:
10,817controls/queries,522.402teacherqueryseconds, all non-action arrays identical.
scripts/verify_relabel_teacher.py --partial checks only immutable completed
groups and writes partial_verification.json. Default still rejects incomplete
batches, including from fit_recovery_actor.py. Frozen live sources unchanged.
Deadline13:38:23local; inspect OS and progress before next action, no restart.

New actual audit COMPLETE in actor-only batch/learner_state_label_audit:
worker55284 exitednormally (exec50247 exit0). Cases static train8101/8107,
validation7301/7302 all collide137/164/159/153controls; total613controls/queries,
zero teacher-executed controls and zero updates. Independent raw verification
matches all312validationcontrols' actions/state exactly against original eval.
Heldout8107diagnostic labels MUST NOT enter training. SteeringMSE on self-visited
states .00658/.01296/.00654/.00614 vs original staticheldoutvisits .000655 with
current-state labels. This supports coverage diagnosis, not causality or success.
Path/action figure learner_distribution_shift.png regenerated and inspected.
Full CPU tests137passed/5skips/2existingwarnings in partial_relabel_regression.xml.

New scripts diagnose_action_labels.py, audit_learner_state_labels.py,
verify_learner_label_audit.py, plot_learner_label_audit.py plus partial verifier
and test changes are being published; inspect Git for final state. After full
relabel verification use the unchanged single-factor fit command below; still
NOT prepared/started. Choose further data/learning intervention from its real
result, preserving all old failures and fair supervision costs.

# Verified publication/reproduction — 2026-10-23 11:54 local

Source ff6b5782b99d81e2562397f161ff7d01c9746ba0 pushed and remote hash matched.
Both root and independent pinned-environment checkout pass136tests/5skips/
2existingwarnings. Clone receipt runs/recovery_aggregation_reproduction.json
loads five local models with frozen stats and reconstructs summary/figure with
byte-identical hashes. Clone test receipt runs/recovery_aggregation_reproduction_tests.xml.
These newer diagnostic artifacts are LOCAL, not uploaded assets. Root and clone
were clean before this documentation-only update. Source/full research unchanged.

Relabel worker21256/guard36560 OS-verified live11:53, original cases0..24complete,
observations identical. Revalidate and continue the same job, no duplicate.
Current-state actor fit still NOT prepared/started. Next command and fixed
comparison constraints are below. This goal turn is progress; no hard blocker.

# Active continuation snapshot — 2026-10-23 11:52 local

Goal remains ACTIVE/incomplete. Previous user-summary turn was a verified wait:
OS worker21256 and guard36560 were confirmed alive. This continuation adds full
raw PPO audit, completed-results reporting, a relabel verifier and regression.
No agents, no hard blocker. No CARLA server is expected; do not reuse old PIDs.

Recovery aggregation `20261023T030731Z_recovery_demonstrations` COMPLETE:
21,721 collection controls/queries, 32,200 extra supervised optimizer steps.
Mixtures32/32 and27/32 are NOT learned-policy scores. Pure learned endpoints:
BCround1=4/10, BCround2=1/10, PPO25,600=0/10, PPO102,400=4/10. Final PPO has
16,000 optimizer steps, actor L2 change2.88833; every direct training control,
episode clock, evaluation trace and terminal metric checked by analyzer. Figure
recreated and inspected. Raw training:65deadlines/72collisions/23successes.
Actor-only control `20261023T031926Z_recovery_actor_only` COMPLETE:19,400
supervised updates,2/10,parking0/0/1/1,8,144evalcontrols; critic and stats unchanged.
This is not a solution to B1 competence. Reports now retain all final negatives.

LIVE `20261023T033822Z_current_state_relabel`, pointer runs/current_state_relabel.json:
worker21256 created11:39:12, guard36560 created11:39:14, both OS-confirmed live
11:50. Deadline13:38:23local. No autoretry. It physically replays96retained cases
(32,538 controls) and queries fresh current-state labels; original commands only
execute. Non-action arrays remain identical. At11:50original cases0..18complete,
all observation differences0. Revalidate process identity/progress before action.
Do not edit its frozen source or restart due to an observation timeout.

Fresh-teacher history audit:3,120 replay controls/queries, exact states; within
decision-block labels differ by up to2.0longitudinal. Current-state teacher actual
competence audit:4/4training cases,1,341controls; not a learned-policy result.

Next transition after successful relabel completion (NOT started yet):
`python scripts/fit_recovery_actor.py --parent runs/20261023T030731Z_recovery_demonstrations --relabel runs/20261023T033822Z_current_state_relabel --control runs/20261023T031926Z_recovery_actor_only`
This calls full scripts/verify_relabel_teacher.py first, creates pointer
runs/current_state_actor_fit.json, freezes sources and validates all non-action
arrays/initialization. Launch the new batch's frozen fit_recovery_actor.py with
--batch under the existing identity-matched watchdog; 30min,12GiBdisk/4GiBRAM,
zero autoretry. Same seed13,200epochs,19,400actor-onlysteps; only labels change.
No PPO in this contrast. Use its actual result for the next development choice.

Current full CPU regression136passed/5skips/2existingSB3warnings:
runs/relabel_verifier_regression.xml. Publication scan317files/no findings.
AccountLeo-7GOAT, remoteIntelligent-driving-system verified. Source publication
and independent-checkout verification follow this snapshot; inspect Git/receipts
for final hashes rather than assuming completed. No new diagnostic data/model
release assets have been uploaded. Previously published bounded/softsign assets
remain verified as below. The full original research gates are unchanged.

# Historical verified snapshot — 2026-10-23 11:01 local

Source cc76677dc5c2f36245758e96565bde3b1734ecb0 pushed/remote-verified.
Independent checkout at that exact source also passes 133 tests / five skips /
two existing warnings (runs/recovery_teacher_reproduction_tests.xml). Root and
clone are clean before this documentation-only update. No training, CARLA,
upload or download process remains live. All results/publication receipts and
next concrete recovery-state data work are described below. Full goal ACTIVE,
incomplete; this turn is progress with no hard blocker. A documentation-only
commit follows; inspect Git for its final hash.

# Verified continuation snapshot — 2026-10-23 10:59 local

Supersedes pending transfer states below. Both mean-transform assets uploaded
successfully, actual downloads verify all 473 manifest files and archive SHA256s.
Independent checkout at 10fff2c loads all 12 endpoint models with frozen stats,
identical scientific rows and byte-identical plots. Receipts inside clone:
runs/bounded_download_reproduction.json and runs/softsign_download_reproduction.json.
Downloads/extracted assets/verification receipts are under
runs/prerelease_20261022/mean_download. Prerelease body updated with exact hashes,
source requirement and negative scope; tag unchanged. No upload/download remains
running. Native recovery and BC/PPO artifacts are still local, not these assets.

New `navstack.imitation.RecoveryTeacher` implemented as explicitly invoked
training-data oracle, NOT used during policy inference. It reads the actual
learner state/world in a private planner/MPC workspace, returns labels and
query/plan/controller costs, and does not advance physics or mutate learner
state/episode metrics. Repeated unexecuted-control queries and stale reset
workspaces reject. New actual-plant test reproduces complete B3 6m parking via
separate direct controls within1e-4 all state fields. The first test used a
wall-clock planner and was nondeterministic; corrected test uses the intended
fixed node budget (`max_seconds=None`, analytic/vectorized) rather than relaxing
tolerance. Full root regression133passed5skips2existingwarnings. Collection and
learning with this recovery-state oracle have NOT started; that is the concrete
next development work, with supervised data/cost/fairness explicitly disclosed.

All earlier training/norm-factorial/native verification is complete; no CARLA
server expected. Publication preflight/commit/push and independent oracle check
follow this snapshot; inspect Git and clone receipts for actual final status.
The overall objective remains ACTIVE/incomplete, no external blocker, no agents.

# Historical continuation snapshot — 2026-10-23 10:49 local

This goal turn is progress, not completion. Full objective remains ACTIVE with
no hard blocker. Source d9f48cd28b892928f69822f3131c7263489db0a9 is pushed and
remote-verified. Both root and independent pinned-environment checkout pass
132 tests / five conditional skips / two existing deprecation warnings. Later
changes are reports plus normalization-factorial protocol support; inspect Git.

All training is now COMPLETE and worker handles are gone:

- Native recovery 010623Z: H 70,080 new controls/438 optimizer steps; B1 100,270
  controls/630 optimizer steps. All 176,167 training/evaluation controls matched
  raw CARLA frames/state/IMU/clock. Both training-road evals 0/4. B1 two bounds
  and two deadlines; training 111 bounds/61 deadlines/one success/one unfinished.
  H recovery charged lower bound 164,409 vs retained lineage 100,160; do not
  erase the parent's 64,249 lost controls or unknown unflushed tail. Plot read.
- Independent checkout live v3 batch 023532Z: H 640 controls/701 ticks/four
  optimizer steps; B1 952 controls/1,044 ticks/six updates. All 1,592 frames
  verified, reload exact. Actual clean-source runtime check, not performance or
  independent-seed evidence. Both bridges exited; parent-owned server 21808
  was verified empty/identity-matched and stopped at 10:36. Receipt in recovery
  infrastructure/server_stop.json. No CARLA server remains expected.
- Warmstart 021813Z: supervised 6,400 updates, then actual PPO 25,600/102,400
  controls and 4,000/16,000 optimizer steps. All three endpoints 0/10. Final
  short parking 2/0/0/0 out of two each. Both PPO full-task endpoints eight
  deadlines/two narrow collisions. Guard exit0 and OS worker gone. Full analyzer
  verified source/data/replay/labels/evaluation traces/terminal metrics. Plot
  inspected. Independent checkout loads all three local models with frozen
  stats; receipt runs/demonstration_model_reproduction.json inside clone.
- Eight BC-only demonstration-layout replays: train and holdout each 1/4,
  narrow only. Additional 48-case weight/normalizer factorial completes: all
  four combinations 0/10 full task. Only BC weights + later norm get 6m2/2.
  Changing norms alone does not repair full-task performance. BC empty rollout
  stops short at x25.753/v.062 then accelerates through x27.082/v1.135 and leaves
  bounds. Investigate physically visited recovery-state coverage and terminal
  timing; do not claim effective B1 or blindly repeat PPO budget.

Current publication in flight (revalidate before retry): gh.exe PID31084,
created 10:40:51.530 local, uploads two new mean-transform assets; exec session
89608. At 10:48:51 OS live, read-transfer count 221,027,679, localhost proxy
connection established. Neither upload success nor new remote assets confirmed
yet; do not claim published or start duplicate uploads. Local immutable assets:
bounded_mean_stability.zip 185,786,166 bytes/255 manifest files/SHA256
e3dba9ce5f6346a8c4710236a2dbe24900e1554109263eb9faced7e659d4628a;
softsign_mean_stability.zip 92,312,679 bytes/218 files/SHA256
cb57defabc169472ad36de1fc37b89c6a5f319154829a95179a20be0b5eb948f.
Publication still requires actual upload/download/member hashes/model reload and
chart reproduction before documenting completion. Native recovery and BC data
are not release assets. Current main report/readmes reflect completed failures.

Next meaningful research intervention should follow the closed-loop evidence,
e.g. supervised recovery-state coverage from actual learner-driven trajectories,
with teacher queries/cost disclosed and no teacher at learned-policy inference.
This has NOT been implemented or started. Preserve the required PPO hierarchy,
fair strong B1/B3/H0/B2 comparisons, five policies, native competencies, fresh
120-condition confirmation and final acceptance thresholds. No agents or other
project work. Finish publication/source reproducibility without substituting
those checks for actual research competence.

# Historical continuation snapshot — 2026-10-23 10:33 local

The preceding summary turn was a verified wait (OS-confirmed native worker).
This turn makes progress: executed demonstration fitting and real PPO, completed
raw dataset/replay and closed-loop audits, and full 132-pass/5-skip regression.
The full objective remains ACTIVE and unmet; no external blocker or agents.

- Native recovery `20261023T010623Z_native_recovery`: H complete and verified as
  below. B1 completed 100,270 controls / 105,663 world ticks / 630 optimizer
  steps, exact reload, and is evaluating. Main process 9248/launcher 17016;
  B1 bridge 42388 and owned server 21808. Revalidate OS before touching them.
  Full batch summary/raw-frame analyzer and independent live v3 replay pending.
- Softsign `20261023T015449Z_softsign_mean_development` COMPLETE: all endpoints
  3/4/5/6 of 10; final static and unseen both 0/2 collisions. All parking probes
  pass after the initial point. Final total 5,698,852, addition 600,276.
  Analyzer verified and plot inspected. Report now records the negative result.
- Demonstrations `20261023T021232Z_executed_demonstrations` COMPLETE: 32 cases,
  all teacher/direct successes, 10,817 controls each, 1,088 teacher decisions.
  Full paired raw states and labels verified, max error 2.66573e-7. Dataset hash
  dfaecf8cd4a381dc71c50e991f2d34c866c75928977e4945517fadf7df8cee71.
- Warmstart `20261023T021813Z_demonstration_warmstart` LIVE: worker 46772,
  launcher 28024, guard 26136/launcher 10368, created around 10:23:40. Fixed
  deadline 11:18:14, guard resources/progress and no retries. Frozen source;
  never edit it. 200 supervised epochs/6,400 separate updates complete; 8,113
  train/2,704 heldout examples. BC-only 0/10 (4 bounds/6 collisions), parking
  0/2,2/2,0/2,0/2. Additional train/holdout layout audit each 1/4, only narrow
  succeeds. Actual PPO reached 25,600 controls/4,000 updates; fixed endpoint
  evaluation running, then 100k. This is extra supervision and hold1, not pure
  PPO or a single-factor comparison with previous hold10. Do not relaunch.
- New report `reports/demonstration_diagnostic_20261023.md`; analyzer verifies
  sources/data/raw controls/labels/endpoints/model hashes and separately counts
  BC and PPO. It requires completion unless --partial explicitly requested.
  New models/data are NOT yet release assets. Publication preflight 306 files,
  no findings; GitHub authenticated Leo-7GOAT and correct remote verified.

Next: finish native evaluation and all raw frame verification; independent
checkout live v3 run only after original bridge/worker exit, then stop the
verified empty server through its owning recovery batch. Finish all warmstart
endpoints before selecting a new intervention. Low teacher-state MSE but failed
closed loop suggests insufficient recovery-state coverage, not a working B1.
Publish/reproduce tested source and preserve all negative outcomes. The current
132-test result predates documentation/analyzer-only changes later this turn.

# Historical continuation snapshot — 2026-10-23 10:04 local

Sourcefba43c7fa4dfa579850dac31b0d0d7485ae17f7fis pushed/remoteverified. Mainworktreeand independentnative_road_checkoutareclean. Independentpinnedvenvfullregressionatthatcommit:130passed5skips2SB3deprecationwarnings (`runs/softsign_reproduction_tests.xml` insideclone). Earlierdownloadedrateasset8modelreload/endpoint/figurechecks remain verified. A docs-onlycommitmayfollow.

Only two training workloads remain live, both OS identities rechecked: native main9248 (created09:07:26) /B1bridge42388(created09:39:10), sameownedserver21808; B1~42.4kcontrols/133phases, immutable40,083checkpoint252optimizersteps/SHA3c0faa45ed7dd333feb9361c5e9e36a8e41652c2b90ca2aea76a7801462eaa03. Softsignworker8336(created09:55:17)/guard47768(created09:55:50), +300,488endpoint5/10 andall8parkingprobespass; first0/+100k/+300kfulltask3/4/5outof10. Final+600k pending. Do notrestartthesejobs. GPU50C,9,904of16,311MiBused;~35.4GiBRAMand61.95GBdiskfree. Native deadline11:06, softsigndeadline10:54:52. No resourceblocker.

Finish/inspect currentendpoints before anotherparameterchange. Static/unseenstillunresolved: do not callhealthyB1orhierarchyadvantage established. Offline route-proposal function remains diagnostic only, not enabled in training. Hnativecomplete/rawverified; B1fullanalyzerandnewindependentliveCARLAreplaypending. Full objective remainsACTIVE/incomplete.

# Historical continuation — 2026-10-23 09:56 local

This turn is substantial progress; full goal ACTIVE/incomplete/no hard blocker/no agents. Read full objective attachment before continuing. Supersedes old live entries below.

- Native recovery010623Z: H COMPLETE70,080newcontrols/72,497trainingworldticks/219phases/438updates/reloadexact/actorL2change1.176118. Model lineage100,160; chargedcostlowerbound164,409;64,249lostparentcontrols. All4trainingroadprobesdeadlines (progress7.594/10.161/16.987/9.447m). All73,680train+evalcontrols matched rawCARLA/frame/IMU/time; receiptH/frame_verification.json, rawHbridge94695b22SHA11a1428b136ee6cf5d3521341500f393e6be04915b313345969aff913c0f9895. B1 NOW RUNNING fresh100k, main9248/launcher17016samecreated09:07:26; ~30.6k at09:55, actualupdates. Bridge rotated toB1 (newresponsecc3b5501); recheck actualPID. Serverstill ownedby010623Z12320/21808. Deadline11:06. Do notstopserver or startsecondtickowner. FinishB1 thenfullanalyzer incl recoverycostchecks; oldparentH0/interruptions mustremainin eventualasset.
- Ratepair011009Z complete and PUBLISHED: paired_ppo_rate_stability.zip67,837,038bytes/256files/SHAb284991976fe7ad87405666a548f837b6561bdcc84b3f674e276a7213bc0fb71. Actualupload+freshdownload/memberhashesverified. Independentcheckout atsourcee81b378loadsall8downloadedmodels/frozennorm, allendpointscientificrowsidentical andfigurebyteidentical. Receipt native_road_checkout/runs/rate_stability_download_reproduction.json. Existingtagunchanged; releasebodyupdatedwithsourcecommitrequirement. Sourcee81b378pushed/remotematched. NeweditsbelowUNCOMMITTEDatthissnapshot.
- Tanh boundedpair012208Z COMPLETE, workers33992/32048exitedOSverified, guardexit0. Original/tanh all0/+100k/+300k/+600kfulltask5/3/5/6vs6/6/5/6. Finalactual5,700,342vs5,700,646controls, all8parkingprobespassallendpoints; static/unseenallfail. Analyzerhashcheckscomplete/plotvisuallyinspected. Original/tanhKLmedian.02873/.01765;P95.05145/.04771;max.07702/.06134. Notamainhypothesissuccess; neverselectintermediate6vs3asfinaladvantage. Notyetreleasedasset.
- Finaltanhobstacleaudit: obstaclecoordinatesnotnormalizedclipped, staticGaussianmeans.9992–.9999fromraw3.94–5.11. Allnoise-frequency1/4/10staticprobes0/8trainandvalidationstochastic(anddet), collisionsat3m/s. Correctedexplicitstaticdiagnosticsdisableresetcurriculums andassertepisodeidentity; defaultoldemptyprobesunaffected.
- LIVE softsign followup015449Z (`runs/softsign_mean_development.json`): worker8336/launcher11936 created09:55:17; guard47768/launcher12804, freshguardhealthy. StartsSAME5,098,576originalcheckpointascompletedpair, onlymeantransformsoftsign; onebranch, completedunbounded/tanhcomparatorsNOTretrained. Targets0/+100k/+300k/+600k, deadline~10:54:52. Zero-update3/10,23m1/2(retainnegative), actualPPO~40,473newcontrols/16newphasesat09:56, actorupdateL2.0794. Execute frozenpaired_update_stability.py alreadyrunning; no duplicate. Allsourcehashes/initialweight+optimizer+RNGmatch/protocolparenthashrecorded. Guardverifiedimmutableprocesshandle/resources/progressdeadline/noautomaticretry.
- Offline routeproposalonly: newroute_frame_candidates usesknowncentrelinearc/tangent andterminaloffsetenvelope, nevermovesstate/notenabledinanyenv. Initialwitnesses18/4/22/12vsgoal-frame10/4/14/10; ego24/14/26/22. Terminalroute0/8/0/0sameasgoalframe; cannotrepairthreedeadendposes. Keepnegative, noadoptedimprovementclaim. CandidateauditnowmatchesgeometryIDnotfirsttrainingorder; regressionadded. EvidenceH/route_candidate_comparison.json.
- Latestfulltests130passed5skips (softsign_regression XML), 2SB3deprecationwarnings. Need source/docs preflight/commit/push andindependentcheckouttests afterlatestchanges. Mainreportrecovery_stability_20261023.md/readmesupdated. Subsequentcommitsmayfollowthissnapshot; inspectGit.

Next concrete work: finishsoftsignandnativeB1; analyzeallendpoints/failures andrawframes; source/release/reproduction. MainresearchgateshealthyB1,5independentpolicies,H/H0/B2/strongB3/frequency/shield/fixedref,native static/dynamic,comfort,120freshconditionsstillUNMET. No external blocker; continue failure-driven iteration, not acompletionclaim.

# Historical continuation — 2026-10-23 09:25 local

Previous summary turn was a verified wait (OS-confirmed native and paired workers); current turn is progress: completed rate-pair analysis, executed obstacle-response probes, implemented/tested a bounded Gaussian mean, and actually launched the paired experiment. Goal ACTIVE/incomplete; no hard blocker, no agents. Read the full objective attachment before continuation. New report: `reports/recovery_stability_20261023.md`.

- Old native batch152120Z interrupted: original worker/server gone, H94,329 readable controls, truncated gzip preserved, last retained model30,080, B1not started. H0complete100,160controls/0of4eval. Cause unknown; not an OS reboot. `interruption_audit.json` records losses, `recovery_registration.json` names the sole child. Do not rerun parent or old PIDs.
- LIVE recovery `runs/20261023T010623Z_native_recovery`: main9248/launcher17016, bridge48144 at last check, owned server12320/21808 created09:06:35. H49,580newcontrols/154phases at09:25; budget69,920new then B1fresh100k. Deadline11:06. H lineage starts30,080, charged parent lower bound94,329, unknown tail; not budget matched. New periodic checkpoints every10k contain model/norm/optimizer/RNG. Old checkpoint uses explicit freshseed11 RNG and new actual episode. Recheck OS identity, never restart on stale heartbeat. After all bridges exit, cleanup through `scripts/stop_owned_carla.ps1 -Batch runs/20261023T010623Z_native_recovery` with existing isolated3.7 client.
- COMPLETE ratepair `runs/20261023T011009Z_update_stability`: full tasks4/1/4/4vs4/6/5/5, static0/2atallpoints, 23m2/0/2/2vs2/2/2/2. All otherretention2/2. Full models/cases/source verified by analyzer; figure inspected. Final lower endpoint5,098,576 used for next experiment, not the best6/10point. New obstacle audit shows5collisioncases all clippedmaxaccel, staticrawmean3.81–5.18,std~.51; steering responds to obstacle-removal inputs, which never execute.
- LIVE boundedmeanpair `runs/20261023T012208Z_bounded_mean_pair`, pointer`runs/bounded_mean_pair.json`: actualworker33992/launcher48544 created09:22:24; actualguard32048/launcher21900, identity file and freshguard.json verified. Samecheckpointmodelweights/optimizer/norm/RNG/env, only Gaussian mean mu→tanh(mu); bothlr1e-4. All inheritedcosts retained, notfromscratch. Milestones0/+100k/+300k/+600k, deadline~10:22; zero-update5/10vs6/10 is transfer effect, notlearning. Original/bounded+100k3/10vs6/10; static/unseenstillallcollide; inspect remaining. CPUguard checksdisk12GiB/RAM4GiB/GPU85C/progress300s/deadline, zero retries, retainedhandleonly. No CARLA ownership in this pair. Do not rerun started experiment.
- Fulltests123passed5skips includingnewhandleterminationtest. Source/docs currentlyUNCOMMITTED, localHEADb4e893e (docs) notyetpushed; verified remote2b7a55c. Need preflight/fullregression afterlatestguard changes and ordinarycommit/push. New results NOT in oldreleaseassets. Independent livev3checkoutreplay pendinguntilserverfree.

Next: finish both live batches; run analyzer on completed boundedpair, retain all endpoints/failures. Verify native final raw frames and separate resumed lineage/costs; package recovery must include interrupted parent and H0, not pretend parent complete. Update reports/publication/independent checkout. Healthy fairB1, 5independentpolicies, formalablations/strongB3, native static/dynamic, comfortfix,fresh120conditions and researchgates remain incomplete. No completion claim.

# Historical verified publication and live work — 2026-10-22 23:37 local

Supersedes pending details below. Source2b7a55c pushed/remoteverified. Both indexedZIPs uploaded and actualfreshdownloads/all1123memberhashesverified in `runs/prerelease_20261022/indexed_download`. Independent native_road_checkout clean at2b7a55c:117tests5skips, two downloaded models reload/frozen stats, all five batch plots/summaries byte-identical. New live clone CARLA replay deferred until registered queue releases server; do not start another tick owner.

Nativequeue152120Z LIVE: worker52840/launcher8152 created23:21:31; H bridge24812 created23:35:37 (H0bridge35156 exited). H0COMPLETE100160controls/103631worldticks/626optimizersteps/reloadexact/actorL2change1.46485;111deadlines+one unfinished training episode, eval0/4deadlines. Hstarted23:35,2400controls observed23:36:39; B1pending. execsession17202active, deadline01:21:20Oct23. Ownedserver17516/19032 remains; cleanup afterallbridgesexit through owner144123Z. RevalidateOS before actions. No lightweight training live.

B1 normcross session45864 COMPLETE: 4.898M policy with old/new stats3/10and3/10;5.001M policy0/10and0/10; all6m2/2. Evidence `late_regression_norm_cross/results.json` and raw traces. Earlier normalization alone does not repair regression; no KL causality claim. Intermediate full-task4.496M4/10and4.898M3/10;4.996M23m0/8allmodes. Final0/10retained. Next B1 experiment should isolate update stability with paired same-checkpoint branches and frequent identical development validation; no such branch launched yet.

Current turn progress; full goal active/incomplete, no hard blocker. Finish H/B1 queue and verify all frames, then independent live CARLA replay and owned-server cleanup. Five independent seeds, fresh120conditions, strongB3, native static/dynamic, comfort and research acceptance remain incomplete. Documentation commit may follow source2b7a55c; inspectGit.

# Authoritative continuation — 2026-10-22 23:35 local

Previous summary turn discovered the finished native batch; this turn makes concrete progress: all26,470frames verified, new paired feedback training actually started, intermediate B1 regression localized, tests/report/assets implemented. Full objective active/incomplete, no hard blocker, no agents.

- LIVE native batch `runs/20261022T152120Z_native_road_learning`: frozen H0/H/B1, seed11, each100k controls, geometryv3,4546inputs. Only feedback differs H0(zeros)/H(enabled); B1 gets same enabled witnesses. Same originalego actions/reward/MLP64/PPO32rollout2epochs/45s tasks. Deadline01:21:20 local Oct23. Mainworker52840/launcher8152 created23:21:31; bridge35156 was H0 bridge, revalidateOS as methods rotate. execsession17202. H0last~78k at23:32, realoptimizerupdates. Save10k/30k/100k inference checkpoints; these are NOT seamlessCARLAresume. Never restart started batch or run a second tick owner. Guard checks disk12GiB/RAM4GiB/GPU85C/deadline everyrollout. End evaluation is fourTRAININGroads only. Next inspect final models/eval/rawframes, not assume completion from oldheartbeat.
- LIVE CARLAserver owned by `runs/20261022T144123Z_lane_union_audit`: launcher17516/shipping19032 created22:41:25. ReusedforregisteredPPO; do notstopwhilebridgealive. After all work close/verifyempty with `scripts/stop_owned_carla.ps1 -Batch runs/20261022T144123Z_lane_union_audit`. Do not kill reusedPIDs. No other live lightweight training.
- Completed native10k batch150606Z: H10240/B110192controls,64optimizerupdates each/exactreload. EvalH0/4(deadlines),B10/4(2bounds2deadlines). Total26470controls includingeval verified by updatedanalyzer; train20432+eval6038. v3fixedH150241Z4/4/1958controls; B2150456Z4/4/1938controls. v2failure145407Z3/4 retained. Severe−25m/s²actuation spikes unresolved. See reports/indexed_native_geometry_20261022.md.
- v19finished5001122steps/42586updates/final0/10. Completedretention4M/5Mprobes;6/12/18mdet8/8each,23m0/8. Newfrozen4,496,363and4,898,060checkpoints each23m8/8allfourmode/splitgroups; fulltasks4/10and3/10, static/unseencollisions. At4,996,02923m0/8allgroups. LastupdateKL.05715versustarget.03 (not hardbound). This is late regression, not inability ever to learn parking. Normcrossfactorial currentlyexecsession45864 in`late_regression_norm_cross`; inspect completion before rerun. Intermediatefulltaskresults in`intermediate_full_task_audit`, originalfinalnotreplaced. Additional4/5Mprovenanceinruns/v19_retention_complete_provenance.json.
- Fulltests117passed5conditionalskips (indexed_feedback_regression XML); CPython3.7bridgemodules compile. FixedH/B2andPPOfiguresvisuallyinspected. Newassets uploaded to unchangedprerelease: indexed_native_geometry_diagnostics.zip(883members,6814464bytes,SHA0290b78f1ff4a8fafea681c93697ee10dc80bf8f9455feaefc2ee8226bef7e50), indexed_native_ppo_development.zip(240members,16082364bytes,SHAcf353f225bb81dbcf00946f702a39696f76aca7a1a98a239e9fa5344b5cc2864). Actualproxydownloadrunningexecsession86805 inindexed_download; WAITcompletionbeforehash/extract. Sourcecommit/push andindependentcheckoutrepro pending atthissnapshot; inspectGit forlaterupdate.
- Need fullresearchobjective: healthy/fairB1,5independentpolicies,H0/B2/strongB3/frequency/shield/fixedreference,native static/dynamic,comfortfix,fresh120conditions,preregisteredadvantage. Fourroutes/onepolicy/sharedtrainingeval cannot meet these. Continue evidence-driven work, do notmarkcomplete.

# In-progress authoritative update — 2026-10-22 23:03 local

Previous turn progress; current turn also actual code/tests/CARLA/learning-diagnosis progress. Goalactive/incomplete/nohardblock/noagents. Read objective attachment beforecontinuing.

v19COMPLETE5,001,122physics/513,280actions/2005phases/42,586optimizersteps/11,485epochs/25,327episodes;actorL2change13.5108/reloadexacttrue/supervisorexit0. OriginalPIDs14444/49696goneOSverified.4Mfulltask0/10:6/12/18mdet8/8,23mdet0/8;stoch8/7/7/2.5Mfulltask0/10;finalfourdistanceprobesandterminalaudit running in execsession58128 (inspecthandle beforepoll); first6mdet8/8/stoch7/8. Do notrestartcompletedtraining.

CARLAserver LIVE ownedby`runs/20261022T144123Z_lane_union_audit`,launcher17516 fromserver_start.json. Reusedforboundednewcontrolbatches;stopthroughthatowneronlyafterallbridgesexit. NoPPOtraininglive. v2preparedPPO batch145507Z explicitlyNOTEXECUTED becausejunctionomissionfound;receiptretained.

Newroad diagnosis: sixhistoricalrawrejectedpointsallinside refinedDriving-laneunion.48/48inner/outer controls correct. Localunionv2recovers33/0/23/0gridcells,loses0. Actualv2Hfixedplans batch145159Z4/4/1963controls;v2B2batch145407Z3/4/1566controls, onejunctionrawfailure. Bothverifiedframeanalysis; severe−24m/s²brakingstillpresent. Road5/6laneseamsclosedwithoutwidthdilation; atnewfailureclosestroad516isoutsidebutnearbyroads469/515inside. Readonlyproofinlane_union_audit/junction_nearby_roads.json.

Implementedopt-inversionsv2andv3(DrivingSurfaceIndex,1mOpenDRIVEsamples/refinednearbylanesacrossroadIDs); oldv1defaultunchanged. Signatureversionnew, sharedpolicygrid/planning/risk/termination,rawmissesretainedseparately. v3readonlygridauditalso48/48controlsand33/0/23/0recoveredcells. IndexedplanningauditallH/B2feasible. Newv3actualHcontrolcurrentlyexecuting;inspectruns/native_curve_control.json andOS/exec handle. Noasphaltmesh/lanecomplianceguarantee; currentunioncanincludeoppositeDrivinglanes,appropriateonlydeclaredlow-speeddrivable-area task.

SourcesUNCOMMITTEDthisturn. Tests114passed5skips beforev3; focused9passafterv3. Needfullregression,actualv3H/B2,thenfreshpairednativePPOupdates/eval withv3; source/publication/reproductionpending. Newtrainingentrynowacceptsgeometry/budget/wall/episodefields. Emptyfailedofflineprepdirretainsreceipt; sourceinitialhashreconstructioncorrectedLFandverified. Initialdiagnosticmodule/scriptfrozenunderlane_union_audit/initial_source. Fullgoalstillneedshealthyflatmixedtask,5seeds/formalablations/fresh120conditions andpublishedreproduciblefinalresearchevidence.

# Publication/reproduction update — 2026-10-22 22:37 local

Source53361770b667bb6802b1570286e692245882554cpushed/remotematched; documentationcommitmayfollow. NewB2assetuploaded/downloadverified243members/3,105,133bytes/SHA296a5614e17bdcfdd7d1c2fb094e0db289c1cd6b57e1c8a0496dd2c0b0afb76a. Independentnative_road_checkoutclean,111tests5skips,allfourpublishednativePPOmodelsreload/frozenstats,downloadedB2analysisplot+summarybyte-identical. Actualclonebatch143149ZB2execution2/4vsoriginal1/4,1381controls/1504worldticks,allframesverified, ownedlauncher53244serverempty/stopped. NoCARLAexpectednow. Reportexplicitlyretainsoutcomeinstability; noPPO/B2superiorityclaim.

v19onlylivetraining; latest heartbeat 3733898physics/1499phases. Supervisor14444/worker49696 mustbeOSrevalidated. Original5M/deadline~23:41; do notduplicateorclearlease.3Mfulltask0/10;6/12/18mdeterministic8/8each,stochastic8/6/8. Allfive2M/3Mauditcheckpointhashesandfrozensourcesverifiedin`runs/v19_retention_provenance.json`. Next evidence is4M/5Mfull-task plus retainedshortdistances. NewB2 hasnominalfeasibilitybutexecutionclearanceandrawroadsemanticsremainunresolved. Fullgoalactive/incomplete/nohardblock; thisturnwasprogress.

# Active research handoff — 2026-10-22 22:29 local

Previous turn was a verified wait (OS-live v19 worker), this turn is progress. Goal remains ACTIVE/incomplete, no hard blocker/no agents. Read the full objective attachment before continuation. New report `reports/b2_rehearsal_20261022.md` is authoritative over historical snapshots.

- v19 sole live training: batch`20261022T134149Z_route_rehearsal_v19_development`, group`route_rehearsal_v19/B1_route_rehearsal_seed11`, supervisor14444/worker49696 (original creation21:42:58/21:43:02), revalidated OS. Last~3.18M/1280phases; deadline~23:41/5M target. Do not restart.2M6/12m deterministic and stochastic8/8;3M6/12/18m deterministic8/8, stochastic8/6/8. Full-task1M/2M/3M0/10. All probes use frozen batch source. First1M v18/v19 tensors/norms differ (checkpoint can straddle curriculum boundary); don't claim bitwise paired branch. Latest staged rehearsal retains basic parking, obstacle ability unproved.
- New optionalB2 `grid_postprocess=body_spline_v1`: corrected2Dsearch plus150-evaluation local body/curvature spline fit; conservative swept-body/curvature interval certification, at mostfive subdivisions. Legacy default/signatures preserved, newoption versioned.4/4saved nativepaths feasible and reference-bodyfree vsold0/4. Evidence`runs/native_b2_certified_20261022`; preliminary3/4audit and failed smoothing/point-SDF probes retained. NoHybridpath used byB2.111CPUtests pass/5conditional skip.
- ActualB2 fixedplanCARLA batch`20261022T142205Z_native_curve_control`COMPLETE1/4:rawroad/raster/rawroad/success,1111controls/1234worldticks. Zeroobservedcollision/QPfailure; brakingmin−23.6m/s² remains. Rawframesverified, `probe_gap_audit.json`queriesactualnearestlanes. Allfailuregates unchanged. Ownedlauncher19380/shipping30724verifiedempty/stopped; noCARLAexpectednow. Rasterfailure showsnominalpathfeasibilityinsufficientfortrackingclearance.
- Prior asset`rehearsal_native_feasibility_diagnostics.zip`is uploaded; actualdownloadat`runs/prerelease_20261022/rehearsal_download`verifiesarchiveSHAf4dc4d3316c361ec407cce5fde3f94b42813977be04c5fc2cf74e145c1165092andall220members. ItpredatesnewB2. Currentcode/docs publication/reproductioninprogress; inspectGit/releasesratherthanassumingdone.
- Readmes nowremoveoldliveclaimsforcompletedv13–v18. Full five-seed fair comparisons, nativegeometry/control robustness, PPO-vsstrongB3value, final120freshconditions and finalresearchgates remain. Do not markcomplete orblocked.

# Active research handoff — 2026-10-22 21:56 local

**Newest turn is progress:** frozenpolicy/normalization factorial48episodes completed;4Mpolicy2/10withbothnorms,5M0/10withboth,6m0/2all. Changingnormalizerdoesnotrescueparking; oldnormturnstwo5Mstaticcollisionsintodeadlines. Verifiedmodelhashes/frozenstatsandalltracesin`runs/v18_normalization_transfer_4M_5M_verified`. See`reports/rehearsal_feasibility_20261022.md`.

- Newactualv19`runs/20261022T134149Z_route_rehearsal_v19_development`, pointer`runs/route_rehearsal_development.json`, group`route_rehearsal_v19/B1_route_rehearsal_seed11`. Freshsamev18seed11/5M/605MLP/reward/PPO; ONLYroute_rehearsal_probability0to.25. Previouscompleted distances sampled uniformly on25%trainingresets;after4M this replaces25%mixedepisodes. Validationunchanged; allphysicalcostcounted. At21:54~930k/377phases;100k/300k0/10. Worker49696created21:43:02.540,supervisor14444created21:42:58.619,launcher45832/12672. OSverified, singleCPUthread, deadline~23:41,guards/noautoretries. Sourcefrozen; don'teditit/restartorcreateanotherinitializer. First1Mdeliberatelyunchanged; auditretentionafter2Mbeforeconcludinganything.
- Nativeall4Hybridpathsfeasibleinsharedgrid at10000nodebudget:1938/2444/71/204nodes (firsttwoexceednormal1800). Reparameterizedpathsfullbodyfree. B2searchgridroadinflationomissionfixed; allfourstillpostprocessinfeasible, preservedin`runs/native_planner_audit_20261022`. DiagnoseB2curvature/clearanceinstead ofassumingstrongbaseline.
- ActualfixedsearchedpathCARLA`20261022T134803Z_native_curve_control`: outcomesoffroad/success/offroad/success,1462controls,noobservedcollision/QPfailure,lateralRMS.00325/.06335/.01661/.05072. Failurepointsnearest-lanehalfwidthexcess.000505/.006833mwith~.042mlongitudinalprojectionoffset. CARLAsourcechecksplanardistance; sharedgridfivesamplescannotprovecontinuousroadmembership. DoNOTrelaxgates/dropfailures; nextrawgeometryworkshouldinvestigateprojectedlaneunion/properroadboundaries. Actuationlegacyspikes~−26remain. Ownedlauncher49476serverempty/stopped; noCARLAexpectednow.
- Fulltests109passed5skips. Asset`rehearsal_native_feasibility_diagnostics.zip`prepared220files/2,807,329bytes/SHAf4dc4d3316c361ec407cce5fde3f94b42813977be04c5fc2cf74e145c1165092; includessolecontrolbatchall4cases,plannerfailures,normaudit. Publication/reprowillfollow; inspectGit/releaseforfinalstate. Onlyv19traininglive; fullgoalactive/incomplete,nohardblock,noagents.

# Historical handoff — 2026-10-22 21:21 local

**21:33 final authoritative update (supersedes live states below):** v18COMPLETE5,002,111physics/513,024actions/2004phases/47,170optimizersteps/12,566epochs; exactreload/actorL2change14.2612. `completed.json`and`supervisor_final.json`exit0verified, worker11116/supervisor7088OShandlesgone. `worker_status.json`staysstaleevaluating; neverinferlivenessfromit. Finalfulltask0/10(twoemptydeadlines/eightobstaclecollisions),emptygoalerrors1.034/.992m;allobstaclecollisionsat3m/swithoutbraking. Frozen5M23minference0/8det,1/8stochastic;4Mwas8/8and6/8. Last100mixedtraining20success/63collision/17deadline. Terminalaudit5000000saved. No newtraininglaunched; retention/obstaclelearningdiagnosisnext, notblindmoresteps.

Source7a82fbc pushed/remotematched; nativeasset uploaded tosameprerelease and actualproxydownload396hashesverified. Slowdirectdownloadownedgh17416wasidentity-checked/stopped; partialarchivepreservedin`road_download_verified`, verifiedfullassetin`road_proxy_download`. Fresh`native_road_checkout`107passed5skips/all4downloadedmodelsload/frozennorm/secondbatchplotandsummarybyte-identical. Itusesitsownsource+independentclean_checkoutvenv. Actualclonebatch`20261022T132946Z_native_road_learning`H/B1each640controls/4optimizersteps;world701/732;exactreload,H1deadline/B1two rasterfailures,oneunfinishedeach. All1280framesverified. Ownedlauncher44548serverempty/stoppedwithreceipt. NoCARLAserverortrainingexpectednow; revalidateOSbeforeactions. Allfourrastergoalsfree;surveycentrelinebodyblockedpoints0/4/1/0: nativefeasibilitymustbeaudit,notassumed. Latestreporthasfullnumbers and publicationlimits. Fullgoalactive/incomplete,nohardblock. Finaldocumentationcommitmayfollowthissourcecommit; inspectGit.

**Newest continuation is progress:** shared native64x64 road geometry implemented in World/planning/observations/risk/termination, actual H/B1 PPO training and full raw-frame validation completed. All earlier unsupported-native statements below are historical. Enable`road_observation=True`for native curves; base4447inputs/newsignature; old models rejected, experimentalPI/FF still guarded. FullbodySAT covers interior holes; all four architectures receive the full raster.107passed5skips; CPython3.7bridge compiles. Read`reports/native_road_learning_20261022.md`beforecontinuing.

- v18 only live worker11116/supervisor7088, verifiedoriginalcreation20:35:16/20:35:13. Latest~4.65M/1866phases at21:20, mixedstage after4M, stillregistered5M/deadline22:34. Do not restart. 1M/2M/3M/4M fulltask0/10,1/10,0/10,2/10. Frozen2M12m8/8bothinference modes but6m0/8;4M23mdeterministic8/8/stochastic6/8,6m1/8and5/8. Duplicateemptygeometrynotindependentconditions. Final5Mstillpending atthissnapshot; auditfinalresults/retention before nextlearningdesign. Initiallive-source6m audit was repeated usingfrozen source withsame0/8; prefer`retention_2M_6m_frozen`.
- Nativebatches`20261022T130938Z_native_road_learning`and`20261022T131707Z_native_road_learning`complete. H640controls/701worldticksinboth,4optimizersteps/2phases;B1first949/1072world/6updates,second640/732world/4updates. Allactorchangespositive/exactreload. H32successfulplans/29invalidtargets/320solvedQPs/1deadline;B1first3rasteroutbounds,second2. Allretainoneunfinishedepisode; no tasksuccess. Same seedtwo repeatsnotindependentpolicies. Analysischecksall2869state/frame/IMU/time recordsagainstrawbridgecontrols. First runner'sdeclared12GiB/GPUguardwasnotfullyenforced;secondfixesactualguardcallsandrecorderclose. Frozenfirstsourceandfailurelogsretained.
- Mapqueryaudit`20261022T131600Z_native_grid_audit/map_queries.json`examinesrejectedgridpoints;none>5cminsidenearestDrivinghalfwidth/noheightrejections. Narrowlaneseamscaninflateintooccupiedcells; donotclaimphysicalasphaltdepartureorfillholeswithoutgeometryevidence. Actualserversbothverifiedempty/stopped: firstreceipt130938Z,insecondcaseauditbatchreceipt. NoCARLAserverexpected. Sampledgridaccuracy and routefeasibilitystillneeddevelopment; static/dynamicnativeintegrationnotcomplete.
- Newasset`native_road_learning_development.zip`prepared396files/14,053,181bytes/SHA256b54fdd0f142f1944878a9e095363ffde716162f2dd772dcaafd9cb9b01c34757,includesallmodels/rawfailures/twobatches/queryaudit. VerifyGit/releasebeforeclaimingupload; currentpublicationwillfollowthisnote. No agents, nohardblock. Fullfive-seedfaircomparisons/fresh120confirmation remain incomplete.

# Historical handoff — 2026-10-22 20:46 local

**20:52 reproduction update:** source`abb2b1a7498452de72fe9e0118b42883f30f11e7`pushed/remoteverified. Independent checkout105passed5skips, cameraassetmodelreload, downloaded377filenativeassetintegrity, bothplots/summariesbyte-identical. Actualfresh-checkoutFFreplay`20261022T124930Z_native_curve_control`4/4/1920ticks, speedRMS.105–.166, jerk68–78; peaks~−24remain. Ownedlauncher45656server verifiedempty/stopped. v18onlylivetraining, last~1.89M/764phases in12mstage; inspect current heartbeat and OS. After2M, evaluate retained6mandcurrent12mparking plus unchangedfull-taskcases; do not hide regression if any. Updatedpublicationrecordcommitmayfollow; inspectGit.

**Newest state; older paragraphs are historical.** Previous goal turn was progress (actual control experiments, release and reproduction). This turn is progress: actual PI/feedforward CARLA comparisons, measured throttle fit, completed v17 diagnosis, implemented/tested progressive distance curriculum and real v18 training. Full user objective remains active/incomplete; no agents or hard blocker. Read the objective attachment before continuing.

- Live: `runs/20261022T123453Z_route_stages_v18_development`, pointer `runs/route_stages_development.json`, group `route_stages_v18/B1_route_stages_seed11`. Supervisor7088/launcher40856 created20:35:13; worker11116/launcher46556 created20:35:16. OS identity and command verified. Last at20:45:53 approximately1,308,860physical steps/530phases, now12m stage. Deadline about22:34; fresh seed11, fixed6/12/18/23m empty1M each then mixed1M,5Mtotal. One CPU worker; same v16 reward/605MLP/hold10/PPO, no heading PBRS. Canonical config comparison and frozen hashes verified. Never restart on observation timeout; no duplicate jobs.
- v18 actual short-route evidence:100k six-metre deterministic0/8, stochastic7/8;1M deterministic8/8, stochastic6/8. Train/validation starts duplicate simple geometry, not32independentconditions. Inference now uses saved gSDE frequency1 action instead of hardcoded10. Full ten-case1M evaluation remains0/10(two empty deadlines/eight obstacle collisions). Healthy short parking does not establish full-task baseline competence. Training episodes log actual route/goal and physical counts, not hidden pretraining.
- v17 COMPLETE3,001,352steps/1174phases/32,033optimizer steps, exactreload,0/10 at3M, eightdeadlines/twounseencollisions. Last100training2successes. Supervisor exits0. PID45864 was reused by an unrelated browser at20:25:40: leave it alone. No blind continuation of this failed reward hypothesis.
- Actual native control: PI batch`20261022T121852Z_native_curve_control`,1932ticks/4successes, speedRMS.478–.536, jerk96–107; response slow, frequent emergency brakes. Feedforward batch`20261022T122427Z_native_curve_control`,1922ticks/4successes, speedRMS.085–.188, jerk45–75 vslegacy.353–.369/130–142. Residual startup/stop deceleration reaches−24m/s² and some lateralerrors worsen to.064m. No uniform improvement/safety/PPO claim. Defaults unchanged; ordinary training rejects experimental PI/FF because integrator/filter state is not yet in observations. Emergency/finalhold/overspeed paths remain. The121810Zprepared batch has zero ticks and a preserved preflight not-executed receipt.
- `fit_carla_throttle.py`: fits cases0–2, checks case3; throttle-only moving windows, no brake in previous0.6s, measured0.2s velocity difference. Coefficients7.3265144,-.0838300,-.6463184; held-out development RMS.132m/s². All excluded fitting frames remain in full outcome/comfort stats. Fit provenance and calibration hashes preserved.
- Owned CARLA launcher23732/shipping34964 verified empty and stopped20:31:14; receipt in122427Z. No CARLA server expected. Only v18 CPU training remains. New archive`acceleration_feedback_diagnostics.zip` uploaded to existing pre-release,377files/2,487,456bytes; fresh download verifies SHA`c1f4cedb03d085b476c540e69e53a501e93dff0c3087186e4fb3d82c4226d6ba` and everymember. Earlier assets unchanged. New actual source tests105passed5conditional skips;259publication files scanned with nofindings. Publication/current remote SHA must be checked after this note.
- Next: finish registered v18 distance stages and audit retention/transfer before changing learning again; improve measured actuation startup/stop response and model lag, then integrate actuator state and native road observations for matched PPO training. Main five-seed comparisons, strong-B3 RL value, B2/shared shield/frequency, fresh120CARLAconditions and all acceptance targets remain incomplete. See `reports/actuation_curriculum_20261022.md` for actual commands/evidence.

Historical snapshots follow.

**20:09 publication/reproduction update:** code067db54pushed/remoteverified; independentcheckout102passed5skips, publishedcamera-modelreloadvalid, downloadednativeZIP559files/3,909,935bytes/allhashesvalid. SHA256`5f8822bc6caca88429e5830f829b916352c8e6588f0debe637b69399a65160a5`. All3figures/summariesbyte-identicalfromraw. Actualclean-checkoutnativeCARLAreplay4/4,1924ticks,comfortspikesrecur; serverlauncher14296ownedshippingstoppedafteremptyprobe. v17at2M0/10(nine deadlines/onecollision), emptycasesstop3.09–3.12mshortwithyaw~.35;2Mterminalauditcomplete. Itcontinuesoriginal3Mbudget, sameworker45864; noothertrainingorCARLAexpected. Newpackaging/release-recordcommitpendingatthisnote; inspectGit.

**Superseding snapshot.** Prior summary turn was a verified wait (specific OS-confirmed live workers). This turn makes progress: two real native control comparisons,12raw throttle phases, completedv13 B1, new failure evidence, code/tests/reports. Full objective remains active and incomplete. Read the objective attachment specified by the goal before continuing. No agents authorized or used; do not touch other projects.

- Only active training: `runs/20261022T113700Z_pose_potential_v17_development`, group`pose_potential_v17/B1_posepotential_seed11`, worker45864/supervisor20744, created19:38:31/19:38:28. At20:00:14heartbeat1,385,535physical steps/542phases; actualworkerOSidentity verified. FreshheadingPBRS0.1/radius6m, samev16baseprogress0.25/605MLP/commoncurriculum, registered3M target, deadline~21:37, resourceguarded. 100k/300k0/10;1M1/10(unseen7302),eightdeadlines/onecollision. Do not restart on a poll timeout or alter frozen source. Inspect2M/3Mwhencompleted and audit failure traces before another training change.
- v13hold1continuation`20261022T113828Z_staged_v13_b1_continuation` COMPLETE1,000,960physicalsteps/391phases/11,186optimizersteps, reloadexact,0/10all deadlines; supervisor_finalallchildrenexit0; actualworker17416gone. Originalparentcensored898,560steps is retained; no H/H0reruns. v16continuationCOMPLETE3,001,791steps/1175phases/32,781optimizersteps,0/10; six cases reachpositionandstopbutyawfails, fourcollisions. TerminalauditsandPBRSofflineidentityaudit saved. No B1healthyfull-task claim.
- Native fixed-reference diagnostics COMPLETE:112622Zlegacy1932ticks,114902Zcoast1932ticks,115357Zmanual-first1942ticks. Allnames end`_native_curve_control`. Each4/4success/noobservedcollision/nofootprintsampledeparture/noQPfailure, butjerk130–143m/s³anddecelerations~−25m/s²remain. Fixedreference/gains/vehicle/controlleridentity checked by`compare_native_control.py`; no repeatedseed/formalinterval/PPOclaim. Bothexperimentalactuatorsdefaultoff; coastworsensspeedRMS, manualgearimprovesspeedRMSbutnotcomfort. Rawbrakeprobe20phases1016ticks in112622Z; rawthrottleautomatic/manual6phaseseach720ticks in114902Z. Automatic .02–.12throttle staysgear0 for4s; manualfirstmoves, .12ends1.033m/s. This is startupbehavior evidence, notfullrootcause.
- OwnedCARLAlauncher11116/shipping51548 was verifiedemptyandstopped19:55:50; stopreceipt in115357Z. NoCARLAserverat20:00OSinspection. Reusedstartreceipt copiedto115357Z foridentity-checkshutdown;114902Zdidnotlaunchanother server. NeverkillPIDwithoutpath/creation/parentverification.
- New optionalreward fields preservezero-weightcheckpoint signatures; genericcontinuation acceptsselectedcleanwindowends withALLparentleases/exitzero/hashes/countersverified, refusesmanual/resourceSTOPorfailure. NativePPOstillrejectsunsupportednative_curvebeforeRPC; realroadobservations/rewards arenotimplemented. NativefixedMPCdiagnosisdoesnotfulfillthatrequirement.
- FullCPU102passed5conditionalskips, latestXML`reports/native_actuation_final_regression_20261022.xml`; finalsmallabsolute-speedgoalguard3focusedtests pass; cp37bridge/native/actuation/probecompile passes. Publicationpreflight255filesnofindingsbeforelasthandoffedit; publicationandindependentcheckoutverification pending atthissnapshot. See`reports/native_pose_20261022.md`. Rawnewnativeassetslocalonly; existingprereleaseassetsunchanged.
- Next substantial work: usemeasuredactuatorresponse andrecordedlag toredesign/calibrate longitudinalcontrolwithout maskingbrakespikes, preserveurgentbrakingandpairedfixedreference. Diagnosev17registered2M/3M results, retainingunseen/collisionfailures. StillneedhealthyB1/fivepolicyseeds/strongB3RLdecisionvalue/B2frequencysharedshield/nativePPOintegration/fresh120CARLAconfirmation. No hardblock; do notmarkgoalcomplete.

Historical snapshots below retain their original scope and may describe processes now stopped.

# Active research handoff — 2026-10-22 19:19 local

**Latest update superseding older snapshots:** prior goal turn was progress; this turn adds actual CARLA steering measurement, native-map survey, raw-frame verification/plotting and the2Mlearning endpoint. Full objective remains active/incomplete, no hard blocker or subagents.

- Continuation `20261022T110104Z_progress_v16_continuation` reached2,000,379physical steps/783phases/25,904optimizer steps, exactreloadtrue,0/10. Six deadlines/four collisions (narrow andunseen), neither empty case enters0.8mgoal tolerance (minimum1.196/1.313m). `terminal_audit_2000000.json` preserves per-frame evidence. It is proceeding to the predeclared3M, last2,181,199steps at19:19. Worker25592/supervisor13284 remain the owned handles; revalidate OS. v13B1hold1 stilllive, last737k at19:15, deadline19:38; do not duplicate or restart.
- New actual CARLA audit COMPLETE: `runs/20261022T111210Z_carla_steering_audit`, pointer`runs/carla_steering_audit.json`.22phases,3,520control/4,201world ticks before finalcleanup, zero observedcollisionevents. Probe requested speeds0.5/1.5m/s, measured0.290–0.316/1.445–1.510. Existingsteer mapping residual≤.01346rad atsampledpolicyinputs,≤.03250fullsampledrange; oldmappingunchanged. Allrawframe/sensor/timestamp/summarychecks pass, figure visuallyinspected. This is rawactuator diagnosis, neverPPO/MPCEvidence.
- First audit attempt111101Z failedembeddedimport beforeCARLAmutation/zero ticks; preserved. Newfrozen111210Z fixespackagequalifiedimport and reuses ownedemptyserver. Actuatorscleaned,stopprobeempty, ownedshipping46056 stoppedwithreceipt. NoCARLAserver remains. Native read-onlysurvey examined824starts, rejected479heading/curvature+7footprint, retained334possiblyoverlapping24mcurves. No nativecurve drivingyet; noindependentfinalconditionsclaim. See `reports/steering_native_20261022.md`.
- Newsource:carla_steering_audit.py,survey_native_curves37.py,plot_steering_audit.py,oneanalysisunit test.93passed5conditional skips in`reports/steering_regression_20261022.xml`; priorindependentcheckoutat e0b2a0a had92/5. Publication check242filesnofindings beforefinaldocs; inspectactualGit for latercommit/push. AllrawCARLAauditdata local, notyetreleaseasset.
- Next substantive work: inspect3Mendpoint and v13completion; diagnose learning retention/parking ifflatstillfails. Use native-route candidates for actual fixed-reference CARLAcontrol and then sharedscene/road-boundary integration; geometrysurveyalone isnotnativecurvevalidation. Do not modifyrunningfrozen training. Fullfive-seedfaircomparisons, RL-vsstrongB3value, B2/sharedshield/controlablations andfresh120conditions stillneeded.

**Newest authoritative state:** the previous summary turn was a verified wait/read-only endpoint audit; this turn implements/tests a generic bounded continuation, completes physical-frame failure audits and launches actual additional PPO learning. Research acceptance remains incomplete; no hard blocker, no agents. Read the user objective attachment on every continuation. The paragraphs below are historical snapshots.

- v15 pooled B1 is COMPLETE1,001,990steps/392phases/13,016optimizer steps,0/10. v16 progress-reward B1 is COMPLETE1,001,373/392/14,224,0/10. Both exit0/reload_exact. v16's final100training episodes have2successes; fulltraining139/1436. Do not restart either completed batch.
- Their terminal audits now exist in each job as `terminal_audit_1000000.json`. v16 empty stops2.67–2.77m short and brakes for the last5s; none of its ten cases enters goal-position tolerance. Eight deadlines, two unseen collisions. v15 has six stationary failures. No altered success criteria or excluded failures.
- Actual continuation `runs/20261022T110104Z_progress_v16_continuation`, pointer `runs/progress_v16_continuation.json`. SAMEv16 source/config/RNG/optimizer/env/norm/seed, cumulative2M/3M, inherited1,001,373steps. Deadline~21:01local, oneCPUthread. OS-verified worker25592/supervisor13284 (launcher33904/43192) created~19:01; phase399already14,383optimizer steps/actorupdateL2.2154 at1,019,292steps. Revalidate identities/log growth, never restart on polling timeout. Noautomaticretries;12GiBdisk/4GiBRAM/85CGPU/300sno-progress guards.
- v13 B1 hold1 remains in its original batch, OS-confirmedworker35396/supervisor22032, last663,153steps at19:03, deadline~19:38. CompletedH/H0remain8/10and6/10. No other training or CARLA was started.
- New `scripts/continue_development.py` only permits fully successful completed parents and strictly higher cumulative milestones, holding parent supervisor/worker leases while checking/copying. It is not a recovery tool for censored/incomplete/failed batches. New `scripts/audit_terminal_failures.py` analyzes saved physical frames without new execution. Publication/tests recorded below or in the next update; inspect Git rather than relying on historical pending statements.
- Next: inspect2M/3Mcontinuation outcomes and v13endstate, retain failures, decide the next hypothesis from actual progress/parking/safety evidence. All matched-reward retraining, healthyB1, H-vsstrongB3, B2/sharedshield/control comparisons, nativeCARLAroads, five independent policies and fresh120conditionconfirmation requirements remain. Do not treat this continuation or a publication as research completion.

Full current regression92passed5conditional skips, XML `reports/continuation_regression_20261022.xml`. New functionality has real continuation evidence in addition to filesystem unit fixtures. Last published base0f6bb17; inspect actualHEAD/remote for this turn's subsequent commit. No newreleaseassetisneeded for these failed developmental policies.

**Authoritative update, 18:44 local:** previous goal turn was progress; this continuation also adds diagnostics, actor code, tests and actual training. Read `reports/flat_diagnosis_20261022.md` before the older snapshots below. Full goal remains active and incomplete; no hard blocker and no agents used.

- v13 H is COMPLETE at1,001,775physical steps/397phases/14,390optimizer steps,8/10. H0 complete1,002,283/395/15,100,6/10. Supervisor confirms both exit0. v13 B1 hold1 remains live (actual35396, supervisor22032, created17:38), last~535k; deadline~19:38. Revalidate identity. Do not restart completed H/H0 or rely on recycled PIDs.
- v14 hold10 is COMPLETE at1,000,149steps/100,096actions/391phases/13,937optimizer steps,0/10 with zero progress in every endpoint case. Five training successes among1,218episodes, actorL2change7.5002, exactreloadtrue, supervisorexit0. Preserve this failed frequency control.
- New actual v15 pooled-actor batch: `runs/20261022T103301Z_flat_pool_v15_development`, pointer `runs/flat_pool_development.json`; supervisor35788(launcher5904), worker41256(launcher2404), created18:34. Last~513k,100k/300k0/10. Only actor architecture differs from v14; same605inputs/critic/reward/hold10/curriculum/1Mbudget. Shared candidate encoder and mean/max pooling keep direct controls. Deadline~20:33. Test actual parameter changes in all branches and reload passed.
- New actual v16 reward batch: `runs/20261022T104130Z_progress_v16_development`, pointer `runs/progress_development.json`; supervisor18212(launcher9000), worker16208(launcher39344), created18:41:55. Last~102k; phase21 already811optimizer steps/actorupdateL2 .1691. Only signed actual-progress weight0→.25 differs from v14 MLP. ExistingPBRS40 retained. This changes the base objective, so do NOT compare against old-reward H as hierarchy proof; shared-reward H/H0/B2/B3 retraining is required if retained. Deadline~20:41, standard guards/noautoretry. Inspect milestone files, do not infer success from updates.
- Read-only inference/feature audits are COMPLETE in `runs/flat_feature_audit_20261022`; two full23m empty seeds per model, original policies execute, input-zero probes never execute. Old351-input v9 succeeds2/2; v13/v14 fail2/2. No core clipping; first-layer saturation4.23/17.22/10.52percent. Late v13 speed0 with100percentbraking; v14 speed.040m/s. Figure rebuilt/visuallychecked. These are not causal architecture proofs or new independent seeds.
- Offline reward audit reconstructs all38recorded physical returns within1e-6. v14 partial10.078m deadline return12.992 versusstationary13.010. Correct terminal-zeroPBRS does not preserve partial progress among same-duration failures. Newreward adds2.349 to the former,0 to latter; those are reward-only counterfactuals, not measured new performance. Tests cover whole-route rankings and event margin;79passed5skipped fullregression.
- Last published SHA before this turn:87fb09886104da60f144f6cefb893dd6b7fc7698. This turn's actor/diagnostic/reward-initializer/docs need publication checks and commit/push; inspect Git for subsequent completion. Previous camera asset/pre-release/source-replay verified; no CARLA server active this turn. Current training source snapshots stay immutable when live code changes.

Continue existing runs and make the next evidence-driven change; do not open final confirmation while B1/feedback/generalization remain unresolved. All five-seed/fresh120-caseCARLA/strongB3/B2/frequency/sharedshield/control/native-road requirements still apply. An expiring batch is not project completion; only create an explicit cumulative continuation after confirmed terminal state, preserving source/RNG/optimizer/counters.

18:46 milestone update: v16~317k with79training successes/462episodes;100k evaluation2/10 (emptybothsuccess),300k0/10 (empty~21mdeadline, static/narrow/dynamic/unseen collisions). Preserve this regression and do not call the early empty success convergence. v15~633k/907episodes has zero training successes and100k/300k0/10; v13 B1~552k/672episodes has one training success. All three remain active. Full tests79passed5skipped; publication preflight233files/no findings. The flat diagnosis report and actor/reward initializer changes await the current turn's commit/push; verify actual Git status on continuation.

Additional frozen v15/300k two-case empty replay completed: bothfail (6.800/6.590m), no global-context first-layer saturation, extra-zero action sensitivityL2 .445. That context layer excludes candidate tokens, so its saturation is not directly identical to the dense first-layer metric. This limits the representation-only hypothesis; do not claim its architecture has fixed B1. Evidence `runs/flat_feature_audit_20261022/v15_pool_hold10`.

**Latest authoritative continuation: 2026-10-22, 18:10 local.** The sections below are historical snapshots; do not treat their old PIDs as live. Read `reports/camera_staged_20261022.md` first for completed v11/v12/CARLA outcomes and camera evidence. The preceding summary-only turn was a verified wait (OS-confirmed v13 processes); this continuation makes code, real-video verification, packaging and reporting progress. The overall goal remains active and incomplete.

Current live batch: `runs/20261022T093852Z_staged_v13_development`, pointer `runs/staged_development.json`. Actual OS processes confirmed this continuation: supervisor 22032, H 7360, H0 41352, B1 35396, all created around 17:38. Revalidate before acting. Common first 300k full-length empty training, then mixed tasks, all inside a 1M per-policy physical budget; single seed 11, frozen source, two-hour deadline approximately 19:38. H 100k=10/10 and 300k=8/10; H0=6/10 at both; B1 100k=0/10, first training success observed. Keep these regressions; do not select an early peak as final confirmation. Resume only with the existing `development_watchdog.py --batch` entry after inspecting its live lease/status, never duplicate workers.

Latest camera batch `runs/20261022T095316Z_carla_camera_evidence` is complete: static7301 success and curve7301 deadline, 1261 control/1322 world ticks, no unresolved infrastructure, frozen H25k CARLA-adapted model. Owned server stopped with receipt. Landscape/orange-ego recordings encoded and decoded: 362 frames/18.1s and 901 frames/45.05s; contact sheet visually checked. Three earlier camera infrastructure failures and the portrait capture remain retained. No active CARLA server is needed by v13. Package `runs/prerelease_20261022/candidate_v12_H25k_camera_examples.zip` has 145 hashed files and SHA256 `c24ad7dec570f7becaae46f0014ce5320f5e9a2ae87849416d327d5f887aac44`; verified exact model load using the project and independent clean-checkout environments. Check release metadata before claiming upload. `scripts/verify_camera_asset.py` is CPU/package verification, not CARLA replay. Current camera/curriculum changes were uncommitted at this paragraph's creation; inspect Git for subsequent publication.

Next actual research work: finish the existing v13 milestones; diagnose B1's speed/parking/learning failure from real traces; retain H regression and unseen failures; implement only supported changes as new versions. Healthy B1, five independent policies, fresh final 120-condition CARLA test, strong B3 decision-value comparison, B2/frequency/shared-shield/control comparisons and native-road curves remain incomplete. Follow the full active objective, not just this next batch.

Publication and fresh replay update, 18:16: code commit `02f75a8148fbae1fb1cbcc9ac00a4f22b4c5f5ac` is pushed and remote SHA matched. The 2,218,569-byte camera ZIP is actually uploaded to the existing development pre-release; GitHub digest and downloaded hash match. No tag replacement. New clean `camera_checkout` passes 74 tests/5 skips, downloaded-model reload and video decoding, then actual CARLA static success/curve deadline replay (1242 control/1303 world ticks) with complete new camera recordings. Detailed progress differs, so only categorical outcomes match. Server shipping PID22084 was verified empty and stopped; no remaining CARLA processes at inspection. Source/asset/replay details are in `docs/release_status.md`.

v13 B1 reached its 300k checkpoint and is evaluating the frozen full-task cases. A separate bounded, read-only 32-episode inference diagnostic was started using that checkpoint and the batch's frozen source: `analysis/B1_300k_empty_inference`, full23m empty task, train/validation starts and deterministic/stochastic gSDE inference. It does not update the model or count as new training. Inspect its process/output before retrying. H/H0 and B1 remain under the existing two-hour supervisor.

18:22 superseding update: the inference diagnostic is COMPLETE, all four groups0/8, all deadlines; deterministic mean progress7.121m/jerkRMS11.362, stochastic9.655m/14.134. Train/validation empty initializations are identical and must not be counted as32independent scene conditions. B1 full-task300k is0/10, including two unseen collisions. H0 is COMPLETE at1,002,283physical steps/395phases/15,100optimizer steps and6/10; its original Python process exited0. PID41352 was reused by an unrelated updater, which was left alone. H and B1 continue.

New actual training: `runs/20261022T102056Z_flat_frequency_v14_development`, pointer `runs/flat_frequency_development.json`, supervisor launcher37512 at18:22:22. Verify actual child PIDs/log advancement. Fresh B1 hold10/seed11, same v13 inputs/reward/vehicle/common300k-empty curriculum/1M totalphysical budget; only flat_hold1→10 and SDE resample10→1 (same0.5sphysicalnoise) differ. Single CPU worker, deadline~20:20, standard resource guards/no automatic retries. Source/option comparison saved in `comparison_verified.json`. Full regression76passed5skips. Continue this required frequency diagnostic, do not stop at script generation or claim training success before optimizer logs exist.

Verified v14 live state at18:24: actual supervisor30020, worker27080, launcher37560. At136,413physical steps/53PPOphases the heartbeat was fresh; earlier phase27 recorded846optimizer steps and actor-updateL2 .4498. The100k checkpoint is102,103physical steps and0/10, so this is real learning activity without demonstrated task success. v13 H~906k and B1~372k were still live; H0 remains completed. The v13 report was rebuilt from60grouped milestone rows with153frozen source hashes verified. Revalidate OS identity before any action.

Goal remains active and incomplete. Read the current user objective attachment before continuing. Existing public repository `Leo-7GOAT/Intelligent-driving-system`, branch `feature/physical-hierarchical-ppo`, base8142224. Development commitbbaf303 pushed to the existing feature branch and remote SHA verified. Do not modify other projects, reset/reinitialize Git, erase failures, or reuse inspected final conditions as unseen confirmation. No subagents authorized. Historical handoff retained in `reports/handoff_before_v6_20261022.md`.

## Live work (revalidate OS processes before acting)

- H/H0 continuation: `20261022T051509Z_hierarchy_v8_continuation`. Pointer `runs/hierarchy_continuation.json`. Both continue their completed200k checkpoints using EXACT parent frozen source, RNG, optimizer, environment state and normalization. Cumulative targets300k/1M, two-hour window from13:15. Actual workers8676/20476 were alive and optimizing at230k+, phase90. Check current identities, never assume these PIDs remain valid.
- B1 distance diagnosis: `20261022T051509Z_distance_v9_development`. Pointer `runs/distance_development.json`. Fresh seed11 fixed23m versus uniformly sampled6–23m, empty training, common45s deadline/beta40 PBRS/gSDE settings, targets100k/300k/1M. Actual workers1892/24656 were alive and optimizing at76k+, phase29/30. Four CPU workers total across both pairs, each Torch single-thread, no GPU training;12 logical CPUs, approximately43GiB RAM and72GiB disk available at launch. Independent supervisor leases, no automatic failure retry.
- CARLA transfer `20261022T051716Z_carla_v8_transfer` is COMPLETE, both learned H/H0 frozen200k policies score2/6 (one curve7302 and one dynamic7302), four deadlines each, zero observed collision/bounds, zero infrastructure failures. H4793 world ticks includingreset/4608control; H0 4861/4676. Normalization frozen. NO CARLA PPO updates in this transfer. Server launcher37724/shipping11520 was confirmed empty and stopped; receipt `infrastructure/server_stop.json`. Exec session11430 exited0, no remaining bridge.
- At13:21 H/H0 continuation both300k evaluations8/10; actual training313–325k and progressing. B1 full23 at300k gets2/8 pooled distance probes, uniform6–23 gets0/8. Inspect per-distance data before calling baseline health passed. Full1M budgets still running.

## Completed evidence

- v6 B1 two reward variants each1,000,960 physical steps: both8/8 on6m parking; PBRS40 model0/8 on23m/45s. Full-task baseline health NOT established. Retain PBRS40; explicit task-progress reward not needed by current evidence.
- v7 H/H0 gracefully superseded at12,790/10,225 physical steps. Partial logs/checkpoints retained, no performance conclusion.
- v8 original `20261022T050532Z_hierarchy_v8_development`: both supervisor exits0 and all milestones complete. At100k both6/10; at200k H8/10, H0 6/10. H completes empty/static/narrow/dynamic, H0 fails static; both fail unseen. H dynamic intervention~.145 vsH0~.025, so do not equate higher completion with uniformly better safety/efficiency. One seed, two near-identical conditions per family; no confidence interval or independent confirmation claim. Source/model verified48grouped rows, plot and report in `analysis/`.
- Actual CARLA B3 low-level diagnostic in v7 batch:3/6 before terminal envelope,4/6 after, with degraded static7302 retained. These are rules, not learned PPO. Server was confirmed empty and stopped before the new transfer.
- Regression:66 passed5 skipped in `reports/v9_transfer_regression_20261022.xml`, including calibration-preserves-policy-config regression. Raw XML ignored because local personal paths. Old failed tests retained.

## Implemented this turn

- Optional training-only `training_route_range`, deterministic seeded distance sampling; validation/test remain full-length unless explicit diagnostic override. Mutually exclusive with fixed-route/curriculum. Applies equally to architectures if used in later fair comparisons.
- Generic development evaluator handles B1 continuous actions and explicit route cases; closes episode envs; per-job PPO options. Running v8 continuation uses old frozen source, unaffected.
- `initialize_distance_development.py`, `initialize_development_continuation.py`, and `summarize_paired_development.py`. Continuation is a new batch with copied final checkpoint and inherited-step provenance, not hidden warm start or an extra independent seed.
- Fixed CARLA evaluator/fine-tuner dropping new policy observation/action/planner settings. `calibrated_config(...,source_config=...)` now preserves all source fields except calibrated vehicle; regression covers H/B1/B2 and shape450. Old frozen historical experiments unchanged.
- `carla_feedback_transfer.py` prepares/provenance-checks and evaluates learnedH/H0, labels calibrated low-level frozen policy, no rule replacement.

## Next executable actions

1. Transfer completed and server stopped. Inspect `action_failure_audit.json` for H/H0: failed cases repeatedly select the same candidates, producing search_budget or start_or_goal_body_collision_or_bounds. This is a policy/candidate/domain-transfer failure, not evidence of collision safety or successful CARLA learning. Both policies need actual adaptation after full lightweight milestones; do not erase the frozen-transfer negatives.
2. Inspect both training supervisor files plus actual OS PIDs, current optimization logs and milestone results. Continue their bounded windows; no resetting completed jobs.
3. Rebuild either new pair with `python scripts/summarize_paired_development.py --batch <batch>` after completed evaluation markers appear. Parent H/H0 milestones remain in original batch; continuation counts include parent training.
4. Update acceptance/README/handoff with final CARLA transfer outcomes and latest learning. Resolve unseen failure and healthy B1 before final protocol; don't prematurely freeze/publicize superiority.
5. Remaining full goal: five independent policies and sufficient formal comparisons, healthy matchedB1, PPO-vsB3 interaction value, CARLA true adaptation and120unique final conditions, native-road/actuator validation, real success/failure videos, clean-checkout reproduction, release hygiene, truthful bilingual docs/CV, real GitHub commit/push/tag. Research acceptance remains NOT met.

Recovery command for a stopped nonfailed bounded pair: `python scripts/development_watchdog.py --batch <exactbatch>`. Existing live lease prevents duplicates; STOP/expired window/failures require diagnosis, never blindly remove them. A new continuation window must retain source and cumulative counters explicitly.

Latest verified OS process/heartbeat snapshot13:23: H378,749 physical steps (149optimization phases); H0 359,981 (141); B1 full23 491,293 (191); B1 uniform500,061 (195). All four workers live, both supervisors without STOP, GPU35C. Do not treat these numbers/PIDs as permanently current. Prior goal turn was a verified live-process status inspection; this turn made code/test/actual training and CARLA evidence progress. Detailed development report: `reports/development_20261022.md`.


## Subsequent continuation (2026-10-22, publication and real adaptation)

Previous turn is classified as progress. This turn also makes concrete code/test/training/publication progress. B1 distance jobs both completed1,000,960 physical steps: fixed23m2/2 on23m but0/6 at shorter probe lengths; uniform6–23m0/8. Keep failures. H/H0 lightweight continuation is still running toward1M; inspect OS PIDs and current markers.

CARLA adaptation batch `20261022T052635Z_carla_v10_adaptation`, pointer `runs/carla_adaptation_development.json`: source-frozen paired H/H0 from the SAME200k models used in the transfer diagnostic. Interleaved cumulative25k/100k actual world-step budgets, reset cost included; seeded training conditions0..9; six existing development evaluation conditions. Both25k milestones complete at26,598/26,558 ticks and2/6 each, no improvement yet. H continued beyond69k with real optimizer updates; H0 waits its turn afterH100k. Actual coordinator46440, supervisor52864, bridge34208 at last verification; recheck identities. Server launcher10328 owned by this batch. Frozen bridge uses NAVSTACK_CARLA_PYTHON to reference the existing isolated runtime, while its Python source stays frozen. Worker finally attempts identity-checked empty-server shutdown; inspect cleanup receipt rather than assuming success. Do not launch another CARLA owner while this one is live.

Development asset `runs/prerelease_20261022/development_v8_seed11.zip`, SHA256ad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c.152files, two models, full20development evaluations, frozen source, optimizer/training-episode summaries. Verified all20replays both locally and in a fresh clone/newvenv. Fresh clone is `runs/prerelease_20261022/clean_checkout`; independent pip installation/pipcheck/66tests5skips succeeded. PureEOL issue in originaldialog_parser.py index is normalized in follow-up commit, with no semanticdiff. Original12snapshotfiles retain exactSHA256 through .gitattributes exceptions. No secrets/private user paths in publication candidates or asset; generic scanner path literals reviewed as expected code.

Git push needed per-command `-c http.proxy=http://127.0.0.1:7890`, matching enabled Windows proxy; directGit timed out, ghAPI worked. No global config changed. Authentication remainsLeo-7GOAT. Pre-release intendedtagv0.2.0-dev.20261022; inspect realGitHub status before claiming it exists. Research acceptance remains incomplete even after publication.


## Latest authoritative update (2026-10-22 13:51 local)

Publication is real: GitHub development pre-releasev0.2.0-dev.20261022 exists at9bcaa79d34a6985a3d12764d3bf01383c7f06d83,3220328-byte model/data ZIP uploaded, downloadedhash matchesad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c. Fresh tag clone `runs/prerelease_20261022/release_checkout` is clean;66tests5skips and20replays allpass using the freshly installed pinned environment from the first clone. NoCIclaim. Firstclone retainedwithEOLdirtyfile; newtagcloneverifiedcorrect. Pre-release doesNOTsatisfyresearchgates.

Live batches:
- H/H0 continuation (`runs/hierarchy_continuation.json`) at~897k/916k respectively, still training to1M. ActualPIDs8676/20476; revalidate.
- CARLA adaptation (`runs/carla_adaptation_development.json`): H finished100,890world ticks,38optimization phases,3/6dev (bothstaticfail;curve7301fail;dynamicbothsuccess;curve7302success). H0~79k,30phases,stilltraining; coordinator46440, latestbridge26504. Inspect actualprocesses and finalcleanupreceipt; neverstartanotherCARLAowner whilealive.
- B1 mixed feedback baseline (`runs/matched_flat_development.json`): batch20261022T054701Z_matched_flat_v11_development, actualworker14136, supervisor35164; >23kphysical steps,9realoptimization phases. SameHtask/reward/vehicle450obs/candidatefeedback; directcontrols, noMPCactionreplacement. Onefromscratchseed11; physical1Mtarget/twohourwindow. Geometryfeatures atflatfrequencyareadditionalcompute andmaylimitbudget; reportactualsavedsteps,don'thidecensoring.
- B1 distance pair terminal at1,000,960each; fixed23m2/2full-distance,0/6shorter; uniform0/8. NotfairtomixedHuntilthenewmixedbaselinefinishes.

New offline audit `scripts/audit_recorded_carla_candidates.py` completed16recordedinitial/finalstates withfrozen sources. Everyfailedfinalstatehas3–12spatiallyplannableunique candidate targets. This isnotCARLAexecution or dynamic-safety evidence. It rulesout universal lackof candidate paths assolefailurecause. PPOselection/representation/sparselearningremainissues; H staticCARLAtraininghad0successesinrecent18episodes evenafter~69k. Current100kpair willgivefurtherfailure evidence beforechangingobjectiveorpolicyarchitecture. Rewardprogress_weight alreadyexists and isexplicitlynonPBRS; anyfuturetrial must beversioned,sharedwithrelevantbaselines andnotmixedwitholdcheckpoints/results.

Next: inspect/finishrunningmilestones, rebuildtables; confirmCARLAcleanup; continuefailure-drivenlearningandfairbaselines, independentlyseededformalexperiments, native-road/controlvalidation, videos, completeCARLArelease replay andfinalresearchacceptance. Do notmistake uploadedpre-release forgoalcompletion. No external login blocker.

## Authoritative continuation — 2026-10-22 14:18 local

The previous summary turn was a verified wait: OS-confirmed live B1 worker plus fresh checkpoint inspection. This turn is progress: implemented/tested a new candidate policy, ran B3 and actual CARLA comparisons, and started actual new PPO updates. Full goal remains active and incomplete; no hard blocker. Read the full user objective attachment before continuing. No agents were used. Existing processes in other projects remain untouched.

Completed facts superseding old live entries:

- v8 H/H0 lightweight continuation both complete:1,000,551/1,001,774 physical steps,396 optimization phases each,8/10 versus9/10. Both exact reloads pass. This contradicts a stable feedback-benefit claim.
- v9 B1 distance pair both finish1,000,960 physical steps: fixed23m2/8 pooled probes (2/2 at23m,0/6 shorter), uniform6–23m0/8.
- v10 actual CARLA adaptation of the original200k models completes100,890/100,985 world ticks (control96,955/96,895),38 phases/1,520 optimizer steps each,3/6 each. Server was confirmed empty and stopped. This is distinct from the new candidate run below.
- Real release and downloaded-model CPU reproduction remain verified. Fresh release checkout also reproduces static7301 deadline and dynamic7302 success in actual CARLA; retained initial startup timeout, bounded retry,1,369 control/1,430 world ticks and owned-empty-server shutdown. `carla_replay_support.zip` is actually uploaded; GitHub digest f28ce6c76a5ad45584a2f971c60a5cd220567362712c8a1198c19667009f0657. Existing model asset/tag unchanged.
- Strong B3 on all ten identical v8 lightweight development cases gives10/10,3,298 evaluation physics ticks,18.439s total. Evidence: `runs/20261022T060649Z_candidate_v12_development/reference_v8_B3`. Uses verified v8 source, includes rule selection/search cost. It also succeeds on both unseen cases failed by H; retain this negative control.

New version and live handles (revalidate OS, never assume PIDs persist):

1. Candidate v12 batch `runs/20261022T060649Z_candidate_v12_development`, pointer `runs/candidate_development.json`. H/H0 shared candidate scorer from scratch, identical605-input architecture, feedback versus zero-feedback ablation. Unchanged reward/low level/actions/train distribution. Candidate geometry scales are fixed; global state still normalized. No masks, teacher, auxiliary labels or action replacement. Milestones0/25k/100k/300k/1M, two-hour window until16:06 local. Actual workers49676/53200; supervisor12336 (launcher52732). Last heartbeat~179k/240k,70/94 phases. At25k H0/10,H0 4/10; at100k H8/10,H0 6/10. Compare later budgets before drawing any conclusion. v12/v8 differs in representation and architecture together; H/H0 inside v12 isolates feedback.
2. v11 matched450-input B1 remains live: `runs/20261022T054701Z_matched_flat_v11_development`, pointer `runs/matched_flat_development.json`, actualworker14136/supervisor35164. At100k and300k0/10. Continues toward1M within its15:47 window. It is not a matched605-input final comparator for v12.
3. v12 fixed100k frozen-policy CARLA transfer COMPLETE: `runs/20261022T061420Z_carla_feedback_transfer`, pointer `runs/carla_transfer_development.json`. H4/6 (static7302,curve7302,dynamicboth), H0 3/6 (curve7302,dynamicboth). H3,459 world/3,274control,H0 4,007/3,822; zero PPO updates, normalization frozen, no evaluation infra failures, no collision/bounds episodes. Source H101,923/H0 101,965lightweight steps. Startup first15s probe timed out, second connected; no restart on observation timeout. Owned empty server stopped. Do not call low-level calibrated transfer zero-adaptation system performance.
4. NEW paired CARLA adaptation `runs/20261022T061639Z_carla_feedback_adaptation`, pointer `runs/carla_adaptation_development.json`. Same100k pair used above, interleaved25k/100k actual world ticks per policy including resets. Existing isolated client and calibration reused. Actual supervisor10072 (launcher12136), coordinator45856 (launcher9108), bridge42384, server launcher9932. H already7,973world ticks,3phases/120optimizer steps, actor-update L2 .712 atlastphase; H0 waits its turn. Two-hourwindow ends16:16. Verify logs and finally cleanup receipt; do not start another CARLA owner while live.

Resource snapshot:12 logical CPUs,~74.9GB free disk,~41GiB available RAM, three lightweight workers plus one CARLA training worker, Torch single-thread each; GPU38C lastsupervisor. Watchdogs have no automatic failed-job retry,300s no-progress guard,12GiB disk guard,4GiB memory guard,85C GPU guard. An observation timeout alone does not mean training stopped.

Validation: full regression70passed5conditional skips. New tests cover common candidate geometry across methods, exact H0 ablation, candidate-score permutation, unmasked probabilities, fixed normalization, real PPO updates, checkpoint/resume and custom class reload; CARLA calibration preserves605input settings. New initializer has a standalone default requiring no historical local data; verified that its task/config/cases exactly equal the existing v8 reference. CLI transfer now selects fixed milestone checkpoints from generic run_group and supports a separate measured calibration batch. `publication_preflight.py` scans Git publication candidates without printing possible secret values;212files/no findings atinitialcheck.

Next concrete work: inspect completed300k/1M candidate milestones and B1; finish new actual CARLA adaptation, retain all negatives, compare per-family progress/interventions. New pair learning is not independent confirmation; do not claim five seeds or healthy605-input B1. Fair605-input baseline, richer interaction/generalization training, native-road/control calibration, full five-seed formal comparisons, fresh120-condition CARLA confirmation and real success/failure videos still remain. Existing evaluated conditions are development/historical only. Recover only after confirmed terminal/missing owner via `.venv/Scripts/python.exe scripts/development_watchdog.py --batch <exactbatch>`; do not clearSTOP/failure blindly or duplicate live workers.
