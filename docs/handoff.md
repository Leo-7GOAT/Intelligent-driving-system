# Current handoff — 2026-10-31 20:45 local

AUTHORITATIVE21:18update;all older live statuses below are historical:
Original B1driver26916FAILED at seed22initial gate:6/12,5bounds/1collision,
4404audited controls/12initial actions; basic-motion safety gate FAILED.
No original22/33/44/55PPOwas started. Driver26916,dependent B238164and shared10408
have exited,all failures retained. Their status.json files are STALE.

ACTUALLY LIVE B2execution:
runs/B2_five_seed_PPO_execution_20261031_v3_independent_after_B1_gate_failure,
Python32792,tool session85650,worker15416/guard32664. Seed11original batch
20261031T104032928978Z_native_planner_training is truly training (>8686controls
and27optimization phases already observed21:16). Original five B2initializers
and20480-control recipe unchanged. New tracked runner explicitly verifies the
closed failed B1predecessor,normal native cleanup and all old process deaths;
it retains the B1failure and does NOT claim B1completion. All five B2PPOseeds
will continue/audit sequentially if the native lifecycle succeeds.

New B1ALL-FIVE actor maturation is PREPARED,NOT DRIVEN:
fixed plan protocols/native_B1_all_five_actor_maturation_20261031.json,
prep runs/B1_all_five_actor_maturation_preparation_20261031. All five original
zero-PPOactors get1024extra TRAIN-only fits each at5e-5 on the same3496rows,
including corrective rows. No seed selection,no broader/validation labels.
Empty buffers aligned1280/batch320,critic/norm/RNG preserved. All5actual
frozen-runtime restores and launcher ValidateOnly checks PASS;five distinct
policy tensor hashes. Report native_B1_all_five_actor_maturation_preparation_20261031.json.
New ACTUALLY LIVE waiting queue:
runs/B1_all_five_matured_PPO_execution_20261031,Python60572,tool session5248,
retained read-only handle to B2driver32792. Frozen310sourcefiles,workspace
bindings immutable. It drives/audits ALL5initial gates before ANYPPO. Any failed
basic gate is retained;all5are still audited,then no PPOstarts. If all pass,
all5get20480controls,LR3e-6,1280rollout,batch320,std-4,KL.01,fixed reward.05,
frozenobs. Seed11already has fixed reward normalizer,so does not promote it again;
the other4are promoted once. Root prepare_native_supervised_ppo.py stays bound.
New initializers are11=131044921585,22=131138220112,33=131235976652,
44=131332227703,55=131429935122,all20261031TZ_native_B1_actor_maturation;
use actual prep status for full names. Shared-brake OLD queue is dead;new paired
deployment queue must wait for healthy completed B1policies and reuse Hmodels.

Public capsules44/55/fresh22H0are ALL verified:125443controls/60initial actions
and all3exact endpoint figures. New22HTTP408upload failure retained;14correct
objects hash-checked,onlymissing15/16/17parts+manifest uploaded,then actual
unauthenticated download/independent reconstruction passed. Completion receipt
native_seed22_fresh_H0_canonical_public_verification_20261031.json. Publisher32584
has exited failed historically;repair session20183completed successfully.
Independent checkout can now advance from666fa72;no publishing job uses it.
Latest pusheda19afdbhosted workflow41203628741SUCCESS. New14guard/cohort tests
pass;new sources/receipts need commit,push,independent verification. All research
thresholds,healthy baseline,feedback benefit,PPO-vs-rule,frequency,light and fresh
120-final conditions remain UNMET. Keep goal ACTIVE;do not restart old queues.

21:00operational update: the corrected actor-only replay/added36probes both
COMPLETE, raw controls11474/11086 and24initial actions each verified; scores
8/24versus5/24,unsafe12/24versus14/24. Zero PPOupdates. Additional data fit
improves on its TRAIN rows but worsens old-row fit; source-bound all-row report
native_B1_training_view_fit_regression_diagnosis_20261031.json preserves this
negative result. Do not adopt the5/24actor as a healthy B1baseline or refit it
again merely because TRAIN loss improved. Native B1driver26916now runs seed22
initializer gate081944045029,worker52148/guard51464;case5observed20:59.
B2/shared drivers38164/10408remain waiting and have not driven.

Source666fa72is pushed;hosted workflow41202408901SUCCESS;30related tests pass
in root and the actual separate checkout. Independent checkout MUST stay666fa72
until publisher finishes: active publication Python32584,local execution
runs/remaining_feedback_publication_execution_20261031,tool session36472.
Three complete archives are already packaged. Seed44is FULLY publicly verified:
667files/66864585bytes/32parts,SHA d8971e84b3e753441ba6d3c28f8ad236565e0a69fdf5fab869c2247e9215dffb,
50308controls/24actions/exactfigure. Receipt native_seed44_canonical_public_verification_20261031.json.
Seed55is uploading;fresh22H0next. Worker has a finite one-hour budget and no
automatic retry. It writes individual public receipts and status/failure.
New archives intentionally lack packaged endpoint figures;use the committed
reports/figures/native_seedXX_canonical_endpoints_20261031.png with
reproduce_native_asset.py --reference-figure. Publisher already did this for44.
CompletedH22in its failedH0sibling batch still lacks a portable public capsule;
do not fake batch completion or discard its failure to force generic packaging.

New H/B3source-bound report verifies identical resolved config and8physical
source hashes:both24/24;H8089/B38033controls,H390/B3380interventions.
No PPOsuccess/time/brake superiority is established. All-five feedback figure
reconstructs from the committed report with script plot_native_feedback_cohort.py;
render inspected,all-five outcomes and the interval crossing zero displayed.

Goal ACTIVE/UNMET. All five canonical trained H/H0pairs are complete: seeds
11/22/33/44/55 score H12/12/10/12/12 and H011/11/12/11/12 respectively.
Totals58/60versus57/60,zero unsafe,on12repeated development conditions.
Mean paired seed difference+1.67pp,95% t interval[-11.82,+15.16]pp,p=1.0.
Report native_feedback_five_seed_development_cohort_20261031.json and tracked
scripts/analyze_native_feedback_cohort.py reconstruct all five pairs. H0seed22
uses fresh batch20261031T093208454506Z_native_H0_canonical_budget_replica;
the interrupted older branch remains retained/charged separately.
Canonical queue47784has COMPLETED; do not restart it.

B3batch20261031T082841Z_native_rule COMPLETE:24/24,8033audited controls.
Same broader development conditions as H24/24; no PPOsuccess advantage.

The former B1driver68480failed when the replay actor-only validation inherited
freeze_observation_statistics=True, which is reserved for learning continuations.
Original replaybatch093757339917has ZERO controls; failure/guard exit retained.
Old B2driver15744and shared-brake64036also exited on dependency failure.
Their status.json files are STALE; failure.json is authoritative.
No B2/shared-brake native controls occurred. Do not revive those executions.

Replacement ACTUALLY LIVE executions (20:42verified):
- B1: runs/B3_and_B1_independent_cohort_execution_20261031_v5_eval_profile_fix,
  Python26916. New replay child20261031T123039922770Z_native_replay_eval_configuration_recovery
  worker63800/guard2100, actual case23controls observed; then measured36actor gate,
  then four registered B1PPOseeds. Completed B3is hash-checked and reused.
- B2: runs/B2_five_seed_PPO_execution_20261031_v2_eval_profile_fix,
  Python38164, waiting on exact B1identity26916 through retained read-only handle.
- Shared brake: runs/shared_brake_five_seed_execution_20261031_v2_eval_profile_fix,
  Python10408, waiting on exact B2identity38164, updated B1execution reference.
All source snapshots remain frozen; new executions retain failed predecessors.
See reports/native_evaluation_recovery_queue_activation_20261031.json.
Root prepare_native_training_view_pair now removes the learning-only flag for
actor evaluation; no refit or PPO update during recovery. Measured36unstarted
batch094314260006was corrected with original metadata preserved.
30related tests pass. No changes to bound prepare_native_supervised_ppo.py.
Continue actual queues, publish remaining local evidence, and address healthy
B1/PPO-vs-rule/feedback/frequency/light/fresh-final gaps without changing goals.

# Historical handoff — 2026-10-31 19:10 local

Authoritative update;older live-status snapshots below are historical.
Goal ACTIVE/UNMET. Canonical seed33 batch093156483162 COMPLETE and normal owned
cleanup verified:H10/12,H012/12,zero unsafe,each20561training controls/650Adam.
All50843rawcontrols and24initial actions audited. Exact paired p=.5,diff-16.67pp;
this contradicts stable feedback benefit. Hfails static5/7 with81/68planning
failures,810/680brake-control interventions;H0passes both without those failures.
Full feedback audit547H/435H0decisions:19Hargmax changes overall,0of180on failed
visited decisions;H0negative control0changes. Do not infer counterfactual driving.
Complete pair/failure diagnostics reports committed and source7dd8475pushed.

Canonical driver47784 ACTUALLY RUNNING seed44batch093202647818,worker28260
creation1793444046.648,guard14072. Hfinished20502controls/650Adam,recorded12/12;
H0actually training. Whole44pair raw audit awaits H0stream closure;do not score
as a fully audited pair yet. Next55then fresh seed22H0.
Dependent68480still waiting;B2driver15744still waiting through retained read-only
dependency handle. Both verified live;all respective source snapshots immutable.
Codebfa1740hosted workflow41194528717SUCCESS. Independent checkout now7dd8475.
Do not fetch short SHA;use full git rev-parse HEAD output for local fetch.

PUBLICATION COMPLETE: native_seed33_canonical_feedback_pair_20261031_repro.zip,
669files/67070602bytes,SHA647e96acd757a62aa30639402d51dd53bc811d1a7348e16b73ed385b09b7bde0.
All models/raw/checkpoints and complete feedback diagnostics included;8auxiliary
JSON private-path redactions logged.32parts+manifest actually uploaded/downloaded
without login. Independent checkout7dd8475reproduces50843controls,24actions,
byte-identical figure. Both PACKAGED feedback auditors actually rerun and every
non-timing field including script SHAidentical:547H/435H0decisions. Public receipt
native_seed33_canonical_public_verification_20261031.json;commands in model_card.
Upload/reproduction execsessions24615/58706/90988/52233all completed successfully.
No repeat needed. Continue the live registered queues and remaining research gaps.
Main.venvhas no psutil;use existing resource checks/CIM for metadata.

# Prior operational snapshot — 2026-10-31 18:52 local

NEW ACTUALLY LIVE B2queue: runs/B2_five_seed_PPO_execution_20261031,
Python15744,launcher44024;exact process identity in its local process_identity.json.
It waits on a read-only retained handle to dependency68480and fails closed if
that process exits/fails before its success marker. Driver source417f53f,
frozen308files verified. Its source snapshot is immutable;later root CLI changes
do not alter this running driver. Plan/counters/identity/source manifest are local.
No B2CARLAtraining has started. After B1completion it will train and raw/inference
audit all11/22/33/44/55,20480physical controls/7200sec each,no retries/omissions.
Driver is tracked scripts/run_native_planner_cohort.py. It invokes tracked auditors
directly;no missing local postprocessor helper dependency.

Actual prepared B2batches:11=20261031T104032928978Z_native_planner_training;
22=20261031T104126873363Z_native_planner_training;
33=20261031T104129314279Z_native_planner_training;
44=20261031T104131479611Z_native_planner_training;
55=20261031T104133575143Z_native_planner_training.
All5ValidateOnlypassed and actual frozen-runtime restore succeeded;zero resets/
controls/updates,empty Adam,frozen obs,distinct policy tensor hashes,GridAStarbackend.
Initial model/norm/RNGbytes equal respective pristine H,not trained Hweights;
same reward/MPC/teacher priors/budget,only B2/body_spline_v1config changes.
19targeted tests pass root13.87sec and independent checkout417f53f3.76sec.
Public reports native_B2_five_pristine_training_preparations_20261031.json,
native_B2_frozen_pristine_restore_verification_20261031.json,
native_B2_five_seed_queue_activation_20261031.json. Do not count these as trained
seeds or repeat preparation over existing registrations. Root preparer later adds
an optional explicit reference-checkpoint path for public-capsule layout portability;
all five existing prepared/frozen sources keep their executed original bytes.

# Prior operational snapshot — 2026-10-31 18:37 local

Original goal remains ACTIVE/UNMET;this block supersedes old live statuses below.
Canonical queue47784 is ACTUALLY RUNNING seed33 batch093156483162. Htraining
complete20561controls/650Adam,recorded known12evaluation10/12,two static
deadlines5/7;H0now actually training. Worker18536,guard29604healthy. Full
method raw/inference audits intentionally wait for pair/bridge closure;do not
promote the recorded Hsummary into a fully audited paired endpoint yet.
See native_H_seed33_lr_correction_recorded_endpoint_20261031.json.

Old high-rate seed33batch082311523384 COMPLETED H8/12,H06/12,4/6deadlines,
zero unsafe;normal owned cleanup and both audits completed automatically before
canonical activation. Both reports are committed. No old native process was
killed. Canonical queue then continues44,55,fresh seed22H0;dependent68480
remains waiting and will run B3,paired training-view probes,four B1seeds.
Use existing drivers/statuses;do not duplicate completed trials or revive old queues.

NEW public provenance ACTUALLY reproduced: native_training36_measured_views_20261031_repro.zip,
18parts/965files/37549119bytes,SHA46ecf67e4cbfc3d7a55cb534c66d0820189133a3b8810df73ebbd2092a77ecd8.
Independent unauthenticated download and496.063second reconstruction verify all
20521TRAINcontrols,36conditions,51episodes,4156byte-identical labels. Full raw
Htraining,all checkpoints,partial interrupted validation,failure and unstarted
H0retained;auxiliary private paths redacted with hashes/ledger. This does NOT
improve measured B1driving or count as new PPO. Receipt
native_training36_measured_views_public_verification_20261031.json;download and
reproduce_native_training_views.py commands in README. Public asset sourceb83f6fd;
independent checkoutbcf1e39,18targeted tests passed4.33s and hosted41192679399SUCCESS.
Completed old first archive without packager snapshot is retained locally and
was NOT uploaded;the *_repro.zip public archive is authoritative.

# Prior operational snapshot — 2026-10-31 17:58 local

Original objective ACTIVE/UNMET. This supersedes ALL earlier live queue references.

- Native high-LRseed33pair remains ACTUALLY RUNNING batch
  20261031T082311523384Z_native_supervised_ppo,worker67772,guard64284,
  original lifecycle wrapper55360. Hfinished20644controls/650Adam and8/12
  known12validation,4deadlines;H0training20604/650complete,evaluation running.
  Do not score the whole pair or package incomplete evaluation. Guard healthy.
- ROOT CAUSE FOUND:the33/44/55initializers retain3e-4,while canonical
  seed11and22use3e-5. The previous env-only matcher missed LR. H33median
  KL.0683,max.6743,61/65phases>.015,actorL2 2.32 versus~.4canonical.
  Original H33BCgate11/12. Preserve the high-LRtrial and its consumed effort;
  no causal claim until matched-rate comparison. Report
  native_H_seed33_initialization_gate_vs_high_lr_20261031.json.
- Only Pythonorchestration60176and idle dependent queues were retained-handle
  stopped BEFORE further trials;native wrapper/worker/watchdog confirmed live
  through read-only retained handles. Original receipts and states preserved.
- ACTUAL CANONICAL queue:execution remaining_v4_seed_PPO_execution_20261031_v5_lr_bound,
  Python47784(launcher68704),runs/run_canonical_H_H0_lr_bound_cohort_20261031.py.
  It waits for original high-LRseed33owned cleanup;audits original complete
  outcomes if normal,otherwise retains failure without scoring;then runs:
  seed33 20261031T093156483162Z_native_supervised_ppo;
  seed44 20261031T093202647818Z_native_supervised_ppo;
  seed55 20261031T093206498809Z_native_supervised_ppo;
  seed22H0only 20261031T093208454506Z_native_H0_canonical_budget_replica.
  All are prepared,NOT started. Explicit3e-5fullsaved-model profiles match
  seed11before any trial. New33old/new initial policy tensors,norm,RNGexact;
  only LR/schedule/Adamrates differ.18targeted tests and actual4ValidateOnly
  checks pass. Newfrozenpostprocessor resolves workspace and auditors from
  its immutable snapshot;no mutable root preparation needed during execution.
- Seed22H0fresh budget replica reuses the EXACT original zero-PPOcheckpoint.
  Existing H22retained;no duplicate Htraining. Earlier GPUtruncation/recovery
  and all extra historical method effort stay reported. This is a predeclared
  new canonical training trial regardless of outcome,NOT a sixth independent
  seed and NOT equal total historical tuning effort.
- ACTUAL DEPENDENT queue:execution
  B3_and_B1_independent_cohort_execution_20261031_v4_training_view_probes,
  Python68480(launcher66212),runs/run_B3_B1_and_training_view_probes_20261031.py.
  Waits for canonical3pairs AND fresh22H0budget replica;then B3exact24,
  BOTH new actor-only coverage probes,then unchanged4independent B1known12
  gates/PPO(22/33/44/55,20480controls,lr3e-6,KL.01,rollout1280,scale.05).
  No retries,seed omissions or selecting a subset of actor-probe outcomes.
  Originalseed11B1known12low-rate arm remains its reference. Broader36/24
  branch is separate development evidence.8hour cap is dependency wait only.
- New B1TRAIN dataset is COMPLETE:20521Htraining frames verified against
  original bridge states/commands/sensors/clocks;4156fixed-stride5and last
  precontrol labels across ALL36training conditions and51episodes,including
  failures/lastpartial episode. NO evaluation trace labels. No plant integration.
  Dataset runs/B1_from_H_training36_measured_views_20261031,
  SHA98943a7ae5e71ec89e7cabb076ebd931c3dd6e1c3828557882635d166bb78e88.
- Matched512actor-only fits now prepared for24development validation:
  replay 20261031T093757339917Z_native_replay_training_view_bc;
  measured36 20261031T094314260006Z_native_measured36_training_view_bc.
  BothzeroPPO;critic/norm/RNGexact. NewTRAIN steering MSE.018325replay
  versus.002062measured;originalTRAIN error increases slightly. TRAIN fit
  is NOT driving/generalization proof. Additional Hteacher prior costs are
  explicit and not completely ancestry-deduplicated;not equal-input evidence.
  Replayfit saved before missing optional metadata field KeyError. Fixed by
  using protocol lineage steps;metadata completed from saved actor and last
  heartbeat WITHOUT repeating512updates. Intermediate history absent,
  originalfailed preparer/log retained. BothactualValidateOnly380files pass.
- Previously complete broader DEVELOPMENT scores unchanged:H24/24,H021/24,
  B19/24(9bounds,3collision,3deadlines). All3capsules publicly reproduced,
  and diagnostic239TRAIN labels plus1808feedback decisions publicly verified.
  No final120condition inventory activated. Healthy B1,canonical5seeds,
  PPOvsstrongB3,B2andremaining ablations/statistics remain unfinished.

# Archived handoff before current optimizer-profile correction

# Current handoff — 2026-10-31 17:08 local

Original objective ACTIVE/UNMET. Supersedes every older process description.

- Broader36train/24geometry-separated DEVELOPMENT validation now complete:
  H24/24,zero unsafe;H021/24,three static deadlines,zero unsafe;
  B19/24(curve4/8,static2/8,dynamic3/8),9bounds/3collision/3deadline.
  H20521train/650Adam,H020561/650,B120480/640. B1is NOT healthy on this
  distribution;do not turn its deficit into a formal five-seed hierarchy claim.
  H/B1native-system comparison explicitly reports candidate-filter and cadence
  differences;it is not proof PPO beats strong B3. H/H0gap12.5pp,p=.25.
- All three complete capsules are PUBLIC and independently reproduced:
  H8089raw/24start actions,H030292/24,B130839/24,byte-identical figures.
  Receipts H_v4_broader_evaluation_public_verification,
  H0_v4_broader_public_verification,B1_v4_broader_public_verification_20261031.json.
  All failures and original Hinterrupted evaluation costs remain charged.
- ACTUAL ACTIVE native pair:seed33batch20261031T082311523384Z_native_supervised_ppo,
  worker67772,guard64284. Htraining now completes20644controls/650Adam;
  Hdevelopment evaluation is in progress. H0not started at last observation. Driveractual60176(launcher5444)
  execution runs/remaining_v4_seed_PPO_execution_20261031_v4 then does44and55.
  Do not restart/reprepare its children or mutate frozen snapshots. Verify live
  CIM/creation receipts before adopting any PID. Parent B1owned cleanup is complete.
- Four B1seed22/33/44/55v4copies and512step actor fits are complete,using the
  SAME29TRAIN corrections/base teacher bytes as seed11. Distinct weights,
  exact critics/norm/RNG,zero PPO. Report native_B1_independent_approach_fits.
  Model saved rollout320 differed from inherited donor protocol1280;BEFORE
  ANY NATIVE LAUNCH,protocols were aligned to320,original protocol/prepared
  bytes and a repair ledger preserved. No model bytes or fitting repeated.
  Future PPO explicitly rebuilds the empty buffer to1280. Actualmodel/declared
  schedules,zero counters and all four source/data identities were checked.
- B3exact24validation batch20261031T082841Z_native_rule is PREPARED/UNACTIVATED.
  All Henv fields,training/validation inventories and physical source are
  identical. New explicit validation selector rejects simultaneous changes to
  catalog/control/planning;5tests plus actual ValidateOnly389sourcefiles pass.
- ACTUAL WAITING queue:execution runs/B3_and_B1_independent_cohort_execution_20261031_v2,
  Python36820(launcher26900),script runs/run_B3_and_independent_B1_cohort_reward_bound_20261031.py.
  Frozen source/preparer/plan and protocol hashes verified. Waits up to8hours
  for original three H/H0PPOpairs,then B3and all four B1initializer gates/PPO.
  B1new20480controls each,lr3e-6,KL.01,logstd-4,rollout1280,frozen observation
  stats and explicit PPOreward scale.05. These four use the SAMEknown12training inventory as the existing
  seed11B1and five-seed H/H0cohort;broader36/24is a separate development branch.
  No retries/seed omissions;failure stops the chain and preserves actors/logs.
  Wait cap applies only to dependency waiting,not total training. Journal
  status.json and plan.json are authoritative. No extra native process now.
- Additional prelaunch check found four plain saved normalizers inherited a
  donor protocol fixed-reward descriptor. BEFORE any gate/PPOlaunch,only the
  idle waiting queue was retained-handle stopped;original journal retained.
  Gate protocols now correctly declare their plain normalizers;future PPO
  explicitly converts to fixed scale.05. No model/norm/RNG bytes changed.
  All4temporary reconstructed learners match seed11rollout,LR,KL,epochs,batch,
  gamma/GAE/entropy/value/clip settings.13targeted tests pass. Reports
  native_B1_independent_reward_units_prelaunch_fix,
  native_B1_cohort_offline_PPO_configuration_verification and
  native_B3_and_B1_reward_bound_queue_20261031.json are authoritative.
- New broader B1TRAIN-only diagnostic:242queries/239valid labels,all36first
  completed training conditions,stride80and last pre-control state;3invalid
  poses rejected. Initial/final command-label MSE.280265/.280675;steering
  MSE.004076/.004034. Identical RMSand normalized inputs. This does not show
  catastrophic PPOdrift on these training observations;it does not prove
  driving quality,coverage sufficiency or unseen generalization. No fitting
  or validation-state training. Report native_B1_broader36_training_oracle_drift.
- Full new diagnostic bundle is PUBLIC and independently reproduced:
  native_broader_training_feedback_diagnostics_20261031.zip,199599bytes,
  SHA256df158c5c5b147e27acd64ffd7730e2400ede25fb7cb5ddd22dd2f0c9ece81ca8.
  Source4e7f6b2,239valid TRAIN label checks plus823H/985H0decisions all
  non-timing/non-script-byte fields exact against public parent capsules.
  Hchanges24actions,H0zero;H-only win case13has no argmax sensitivity on
  Hvisited states. Not causal performance evidence. Receipt
  native_broader_diagnostic_public_verification_20261031.json.
  Full per-decision files live in runs/broader_feedback_diagnostics_20261031
  and the public ZIP,aggregate in native_broader_feedback_reliance_summary.
  Hosted4e7f6b2run41186775836is SUCCESS;full-log request was transient and
  count not inferred. Root13reward-binding/actor/runtime tests pass5.49s.
- Current source tests:51scene/spawn/preparation;9trajectory-query;7pristine
  geometry(+7independent);3training-lineage;6actor-fit;4runtime compatibility;
  5B3validation selector. Hosted0643passes723/18skip/4CARLA deselected;
  later0602also succeeds. Newer hosted statuses require verification.
- Main blockers remain healthy full-task B1,>=5actual trained policies per
  formal comparison,PPO-vs-strong-B3value,adequate mechanism ablations,and
  >=120FRESH final paired conditions. No final inventory activated/viewed here.
  Diagnose B1new-layout failures from TRAIN traces;do not train on validation
  states or claim these development conditions are final. Candidate geometry
  features/coverage and training length are possible explanations,not proven.

# Archived earlier handoff

# Current handoff — 2026-10-31, trained H v4 complete

Original objective ACTIVE/UNMET. This section supersedes all earlier live
process identities. Verify receipts and creation times before acting.

- Latest actual state:recovered H0finished10465new training controls/330Adam,
  retained lineage20537/650Adam. Full endpoint10/12,zero unsafe;static cases5/7
  hit deadlines. New segment15887rawcontrols and12start actions audit. Hseed22
  remains12/12. Pair report native_v4_seed22_recovered_feedback_pair_20261031.json;
  p=.5,one developmental recovered pair,not a consumed-budget-matched result.
- The operational override snapshot omitted probe_carla37.py. Its first cleanup
  failed,then the ORIGINAL fallback correctly confirmed the empty server and
  stopped it. Preserve launcher_failure.json/server_cleanup.json. Guard exit0,
  complete summary,owned stop/empty ports and dead recorded lifecycle were
  explicitly verified. Reconciliation receipts
  native_H0_recovery_cleanup_reconciliation_20261031.json and
  native_seed22_recovery_completion_and_probe_fix_20261031.json. No training rerun.
  Snapshot now includes the probe;18dependency/cleanup tests pass14.39s. The
  actual new snapshot probe also connects readonly to active CARLA0.9.15and
  sees its3actors;it never advances the world. Complete cleanup on the new
  dependency snapshot will be observed at the next native stage exit.
- First queue61728stopped on the preserved parent launcher failure. After
  verified fallback cleanup,B1replay042301092506was explicitly activated clock
  only and launched once. ACTUALLY TRAINING worker69712(wrapper49880):first
 1280controls include2actual Adam updates,actorL2change.0156. No endpoint yet.
  Actual new guard snapshot contains the probe and correct hashes. Receipts
  runs/B1_replay_PPO_reconciled_launch_20261031.json and
  runs/B1_current_guard_snapshot_readonly_probe_20261031.json. Downstream queues
  17716/652/10840/67120/22264remain valid,using the updated root launcher at start.
- Remaining-seed driver is now V3,actual worker55888(wrapper9476),waiting ALL
  original gates,no prepared children. Status
  runs/remaining_v4_seed_PPO_execution_20261031_v3/status.json;launch
  runs/remaining_v4_seed_PPO_probe_fixed_driver_launch_20261031.json. V2worker71584
  was stopped only while waiting with no children;receipt
  runs/guard_driver_v2_superseded_for_probe_dependency_20261031.json. Do not
  invoke either older driver. V3freezes the dependency-complete new launcher.

- NEW authoritative operational state:seed22parent025409042778STOPPED after
  one10s nvidia-smi probe timeout raised in its legacy guard. Actual worker
  ended;owned-empty server cleanup confirmed. This was NOT a measured overheat:
  subsequent query returned45C in0.062s. Hwas already complete12/12with20539
  training controls/650Adam and4319evaluation controls;all24858rawcontrols and
  12start actions now audit. Report native_H_v4_seed22_completed_method_after_GPU_timeout_20261031.json.
- H0parent consumed at least16770controls;checkpoint10072/320Adam retained.
  At least6698controls plus any unknown unflushed tail remain charged. Its new
  recovery045159is ACTUALLY RUNNING under worker14448/guard27668,only H0,
  remaining budget10408controls to lineage20480. Model/normalizer/RNGbytes,
  env config,17physical/learning module texts and native actuation script are
  unchanged. Report native_seed22_GPU_timeout_recovery_20261031.json. Do not
  relaunch either parent method. New H0postprocessor writes
  native_H0_v4_seed22_recovered_endpoint_20261031.json after normal cleanup.
- GPUguard fix:two bounded10s probes;85C threshold unchanged;two failures stop.
  Explicit -UseCurrentGuard snapshots/hash-binds operational sources separately,
  leaving worker/policy snapshots intact. Recovery already uses it and reports
  healthy45-50C. Full local700passed/1skip/4CARLA exclusions/3warnings246.47s;
  initial65guard checks12.03s and51snapshot checks13.43s. Logs
  runs/gpu_guard_recovery_full_cpu_20261031.log. Sourcecb9c84fis pushed/remote
  verified;hosted41171158775SUCCESS683passed/18skip/4CARLA exclusions195.45s.
  Independent cb9c84falso passes71guard/recovery checks21.07s. Actual resumed
  guard source bytes equal committed source. Receipt
  native_GPU_guard_source_verification_20261031.json.
- Current queues ALL use -UseCurrentGuard:61728recovery045159→B1replayPPO042301092506;
  17716replayPPO→approachPPO042300931172;652approachPPO→broader025445532080;
  10840broader→seed33gate040926397622;67120seed33→seed44gate041003482907;
  22264seed44→seed55gate041003481908. Receipt runs/gpu_guard_owned_queue_launches_20261031.json.
  Old waiting queues42340/15780/24484/60144/64416and preparation worker30576
  were superseded only after identity/creation/untouched-child checks;no native
  worker affected. Receipt runs/gpu_guard_waiters_superseded_20261031.json.
- New remaining-seed driver V2is live,actual worker71584(wrapper64060),still
  waiting ALL33/44/55gates with no prepared children. Its updated frozen launcher
  explicitly selects the snapshotted new guard. Source291files and all old
  snapshots retained. Status runs/remaining_v4_seed_PPO_execution_20261031_v2/status.json;
  launch runs/remaining_v4_seed_PPO_gpu_guard_driver_launch_20261031.json.
  Do NOT invoke the superseded original driver or independently prepare children.
  Its source_commit is the base commit at freeze time;per-file hashes bind the
  then-uncommitted operational fix. The active guard's exact committed bytes
  are now independently checked againstcb9c84f. Source snapshots are immutable.
- Prior runtime/reliance sourcebd645aeis independently reproduced:all four
  runtime JSONs byte-identical;all439H/494H0decisions and counterfactual
  probabilities identical apart from offline wall time. Independent14tests
  pass13.26s. Receipt native_v4_runtime_feedback_independent_verification_20261031.json.
  Hosted41169543613SUCCESS. These add zero driving/updates.
- A20s nonblocking20Hz local profile of the resumed worker captured305active
  Python samples with8read errors. Feedback appears in142stacks,batch-road SAT
  in115,scalar-road SATin36. These overlapping active-stack proportions are
  approximate,not wall-time percentages. Raw private-path profile stays local:
  runs/H0_gpu_recovery_cpu_profile_20261031.txt. This supports a separate exact
  SAT compute prototype;NO kernel optimization has been implemented/enabled yet.

- New readonly runtime/reliance audit is complete;report
  reports/native_v4_runtime_and_feedback_20261031.md. All4complete endpoints
  retain every frame,including terminal sensor refreshes;actual controls are
  recovered from same episode/frame native responses(12permethod),none dropped.
  HenvdecisionP95/P99/max348/418/4224ms,0.68%above500ms;H0P95705ms,10.12%above.
  B1PPOfeedback aloneP95220msat50msperiod;full prediction/runner latency not
  logged,so no realtime certificate. Hstationary8.00%,dynamic brake290controls;
  H0stationary24.31%,failed-plan brake460controls;B1PPOstationary36.24%.
- New full-history assay reproduces all439H/494H0decisions and9223native
  control histories. Fixed weights/statistics/physical endpoint mask;hide only
  learned feedback. H10actions change across1/3/4/7/10/11;H00changes/zeroTV.
  H's exclusive success case5has no changed action on Hvisitedstates,so this
  does NOT explain the performance gap. Reports native_H_v4_feedback_reliance_20261031.json
  and native_H0_v4_feedback_reliance_20261031.json. No new controls/PPO updates.
  Both use the exact frozen batch runtime via NAVSTACK_AUDIT_SOURCE_ROOT.
- New readonly tools summarize_native_runtime.py/audit_native_feedback_reliance.py
  leave all active/queued frozen sources intact. Dynamic history/shared mask/
  terminal-frame regression subset14passed8.36s. No new full local suite claimed.
  Initial runtime probes failed on terminal refreshes lacking vehicle_control;
  logs retained. The corrected tool joins original native actuator reports,
  covers every frame and rejects missing/conflicting reports.
- Seed22pair025409042778has Htraining COMPLETE20539controls/650Adam and H
  evaluation complete;H0is ACTUALLY TRAINING worker42192,last observed11835controls.
  Original method postprocessors still wait for whole-batch normal cleanup.
  Do not create their default audit/inference output files early.

- Batch025409165666 is FULLY COMPLETE: **H12/12,H011/12,zero unsafe**.
  Both method postprocessors completed and normal guard0/owned-empty server
  cleanup is recorded. H0's only failure is static case5(deadline). Actual
  H020,562training/4,886evaluation controls/650Adam. Combined50,348rawcontrols
  and24start actions verify. Reports native_H_v4_seed11_endpoint_20261031.json,
  native_H0_v4_seed11_endpoint_20261031.json,and
  native_v4_trained_seed11_feedback_pair_20261031.json;exact p=1.0.
  Both B1actor-fit evaluations have since completed;seed22v4PPOpair025409042778
  is ACTUALLY TRAINING under worker42192,currently H. Check its live heartbeat.
- Earlier completed-H method snapshot remains intact for provenance.
  Actual20,563training controls/650Adam,4,337evaluation controls;all24,900raw
  frames and12measured-start actions verify. Early report
  reports/native_H_v4_seed11_method_complete_20261031.json. This earlier method
  report does not claim whole-batch closure;use the completed pair above.
- Frozen-v4 migration pair023833/023839 is complete **H9/12,H010/12**,zero
  unsafe. Difference-8.33pp,exact discordance p=1.0. This reverses the older
  v3gap;feedback benefit remains unproven. Reports
  native_H_geometry_v4_endpoint_20261031.json,
  native_H0_geometry_v4_endpoint_20261031.json,
  native_v4_frozen_feedback_pair_20261031.json.
- B1v4PPO022652 remains7/12,zero unsafe/five deadlines. Public upload and
  unauthenticated independent reconstruction are now COMPLETE:22parts,
  44,337,162bytes/489files,27,737controls,12actions,identical PNG. Receipt
  reports/B1_v4_PPO_public_verification_20261031.json. Do not repeat publication.
- Offline exact policy-history reconstruction shows the original zero-PPO
  initializer already brakes at the four stopped failures' visited states;
  final PPO brakes somewhat more. This is not initial-policy closed-loop
  performance. Training-only approach queries40states,29solved reference
  labels/11waits. A27interior-goal-pose prototype recovers only1of10waits and
  is NOT wired into production. No task gate or geometry changes in this probe.
- Two matched actor fits are complete,each512MSEsteps from the SAME original
  zero-PPO model. Replay uses original3467teacher TRAIN rows;approach adds all
  29valid solved-reference training labels(weight20),including braking. Critic,
  empty PPO optimizer,normalizer/RNGbytes/log_std unchanged. Full native scores:
  replay8/12,3bounds/1deadline;approach10/12,2bounds. Both basic families4/4safe.
  All10,193rawcontrols/24start actions verify. Complete comparison
  native_B1_v4_approach_pair_complete_20261031.json. Still zero PPO updates.
  Pair034614808334;replay034614809334;approach034641272228. Reports
  native_B1_v4_approach_pair_preparation_20261031.json and
  native_B1_v4_stop_diagnosis_20261031.md. This is supervised baseline development.
- Queues64312/63196completed both actor fits;67468activated seed22PPO025409042778.
  Current queues63456:seed22→B1replayPPO042301092506;
  42340:replayPPO→B1approachPPO042300931172;
  15780:approachPPO→train36/validation24pair025445532080. Both new B1PPOphases
  have20,480controls,original1280rollout/schedule,identical empty-PPOinitial
  actor tensors/RNG/normalization and physical config. Each basic-motion entry
  is4/4safe;neither is a healthy-final-baseline claim. Report
  native_B1_v4_approach_PPO_preparation_20261031.json;actual queue receipt
  runs/B1_approach_PPO_owned_queue_launches_20261031.json. Waiter28112was
  superseded ONLYwhilewaiting,before child activation,with retained verified
  handle;receipt runs/validation_waiter_superseded_for_B1_approach_PPO_20261031.json.
  Old seed22waiter66736
  was superseded ONLYwhilewaiting,after identity/child-unstarted checks;
  receipt runs/seed22_v4_waiter_superseded_for_B1_approach_pair_20261031.json.
  All native launchers remain single-server,strict normal cleanup,no retries.
- Read-only postprocessors69256/57704completed both actor-fit endpoints;receipt
  runs/approach_pair_postprocessors_launches_20261031.json. Existing six H/H0
  watchers retain their distinct outputs and21600s caps. Do not duplicate runs.
- B1long031506is PREPARED_NOT_QUEUED_NOT_STARTED;preserve its registration and
  runs/native_continuation.json pointer. Do not confuse preparation with training.
- Fresh-exploration initialization now permits an exact idempotent request on
  a still-untrained model;all non-log_std changes/trained-state use still reject.
  Full local CPU **676passed,1skip,4CARLA deselected,187.30s**;targeted24pass7.35s.
  Logs runs/approach_pair_full_cpu_20261031.log and
  runs/approach_pair_targeted_tests_20261031.log. Hosted verification follows.
- Sourceff3963a is pushed/remote verified;hosted41167159676is SUCCESS.
  Hosted661passed/16skip/4CARLA exclusions193.69s. New initializer-geometry
  guard tests pass7checks3.50s;full root now682passed/1skip/4CARLA exclusions
  197.56s. Log runs/pristine_initialization_full_cpu_20261031.log.
  Independent checkoutff3963apasses40targeted checks in9.36s. Initial command
  used a nonexistent partition-test filename and
  ran zero tests;retained separately. Corrected invocation passed all40.
- Remaining seed33/44/55pristine zero-PPO initializers now have separate v4
  copies040926397622/041003482907/041003481908. Model,normalization and RNGbytes
  are exact;only environment geometry field changes. Original v3teacher inputs
  remain disclosed. All three frozen388-file launch validations pass. They
  are queued AFTER the train36/validation24pair,then33→44→55,with exact parent
  normal-cleanup checks. Queue PIDs24484/60144/64416;receipt
  runs/remaining_v4_seed_queue_launches_20261031.json. Six read-only endpoint
  postprocessors also launched;receipt remaining_v4_seed_postprocessors_launches_20261031.json.
  These are representation gates,NOT completed independent PPO seeds. A real
  continuation program is now live under actual Python PID30576(wrapper58700),
  WAITING_FOR_ALL_THREE_COMPLETE_SEED_GATES. It freezes289preparation/launcher
  source files,requires ALLthree original75percent gates,then prepares and
  runs H/H0PPO33→44→55with20,480controls per method/7200s per pair. No retries,
  seed omissions or task changes;exact seed11env comparison before launch.
  Do NOT independently prepare the same children:the program owns those
  registrations. Live status runs/remaining_v4_seed_PPO_execution_20261031/status.json;
  launch runs/remaining_v4_seed_PPO_driver_launch_20261031.json;public queue proof
  reports/native_remaining_v4_seed_PPO_queue_20261031.json. Actual driver
  runs/run_remaining_v4_seed_PPO_20261031.py;source/hash ledger kept beside status.
- Bounded H/H0v4public publisher59436is COMPLETE:32parts downloaded unauthenticated,
  all609files/50,348controls/24actions/frozen statistics/identical PNG reproduced.
  Publication receipt reports/H_H0_v4_PPO_public_verification_20261031.json.
  A second full stable-source reproduction also passes atbe6a50d after the
  first verification overlapped the independent checkout update;executed entry
  source bytes match both commits. Supplemental receipt
  reports/H_H0_v4_PPO_stable_source_verification_20261031.json. Complete
  archive66,513,018bytes/609files,SHA1374429bca525b878cfdb86fb9ffd9f6b1d3943c24ff8208c056b6458e009351.
- B1replay034614809334is now COMPLETE8/12but3bounds and1deadline,zero PPO;
  all5,251controls/12actions audited. It is NOT a healthy baseline. Approach
  034641272228is COMPLETE10/12with2bounds;all4,942controls/12actions audited.
  Read-only comparison waiter67212completed successfully,source hash bound,
  no repeated controls;receipt
  runs/B1_v4_approach_comparison_launch_20261031.json. Its new analyzer checks
  both complete raw bindings/conditions/physical configs and unchanged model
  SHA;full result native_B1_v4_approach_pair_complete_20261031.json now exists.
  Both five-part public capsules have completed unauthenticated independent
  reconstruction at fixed checkoutbe6a50d:446/445files,5251/4942controls,12actions
  each,frozen stats/identical PNG. Receipts B1_v4_replay_fit_public_verification_20261031.json
  and B1_v4_approach_fit_public_verification_20261031.json. No new PPO yet.
- Sourcebe6a50dis pushed/remote verified;independent seven initializer/geometry
  checks pass4.45s. Hosted41167689480SUCCESS667passed/16skip/4CARLA exclusions
  190.79s. Latest pushed sourceace8138host41167985455also SUCCESS. The comparison
  analyzer is now actually executed on both complete endpoints. No new
  production learning/geometry source changes after the682-test local suite.
- Five independently trained seeds,healthy full-task B1,at least120NEWpaired
  final conditions,frequency/shared-safety ablations,PPOversusstrongB3and
  statistically supported hierarchy/feedback value remain open. All known12
  and separated36/24inventories are DEVELOPMENT;do not relabel them final.

## Archived handoff — 2026-10-31 11:22 local

Original goal ACTIVE/UNMET. This supersedes earlier live process statuses.

- H0shared-mask001825is COMPLETE7/12,5deadlines,no unsafe;20,618new training
  controls/650Adam,7,044evaluation controls. All27,662rawframes and12initial
  actions audited. Guard0andowned-empty cleanup confirmed. H9/12versusH07/12
  is ONE seed;H-only cases4/5/6,H0-only1,p=.625. Report
  reports/native_shared_mask_feedback_pair_20261031.json. No formal benefit claim.
- Longer-rollout B1004737is COMPLETE6/12,4bounds,2deadlines;20,480training
  controls/87Adam,5,720eval. Its six-part public asset downloaded and independently
  reproduced483files/26,200controls/12actions/identical PNG. Receipt
  reports/B1_rollout1280_public_verification_20261031.json. Not a healthy baseline.
- Root now implements explicit carla_driving_indexed_union_256_v4. Original
  versions remain64x64and unchanged. Common policy grid stays64;shared body SAT,
  planner and risk use256collision cells. No lane-width dilation. Offline local
  map diagnostics isolate two coarse-raster terminations;two task-box violations
  remain. These are not revised old outcomes or mesh-safety certification.
- Regression671passed/1skip/4CARLA deselected167.30s;later preparation/migration
  subset30passed5.38s. Full-test log runs/fine_grid_full_cpu_20261031.log.
- Frozen B1v4migration022009is COMPLETE7/12,1bounds/4deadlines,zero updates,
  all12known conditions;basic motion3/4zero unsafe. All6542controls and12actions
  audited;normal owned cleanup. Separate v4PPO batch022652is ACTUALLY TRAINING,
  worker58000/guard8880,target20,480controls,
  same initial policy tensors/empty optimizer/normalizer/RNGas v3004737;only
  environment geometry field changes. Original v3teacher data disclosed.
  Training is now COMPLETE20,480controls/72Adam/39episodes/1635.32s;reload exact.
  Full endpoint is COMPLETE7/12,zero unsafe,five deadlines;families3/4,2/4,2/4.
  Guard0/owned-empty cleanup;27,737controls and12actions audited. All saved
  Adam parameter steps72;RMSmean/variance/count exactly frozen3467.0001.
  Four failures stop1.45–2.37mshort with speed/yaw gates passing;case7is3.45m
  away/too fast. Reports native_B1_geometry_v4_PPO_endpoint_20261031.json,
  native_B1_v4_training_analysis_20261031.json,native_B1_v4_terminal_failures_20261031.json.
  Modelc4cc9c09…,normalizer49b9e7de…. Never rerun this completed phase.
- Hfrozen v4migration023833is ACTUALLY RUNNING,worker16652/guard49088;last
  observed case7. Its new H0migration023839is still queued,not simultaneously
  driving. Both retain old policy weights/lineage with zero new PPO updates.
- Queues8900H0→migration and9516migration→v4B1training have activated their child.
  Current remaining queues:19508v4B1→Hfrozen v4migration023833;
  62136H→H0frozen v4migration023839. Waiter49052was superseded only
  while waiting,before seed22activation;receipt in runs/seed22_waiter_superseded_for_H_geometry_migrations_20261031.json.
  Verify receipts/process creation identities;old waiting queues44456and67436
  were superseded only before either could activate its child. Old locks/logs
  remain. Current receipts:geometry_owned_queue_launches_20261031.json and
  geometry_training_owned_queue_launches_20261031.json. No automatic retries.
  The obsolete unstarted v3seed22/36-24queues58636/47244are now also superseded
  after exact initial-state comparison passed for their v4replacements.
  New queue62148:H0migration→seed11v4pair025409165666;
  66736:seed11→seed22v4pair025409042778;
  28112:seed22→v4train36/validation24pair025445532080. Each method has20,480controls
  and original guarded rewards. These batches are PREPARED,not trained results.
  Current receipts runs/v4_training_owned_queue_launches_20261031.json;proof
  reports/native_H_v4_training_preparation_20261031.json. Old prepared batches
  and the rejected unguarded preparations remain intact,unstarted.
- Bounded read-only postprocessors audit migration and v4training once normally
  closed. Actual launcher receipts live in runs/B1_geometry_postprocessor_launch_20261031.json
  and runs/B1_geometry_training_postprocessor_launch_20261031.json. Do not
  duplicate either experiment. Inspect current live heartbeat,not this snapshot.
  Six new read-only pair postprocessors use distinct method logs/receipts and
  dynamic12/24case counts;bounded21600s,no retries. Receipt
  runs/v4_training_postprocessors_launches_20261031.json.

- H0public capsule5042b3c2…is uploaded in4parts,32,732,190bytes/502files;
  unauthenticated download reassembles exact bytes. First premature download
  failed404before upload completed;failure log retained,retry used a new folder.
  Independent checkout7dfbb3dreproduces502files/27662controls/12actions/frozenRMS
  and identical PNG. Receipt reports/H0_shared_mask_public_verification_20261031.json.
  Archive has no packaged plot;use the committed PNGas explicit reference.

- The saved Hcase9query changes from7200-node failure tov4analytic success at
  node1,with unchanged planner budget/start/goal/vehicle/cost/heuristic. Report
  reports/native_fine_grid_recorded_query_20261031.json. This is offline geometry
  evidence,not an actual Hsuccess or merely a faster equivalent search.
- Source19da5acis pushed/remote verified. Hosted41160661963SUCCESS657passed,
  16skipped,4CARLA deselected159.70s. Independent new-source30checks pass12.60s.
- B1v4migration asset initial8MiB-parts upload failed408;only part003/manifest
  were public at inspection,so that manifest is INCOMPLETE. New chunk2m-named
  transfer has nine2MiBparts of the identical archive8d9720f2…(18,247,209bytes,
  437files,6542controls),all uploaded and listed. Actual download/reproduction
  is being verified. Old incomplete assets remain;no deletion/clobber attempted.
  Download has now reassembled the exact8d9720f2…bytes;independent reconstruction
  uses the newly corrected v4plot,which shows the256collision grid. V3H0plot
  still reproduces byte-for-byte(a61e2f80…). Seven plot/geometry checks pass7.06s,
  and44preparation/partition checks pass7.19s.
  Public migration independent checkout2fe4af8has now completed all437files,
  6542controls,12actions,frozenRMSand identical corrected256-grid PNG. Public
  query inputs also reproduce exact statuses/nodes/costs/path hashes. Receipt
  reports/native_geometry_public_verification_20261031.json.
- Latest code2fe4af8is pushed/remote verified. Full local672passed/1skip/4CARLA
  exclusions217.71s. Hosted41162252424SUCCESS657/16skip/4CARLA191.76s.
- New B1v4PPO public publisher56860was launched with a2400s cap. It plots,
  packages,uploads2MiBparts,downloads without auth,and uses the independent
  checkout to reproduce all raw frames/inference/PNG. No driving/updates/retries.
  Actual receipt runs/B1_v4_PPO_publication_launch_20261031.json;logs
  runs/B1_v4_PPO_publication_20261031.log/.err.log. Do not claim publication
  complete until reports/B1_v4_PPO_public_verification_20261031.json exists.
  Failure receipt runs/B1_v4_PPO_publication_failure_20261031.json if it fails.
- Unchanged B1continuation031506to cumulative102400(81920extra) is PREPARED,
  NOT QUEUED/STARTED. It owns the current runs/native_continuation.json pointer;
  do not mistake this for execution. Strong negative acceleration in the four
  stopped failures makes an unchanged small-noise long run questionable;inspect
  the terminal-state evidence before activating. Preserve its registration.

## Archived handoff before the geometry diagnostic

# Current handoff — 2026-10-31 09:21 local

Original goal ACTIVE/UNMET. Latest evidence: reports/exact_corner_compute_20261031.md and reports/completed_native_diagnostics_20261031.md. Source7dfbb3dcorner optimization has668full local passes/1skip/4CARLA exclusions164.55s;hosted41155398942passes653/16skip/4CARLA158.15s;independent7geometry checks4.41s,all549151hard-query flags/nodes/path unchanged. Public Hcapsule457files/6569controls/12actions/PNG also independently reproduced.

Two bounded postprocessors were launched:36796completed Hreevaluation,
23360waits new B1rollout1280. They only read the completed experiment/model,
audit raw frames/inference and save new reports;no driving or optimizer updates.
Each stops on phase failure or its10800-second cap,with no retries. Receipts in
runs/endpoint_postprocessors_actual_launches_20261031.json. These are running
local programs,not a claim that a model keeps reasoning outside the session.

- B1fixed units020728COMPLETE2/12,5bounds/1collision/4deadline;20480train,208Adam,7417eval;27897rawcontrols and12initial actions audited. Rejected. Public480file model/raw capsule uploaded/downloaded and independently reproduced27897controls/12actions/identical PNG.
- Repaired native control pair004553035320/MPC and004553454599/Pursuit COMPLETE4/4each,1904+1907RPC-bound controls. Both guard0/owned-empty stop. Pursuit lower lateral/speed error and jerk;MPCsmoothness result does not transfer. First024600guard-failed MPC1905controls retained/separately audited. Native1297file capsule public/downloaded and independently reproducedall5716frames/three metric reports/identical PNG.
- Hfrozen evaluation001732COMPLETE9/12,three deadlines,no unsafe,each family3/4;6569controls/12initial actions audited;zero new updates,retains20674training controls/650Adam. Postprocessor36796completed.
- Queue53904has activated fresh B1rollout1280batch004737;queue37576waits B1→H0-only001825;queue44456waits H0→seed22PPO013440;queue47244waits seed22→train36/validation24batch014801. All fixed wall/resource/cleanup guards remain.
- New B1rollout1280changes collection length only versus320control;initial tensors,empty optimizer,normalizer/RNG/config bytes independently equal;20480controls endpoint,all12development conditions,no checkpoint picking. ValidateOnly369files passed;TRAINING COMPLETE20480controls/87Adam,16phases;worker70952/guard48444currently performs all12frozen evaluations. Its frozen road kernel remains41fcb252;new root optimization was not applied.
- Latest confirmed full hosted source8afab8a649passed/16skip/4CARLA deselections194.18s;independent68checks15.95s. Latest b099b38full hosted run41152850977passes652/16skip/4CARLA deselections86.26s;38targeted independent tests7.11s. Both native guarded runs and independent public reconstructions passed.
- Old queue records below are superseded. Active plan receipts: runs/B1_rollout_owned_queue_launches_20261031.json and remaining runs/repaired_control_owned_queue_launches_20261031.json entries3/4.

## Archived handoff before this completion

# Current handoff — 2026-10-31 08:24 local

This supersedes all older live/queue statuses below. Original goal ACTIVE,UNMET.

Final current regression657passed/1skip/4CARLA deselections172.19s.
At08:25B1worker31216had7826controls;worker/guard33992and queue32916creation
identities verified live. This update is being committed/pushed;read git HEAD
for the exact new revision. Component raw-data public capsule remains pending.

- Actual B1fixed-unit batch020728is TRAINING,worker31216/guard33992,
  wrapper60176. At08:23it exceeded7000real controls. Original20k protocol/model
  unchanged;only queued activation clock refreshed after verifying all old
  queue PIDs absent,server process absent and RPC2100..2102empty.
- Shared-mask PPO013316STOPPED28Oct on exact Foreign actors appeared guard,
  during evaluation reset. Htraining COMPLETE20674controls/650Adam;partial
  evaluation10cases/7success/3deadline/4864controls,NOTa12case endpoint. H0never
  started. Guard1,owned-empty server stop and independent empty probe retained.
  Cause unknown;no unknown actor was adopted/destroyed. Diagnostics now capture
  unknown identities while preserving pre-tick refusal,even if snapshot absent.
- Fresh Hreevaluation001732(31Oct) PREPARED,zero updates,trained checkpoint
  retained;H0-only recovery001825 PREPARED,original untouched H0initialization
  retained,teacherNPZ hashed/carried. BothValidateOnly pass. Generic recovery
  --methods H0avoids retraining H;legacy case API compatibility preserved.
- Actual NEW queue order after active B1:H0-only001825,Hreeval001732,
  native MPC024600457573,native Pursuit024604590505,seed22PPO013440,
  train36/validation24PPO014801. QueuePIDs32916/44644/60116/36700/69492/67020.
  Inspect queue_execution_20261031.json,not old queue_execution.json. All child
  programs launched;none of these children activated yet. One server only,
  normal guard/owned-empty cleanup/lease/hash barriers,no automatic retry.
- Fixed-reference control CPU run024216COMPLETE80episodes,38962controls;
  40pairs,five INITIAL-CONDITION seeds,zero neural training seeds. Both40/40.
  Pursuit better lateral error/CPU cost;MPC lower jerk. Full state/tracking/
  termination/timing reconstruction passed. Original centreline run023315has
  20completed episodes/10190controls then invalid reference;retained,separate.
  Total integrator cost49152,NOTCARLA. Two native controller probes onlyqueued.
  reports/controller_and_native_recovery_20261031.md and figures give details.
- New raw controller binding verified1938HISTORICAL CARLAframes;not new pair
  execution. Controller-core full regression650/1skip/4CARLA exclusions162.29s;
  later ownership/recovery/partition58targeted6.24s. Current full rerun lives in
  runs/controller_recovery_full_cpu_20261031.log. Older pushed commit0653d81;
  current changes require final publication check/commit/push and independent CI.
- Goal still lacks healthy B1,5trained seeds,120NEWfinal conditions,PPOvalue,
  feedback performance benefit,complete ablations and formal paired statistics.
  Do not report completion or substitute controller-only80episodes for these.

## Earlier verified snapshots (historical)

Latest state supersedes snapshots below. Goal ACTIVE and all research gates UNMET.

Source6f9777b ACTUALLY PUSHED/remote verified;hosted40619710076 SUCCESS
620passed/15skips/4CARLA exclusions123.46s;independent30targeted8.38s.
reports/native_source_verification_20261028.json records precise source boundaries.
Current acceptance ledger/release status updated to28Oct;older0/4/smoke text
is historical only. Actual Hheartbeat6282controls at10:17local;guard healthy.

Critic013034 COMPLETE:**4/12**,6bounds/1collision/1deadline,20480training+
4128evaluation=24608raw controls.281actual Adam updates confirmed from every
saved state versus562nested post-hook calls. Raw counters/source retained.
Curve2/4,static1/4,dynamic1/4;no improvement over control4/12and more unsafe.
Reject as healthy. reports/native_B1_critic_rate_endpoint_20261028.json.

ACTUALLY TRAINING shared-mask seed11PPO013316,worker60368/guard69120,
queue/wrapper29896. Hheartbeat last4891controls;check live receipts.
Then ACTUALLY QUEUED seed22PPO013440(queue3268),train36/validation24seed11
014801(queue31656),and fixed-reward-unit B1020728(queue41236). All normal
guard/owned-empty stop/lifecycle lease/hash barriers,one server,no auto retries.

Fixed units is a separate fresh profile matched to standard-Adam KL140111:
same tensors,optimizer3e-5,config/RNG,obs RMS,physical source,rollout/epochs/KL
and original12cases;only learner rewards scaled.05. Raw task reward/metrics
unchanged;unsafe learner target-4.5fits initial critic range. NOTcombined with
fast critic or new scene partition. ValidateOnly369files;actual initialization
equality report in reports/native_B1_fixed_reward_units_initialization_20261028.json.
First preparation020529failed before registration/controls due detached SB3
pickle transient fields;partial retained with preparation_failure.json. New
serializer/real-SB3-rollout/frozen-RMS tests pass30related checks in6.85s.
No result exists for020728yet. scripts/prepare_native_supervised_ppo.py adds
--ppo-reward-scale .05;normalizer class/source is serialized and frozen.

Source4f35178 ACTUALLY PUSHED;hosted40618587881 SUCCESS615/15skips/4CARLA
exclusions196.94s;independent70targeted10.44s;rootfull629/1skip/4exclusions211.28s.
Those counts precede the fixed-unit additions;new additions have30targeted passes.
Do not claim all final goals complete or relabel development cases as final.

## Earlier verified snapshots

Latest additions:source345b1ac is actually pushed;hosted40616036339 SUCCESS
598/15skips/4CARLA exclusions194.63s,independent32targeted8.44s.
Critic013034is ACTUALLY TRAINING(worker46252),not just queued. Its frozen
custom Adam.step invokes the external post-hook twice per actual update;
checkpoint2560contains22actual Adam steps versus44hook calls. Rates3e-5/9e-4
verified from saved state. Preserve raw counters and correct endpoint reports
from Adam state;do not call doubled hook counts actual updates. Root optimizer
fix uses pre/post rate hooks and inherits Adam.step;19critic/partition tests pass.

New geometry-only train36/validation24inventory is registered in experiments/
native_development_train36_validation24_v1.json;no final-test claim. Four new
validation roads are separated from132historical registered route inventories
by>=12m centreline clearance. Source supports separate collection/evaluation
inventories and separate raw-reset audits. New H/H0seed11batch014801is PREPARED,
ValidateOnly388source files;not yet launched/queued. Parent original12teacher
conditions retain exact indices;weights,normalizer,reward and low level unchanged.
reports/native_training_validation_partition_20261028.md details evidence.
Latest full source rerun in runs/native_partitions_count_fixed_full_cpu_20261028.log;
earlier625passes predate last additions;one later rule fixture mismatch is fixed.

UPDATE10:02local:complete current CPU **629passed/1skip/4CARLA exclusions211.28s**.
Partition batch014801ACTUALLY QUEUED after013440,queue31656,receipt/script in
runs/queue_train36_validation24_after_seed22_PPO_20261028.ps1. No duplicate launch.
Critic013034TRAINING COMPLETE20480controls/281actual Adam steps,562raw hook
calls;all parameter-state steps agree. Frozen12case evaluation active. Critic
output bound14.4825still cannot cover raw-90unsafe terminal target. Preserve
the result before choosing the next minimal learning change;do not blindly add
another long continuation. reports/native_B1_critic_rate_training_20261028.json.

Common-mask seed22BC003323 COMPLETED H10/12,H012/12,zeroPPO. Both audited.
Its actual PPO013440is queued after shared seed11PPO013316,which is queued after
critic013034. QueuePIDs3268/29896. Do not duplicate activation.

Latest state supersedes the older snapshot below. Do not duplicate active queues.

- B1 continuation235448 COMPLETE:81920new training controls,102400cumulative,
  853new Adam steps/1052cumulative lowerbound. Frozen **2/12**,7bounds/1collision;
  not healthy. reports/native_B1_long_PPO_endpoint_20261028.json binds87220raw
  segment controls. Old inherited ledger omits7123known initialization-evaluation
  controls; retained and separately disclosed. No more continuation of this model.
- Recovered body-spline B2000521 COMPLETE:**7/12**,6357new eval controls,0new
  learning. Prior1486partial controls charged. Raw audit/guard0/owned-empty stop
  verified. reports/native_B2_body_spline_endpoint_20261028.json.
- Common-mask seed11BC002857 COMPLETE:H/H0**11/12each**,0PPO. H reused exact;
  only H0 refitted896actor steps. Shared original physical endpoint mask and raw
  information; learned H0actor AND critic feedback zeroed internally. Both raw
  audits complete. New result does not establish feedback performance benefit.
- Common-mask seed22BC003323 ACTUALLY evaluating,queue38856. Critic-rate B1
  013034 is ACTUALLY QUEUED after its normal guard/owned-empty cleanup,queue63380.
  Fresh phase4BC weights/norm/RNG/config and physical sources match140111 EXACT;
  actor3e-5,critic9e-4,20480controls,7200swall,KL.01. Only disjoint critic Adam
  rate differs. No trained optimizer reset. ValidateOnly passes369source files;
  reports/native_B1_critic_rate_initialization_20261028.json records equality/costs.
- Shared-mask seed11PPO013316 PREPARED,not yet activated. Both original complete
  representation gates pass.20480physical controls per H/H0,guarded reward,
  original categorical exploration and frozen teacher stats. Queue after013034.
- All15actor-supervised initializers for fixed seeds11/22/33/44/55 are complete;
  zero PPO at initialization. Paired H/H0critic/norm/data exact,networks distinct
  across seeds. reports/native_fixed_seed_initializations_20261028.json.
- Fair-mask full CPU regression610passed/1skip/4CARLA exclusions153.16s.
  Critic-rate30targeted checks pass5.70s; full current regression612passed/1skip/4CARLA exclusions143.83s,recorded in
  runs/shared_mask_critic30_full_cpu_20261028.log. Runtime batch sources frozen.
- Cfree22.7GB,last measured;12GiBfloor retained. Only one CARLA server at a time.

Still UNMET:healthy B1,PPOvalue vs strong rule,feedback benefit,5independent
trained policies,120NEWpaired final conditions,train/dev/final separation,
unseen transfer,remaining ablations and locked statistical confirmation.
User objective and all final acceptance gates remain unchanged.

---

# Current handoff — 2026-10-28 08:10 local

Continue the original thesis objective;no restart,old plant,lower final gates,
duplicate activation or false completion. Historical long handoff remains in
../reports/handoff_before_native_twelve_case_20261027.md;Git preserves later states.

## Actual development evidence

All12conditions are FOUR KNOWN ROADS ×curve/parking,staticcar13m,dynamicwalker.
They are not120independent final conditions. Earlier10mstatic failures remain.
Rule08011112/12;actor-supervised094138H/H0each11/12withzeroPPO.
RealPPO100609H11/12,H010/12:20541/20623trainingcontrols,650Adam each,
51429totalrawcontrols audited. Hretains BCscore;H0losesone. StrongB3is12/12;
PPOvalue,feedbackbenefit and independent hierarchy advantage are UNPROVEN.

B1pooledBC0950434/12,orderedBC1024453/12,phase4BC1124466/12.
Corrective1245427/12butfourbounds/basic2/4;zeroPPO,rejected initializer.
Its two infrastructure interruptions retain3578+4030controls;full replay4965.
Snapshot cleanup crossed the old actor57failure and completed owned-empty stop.

B1realPPO1307484/12after20480controls/640Adam,7857eval/28337currentrawframes.
Matched KL-stop140111also4/12after20480controls/199Adam,7673eval/28153raw.
KLstop preserves curves3/4versus1/4but losesstatic2/4to0/4;dynamic1/4both.
No overall gain or healthy-baseline claim. Parent BC6/12;all failures remain.
../reports/native_B1_KL_PPO_endpoint_20261028.json.

Frozen H-to-B2legacy1347234/12,8639evalcontrols,zero new training.
Eight deadlines are dominated by failed planning/braking;not fair upper-bound
B2capability. Existing body_spline_v1 repairs all8offlineinitial open/static
queries versus2/8legacy;computed references are not driving success.
Body-spline141335 STOPPED by exact Disk below12GiB guard after1486controls,
guard1/reasonNone,owned_empty_server_stopped. No complete B2score.

## ACTUALLY ACTIVE / QUEUED — inspect receipts before action

ACTUALLY TRAINING:20261027T235448Z_native_continuation,worker22840/guard14288,
wrapper60012. Parent140111exact policy/Adam/norm/RNG/settings;81920NEWcontrols,
target102400cumulative,10800swall,deadline1793156543.444(~11:02local).
Current progress is in B1/heartbeat.json;report actual controls,not intention.
All12frozen endpoint,zero extra supervision,0automatic retries. This is larger
B1tuning budget than currentH20541,NOTan equal-budget research comparison.
Parent evaluation7673now charged;registered teacher NPZ retained by hash.
Old helper48844 stopped on the B2failure;do not restart its old barrier.
New actual launch:runs/native_KL_B1_long_manual_launch_20261028.json and
runs/native_KL_B1_long_launcher_20261028.log. Queued activation preserves
original protocol/prepared snapshots and changes wallclock only.

ACTUALLY WAITING:20261028T000521Z_native_evaluation_recovery,queue64524,
after235448normalguard/owned-empty cleanup/lifecycle release. All12B2conditions
will rerun with exact saved model/norm/RNG/config,existing body_spline_v1,
zero updates. Prior1486controls charged:31269inheritedinputs,650retainedHAdam.
runs/queue_recovered_B2_after_long_B1_20261028.ps1 and child's queue_execution.json.
../reports/native_B2_disk_recovery_20261028.json. Never duplicate this phase.
Grid search is finite-grid but does not enforce Hybrid max_nodes;record actual
nodes,timings and150spline evaluations;do not claim equal compute or realtime.

Recovery PID29180was reused after old guard exit. Staging now reads creation
time and accepts only a verified NEWinstance;unknown/original live processes
still reject. No foreign process adopted/terminated.44targeted parts/PID/
interruption/prefix checks pass4.45s. Runtime training is its frozen source.

Seed22initialization COMPLETED:144307hierarchicalBC(H/H0),144319measuredBC(B1).
896/896/1792actor-supervised updates;zeroPPO,zero seed22CARLAevaluation so far.
Every measured array/task/norm statistic matches its seed11counterpart;
policy tensors differ and PPO optimizers are untouched. Not completed
independent CARLA policies. ../reports/native_seed22_initialization_20261028.json.

## Storage and public reproduction

Closed runs/prerelease_20261022 moved to dedicated Dproject archive with
original Cpath junction;46410files/4914560944bytes verified. Read-only Git-object
cleanup interruption was retained;all46134remaining source files reverified
before duplicate Cbytes retired. Original interpreter path actually imports
Python3.11.9/Torch2.14.1+cpu. Other old large runfolders were ALREADYjunctioned
to their existing Darchive;preserve them. ../reports/native_storage_recovery_long_training_20261028.json.
The local detailed relocation receipt has the destination. No simulator reinstall.
Keep disk12GiB/RAM4GiB/GPU85C floors and fixed wallcaps. CARLA0.9.15onD,
main.venv/.carla-python37,actualRPC2100;never kill foreign processes by port/PID.

Public/downloaded/independently reproduced assets already cover realH/H0PPO
51429controls,rule/BC18591controls,and legacyB28639controls,including frozen
start actions and byte-identical figures. See model_card.md and associated reports.
New negative B1PPO130748capsule is now ACTUALLY uploaded as6smallparts+manifest,
downloaded WITHOUTAUTH and reassembled byte-exact:43808905bytes/477files,
SHA b97dcb80ed2bf92f7898e39780c6fa52c77346304c26eaf93f2dc770b27f533c.
Two earlier whole-body uploads returnedHTTP408and left starter assets;they are
not downloadable successes. Cleanup command was auto-approval-rejected;records
remain. ../reports/native_B1_PPO_public_parts_20261028.json. Independent checkout8478f19has COMPLETED parts/model/raw/figure reproduction:all28337controls,12startactions,frozen stats,byte-identical figure.

Mechanism audit COMPLETE:all497H/541H0decisions reproduce from measured histories. Fixed-original-mask zero-all-feedback changes11/497Hactions;zero-continuation5,zero-outcomes8. Removing mask changes18/497. H0negative-control feedback changes0/541. See ../reports/native_feedback_dependence_20261028.json;no new controls/updates or closed-loop causal claim. Two targeted helper/policy tests pass4.73s. Existing H/H0mask/feature confound motivates a NEWversion with common physical endpoint mask and internal learned-feedback ablation;do not rewrite old outcomes.

## Source / remaining full target

Correctremotehttps://github.com/Leo-7GOAT/Intelligent-driving-system.git,
branchfeature/physical-hierarchical-ppo,authenticatedLeo-7GOAT;no forcepush.
Existing prereleasev0.2.0-dev.20261022is development only,not final acceptance.
Latest actually verified hosted sourceae3d238passes573/15skips/4CARLA exclusions
194.80s;independent20targeted10.96s. Earlier8f8257afullindependent581/1skip/
4exclusions285.46s. Source8478f19hosted40607068483SUCCESS591/15/4in110.14s;root44targeted4.45s/independent44in5.45s. New mechanism-audit source separately has2targeted tests. Independent checkout and locked environment remain at
their original runs/prerelease_20261022paths through the verified junction.

Still UNMET:healthy sufficiently trained B1;PPOvalue vsstrongB3;explainable
feedback contribution;5independent trained seeds;120NEWpairedfinalconditions
with>=30each core family;unseen transfer;trainedB2/frequency/shared-safety/
fixed-reference-control ablations;locked final protocol and paired intervals.
Targets unchanged:light85%/75%core,CARLA80%/70%core,unsafe<=5%,H-B1>=10pp with
95%pairedlower>0(or prelocked efficiency hypothesis BEFORE final testing).
Do not treat the current12known conditions,greenCPUCI,publication or elapsed
wallwindow as completion. Keep goal active and continue from actual receipts.
