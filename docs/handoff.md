# Shared parking v2 engineering screen COMPLETE; research UNMET — 2026-11-01

All native lifecycles in this goal turn are TERMINAL,ownedemptyserverstopped.
Goal ACTIVE/UNMET. Do not restart closed probes or the abandoned prepared Hv1.
Root CPU1127PASS/1skip/4CARLAexcluded140.04s on v2;factory+full parking completion
regression tests passed. Latest pushed code before pending commit is567241ac...
CI41294888160SUCCESS1097PASS/18skip/4excluded. New source/reports need commit/push.

COMMON EXPLICIT PARKING AID v2: goal_parking_assist=True,method-independent,
physical full brake ONLY inside unchanged goal position/yaw gates,after actual
travel>.1m,and while abs(v)>existinggoal_speed. Steering unchanged. At parked
speed RELEASE to ordinary policy/controller. Original _event and its existing
allowed_success=not intervention guard are UNCHANGED. All successful terminal
frames must be nonintervened and satisfy measured original pose/speed/yaw gates.
The option defaults off;historical serialized configs/signatures remain exact.
Main Native factory propagates it to H/H0/B1/B2/B3. Rootlauncher blocks an
abandoned_before_launch.json marker before any lifecycle.

Verified shared v2 transfers on SAME4known TRAINroads (not12newconditions):
B1 batch20261101T104233573082Z_native_shared_parking_v2_B1:
  4/4,zero unsafe,1409controls,6parkingoverrides;cases356/427/321/305.
H batch20261101T104236093121Z_native_shared_parking_v2_H:
  4/4,zero unsafe,1287controls,8parkingoverrides;cases335/344/310/298.
B3 batch20261101T104917Z_native_rule:
  4/4,zero unsafe,1286controls,8parkingoverrides.
Same helper bytes,calibration,vehicle and goal criteria independently audited
in reports/native_shared_parking_engineering_20261101.json;3982v2controls.
All engine acknowledgements match;all raw physical frames verified. H/B1 starts
reconstructed;B1 whole-sequence verification COMPLETE:all1409actions bit-exact. No new training/BC in these frozen transfers.
B1model72d465ab4e553c189fd04c5e472a5fcc7cc0eafa371ee7c5c62f6eb9e0ab5844;
Hmodelc46da0e23773fad3856dbc77b7681fcb5e2b32ee02976041a67752dbe9213d36.
B1lineage30720actualPPOcontrols/H20579;NOT budget-matched research evidence.
Models were NOT trained under this wrapper;formal shared-wrapper comparisons
must retrain related methods. Do not call native unassisted B1 qualified.

V1 integration bug RETAINED:20261101T103609019454Z_native_shared_parking_B1,
1/4,3deadlines,3021physicalcontrols. Aid kept intervening while already parked,
so original terminal guard correctly suppressed success. V2 fixes release,
not goal criteria. Prepared H v1 batch20261101T103652093173Z_native_shared_parking_H
ABANDONED_BEFORE_FIRST_LIFECYCLE,0controls,markerblocksRootlauncher. Never launch.
Engineering report includes all7003controls across v2three arms+failedv1.

Earlier actual PPO course20261101T094700825994Z_native_B1_feedback_PPO completed
10240newphysicalTRAINcontrols,320actualAdamsteps in all35saved parameter states,
80PPOepochs,RMSmean/var/count exact. TRAIN15success/9bounds+recordedtail;
endpoint2/4,bounds438/634,success311/310. All11933controls strictsync+rawverified;
1693endpointactions bit-exact from measured history. Model eb2c9daa5b6ccf025429bdb1a1f69c55c7d8a8eecec65dd39cff580546f235ec.

Reflection augmentation:3424acknowledged originalteacherTRAINrows+3424virtual
reflections,120original/reflected pose triplets. Original architecture/critic/
PPOAdam/counters/RMS retained. First512auxAdam rejected;exactAdam+sampler512step
continuation(1024total)passed. Nativepure20261101T100704463570Z_native_reflection_feedback_probe:
3/4,leftcases0/1andcase2success373/441/321;case3bounds406,1541controls. Case3
passes within.105mofgoal at1.04m/s with positive requestedacceleration. Model72...
Unassisted gate UNMET. This is the reference for shared-parking B1.

Separate rejected NNacceleration experiment:108hypothetical velocity queries
from27measured teacherstates (not108measuredstates),all4TRAINroads near<=3m;
1297golden held-reference commands exact. Model policy adds177new acceleration
parameters only;all old weights and steering outputs exact. First512headAdam
misses braking-sign gate;exact512step continuation passes100%critical18queries,
queryMSE.57238→.00552. Actual20261101T101916808465Z_native_terminal_acceleration_probe
still2/4:bounds0/2at456/411,success1/3at422/405=1694controls. Modelcb84e591...
REJECTED,not used for shared parking. Prepared protocol had a prelaunch count
correction:27source states generate108hypothetical rows;all original metadata
snapshots preserved. No physical/policy inputs changed by that correction.

Root additions:parking_assist.py/config/env/experiments/nativefactory;newoptional
terminal_acceleration_policy.py is rejected-model reproduction only. Original
flat-sequence auditor v1 is retained byte-exact as audit_native_flat_sequence_v1.py;
currentauditor also reconstructs intervention feedback,needed for shared wrapper.
Docs shared_parking.md explain all limits. No new final conditions generated.

CURRENT NEXT WORK: finish commit/push,independent source/report verification,
package/publish/rebuild the concrete shared engineering evidence using existing
package_native_continuation.py and package_native_rule_evidence.py. Do not
repeat earlier public downloads/cohorts. Then actual5seed training under fixed
shared-wrapper protocol,addedPPOvalue,strongB3comparison,collaboration gain,
remaining fair ablations and120NEWfinalconditions remain mandatory. The global
research goal is not complete. No native training currently runs or is queued.

# Pose response diagnosed; real PPO basic course RUNNING — 2026-11-01 17:50

Goal ACTIVE/UNMET. Do not restart old closed jobs or package a failed fit as a
healthy B1. Sync-delivery source928bdbdb554488b1d16bbf2a94e8d7714e8e4a59 is pushed;
independent checkout1111PASS/1skip/4CARLAexcluded113.78s;hosted CI41293589581
SUCCESS1094PASS/18skip/4excluded128.70s. Independent transfer report deep-exact.

New measured TRAIN diagnosis: 1297 golden held-reference commands reproduce;
30 fixed sampled states have policy/MPC median absolute lateral gain ratio.0747
and yaw ratio.1397. Thirteen lateral/four yaw signs disagree. Poses are hypothetical,
not executed labels. Dataset60triplets(base/minus/plus),all4known TRAINroads.
Original states/history unchanged; no plant step exists in query reconstruction.

Rejected ten-parameter geometry residual: one ridge solve,zero native controls;
modelb517a23b...,source retained solely for reproduction,not adopted in next NN.
Original-architecture contrastive actor fit512steps fails lateral response screen.
Declared1024step extension passes: lateral error.058121→.026020;yaw.052566→.020301;
all4teacher steering RMS<.01rad,acceleration anchor RMS.004609. Original critic,
roadencoder,log_std,PPOAdam/counters/RMS exact. New auxiliary1536Adamupdates total.
First auxiliaryAdam was not saved; extension restart explicitly disclosed. Saved
extensionAdam states all1024steps. Original replay512auxiliarysteps are inherited.
Model2e5925184e4f6eb560d96724526d51dec8cdf4f5c37314e15af3f7d53e46d53b.

Pure fixed-sync deployment20261101T094244452999Z_native_feedback_contrast_probe
COMPLETE/ownedemptyserverstopped;case0bounds489,case1bounds665,case2success320,
case3success308=2/4;1782controls,zeroackmismatches,4measuredstartactionsverified.
Gate UNMET. Case1closestgoal now.991m vs originalsync10.235m,but overshoots;
case0closest.868m and also fails. Do not call local feedback a driving improvement.
Full report reports/native_B1_feedback_contrast_probe_20261101.json. NewrawLOCAL.

ACTIVE actualPPO batch20261101T094700825994Z_native_B1_feedback_PPO,
lifecycle execsession65170,worker41328(identity+creation/exe in worker_process.json;
never use PID without fresh identity checks). Started by owned foreground wrapper
with current guard,7200swall,0retries,C>=12GiB,RAM>=4GiB,GPU<85C. Budget10240
new physical TRAINcontrols,thresholds1280..10240,all4TRAINroads at final endpoint.
Source mean2e592518...,Gaussian log_std retained(~.0183),zero extra BC/runtime
teacher,freezeRMS,unchanged reward/goals/20Hz/gains,direct original architecture.
Lowlevelbatch_sync_v1 fixed for entire new training. Charged prior controls86628,
inherited PPOlineage20480/PPOAdam348. This is basic learning,not full-task promotion.
DO NOT launch the registered batch again. Inspect heartbeat/guard/summary until
terminal,then raw audit,measuredstartaudit,new delivery audit and
scripts/analyze_native_feedback_ppo.py; no result has been claimed yet.

Current newly added policy/control related tests26PASS14.47s;additional learning
proof test needs execution. New source/report edits need commit/push and checks.
Reports copied byte-exact; .gitattributes protects their bound hashes. New source
scripts include prepare_native_feedback_ppo.py and analyze_native_feedback_ppo.py.
No new final conditions generated. Five healthy B1seeds,actualPPOaddedvalue,
strongB3benefit,explainable collaboration,fairremainingablations and120NEWfinal
conditions remain unmet. Do not change the research goal or lower gates.

# Sync delivery fix verified; B1still unqualified — 2026-11-01 17:20

Previous goal turn PROGRESS:actualPPO/public reconstruction complete. Thisturn
finds a real frame-contract gap:SDKreturned controls differ from requested pedals
in51/12219pilotframes(47TRAIN/4eval),allmatching previousrequest. Historical
frame audits checked measuredstates+requests/bridge echoes,not engine-equality.
Do not erase old results or claim these lags explain their policy failures.

NEW explicit calibration field recommended_adapter.control_delivery=batch_sync_v1.
Ego control goes through apply_batch_sync(...,False)beforetheoneownedphysics tick.
Failed submissions reject;inconsistent post-tick control is preserved in rawreply
then aborts the client. No extra tick,actuatorgains/goal/observation/rule changes.
Legacy delivery remains the default for historical compatibility. New training
must explicitly use the versioned corrected low level,not silently reuse legacy.

Completed transfer20261101T084713744547Z_native_sync_delivery_transfer:
fixed4f205708replay policy+db467normalizer,all4original TRAINroads,zeroPPO/BC.
1271controls,allengine controls/gear/handbrake match;strict delivery gatepasses.
Case0success382,case1bounds256,case2success312,case3success321=3/4. Legacy SAME
checkpoint was3/4with case1bounds264and5/1275engine mismatches. No driving benefit
or healthy baseline is established. All1271rawframes+4startsverified;guardnormal0/
ownedemptycleanup. Lifecycle tool6157TERMINAL,no native job remains active.
Onlyego delivery was tested;dynamic actor controls remain untested by this fix.

Rootfull1110PASS/1skip/4CARLAexcluded114.36s beforetheadditionalclock-helpertest;
currentcontrol unit10PASS2.04s and23relatedactuationpartsPASS10.37s. Source/code/
reports for this fix need commit/push at this snapshot. Do not rerun closedprobe.
Reports:native_PPO_legacy_control_delivery_20261101.json and
native_sync_control_delivery_transfer_20261101.json;docs/native_control_delivery.md.
Newtransferraw remainsLOCAL;earlierPPOpublic563filecapsuleunchanged.

Additionaldiagnostics:1334measured original-TRAINmixturestates,fixed4fpolicy,
no new controls/fits.128naturalwitnessfliptransitions isolated actionRMS.03764acc/
.01498steer(normalized),versusfulladjacentRMS.144/.0353. Clock±5swith>=20sremaining
changesactionRMSroughly.028-.037acc/.008-.009steer. These are offline sensitivity,
not causaldrivingbenefit. No input masks adopted. Reports native_B1_witness_jump_
diagnostic_20261101.json and native_B1_clock_sensitivity_20261101.json.
A separate readonly spot-check on4131TRAINrows finds negative acceleration-vs-
speed slope on all3318cruiserows,so do not assume positive speedfeedback. Original
teacherinputs append obsBEFOREcurrentlabels;there is no direct current-action
leak(only11.8%acc requests equal previouscommand). The lastteacher steeringRMS
imitationerrors are~.0036-.0050rad;small pointMSEstilldoesnotproveclosedloopgains.
Do not blindlychangebrakegains:low-speedbrakingdiscontinuitiesarealreadyknown
from20261022;−2.4emergencythresholdwaspreviouslyverified,not an exact−2.5boundary.

GOAL ACTIVE/UNMET. Need a strong healthy5seedB1,added-PPOvalue,strongB3comparison,
explainablefeedbackbenefit,fairablationsand120NEWfinalconditions. No finalset
activated,gatechanged,failedmodelpromoted or ordinarynewtrainingqueued.

# FINAL verification checkpoint — 2026-11-01 16:37

Source5ba24415285cad9413b0580903762f1adde18caf isPUSHED and fullyverified:
root1101pass112.76s;independent checkout1101pass112.25s;both1skip/4CARLAexcluded.
HostedCI41289096637SUCCESS1084pass/18skip/4excluded. Committed verifier f46ae098...
rebuilds the public initial+learning proof exactly;Git recipe SHAalso exact.
All public download/raw/model/figure/Adam/RMS checks complete. No native,training,
upload,reconstruction or CPUtest job remains active. No queued next training.
Only this final metadata record changes after the tested source;no algorithm,
model,raw control,scene,normalizer or figure changes. Overallgoal ACTIVE/UNMET:
B1still2/4with bounds/timeout;five-seed healthy baseline,PPOvalue,stablefeedback,
fair remaining ablations and120NEWfinalconditions unfulfilled. No gate relaxed.
Do not repeat the published cohort,pilot,closed fits,downloads or tests.

# Public PPOpilot reconstruction COMPLETE; research still UNMET — 2026-11-01 16:27

Independent checkout9e726d7 downloaded public7parts+manifest unauthenticated:
archive438778411ba8cba4faf2708c3e07daa39b70c8597927f6db18fc55a1118bdbcf,
56422535bytes,563files;all12219controls+4start actions rebuild. Explicit Git
figure08f30f30e4e7b52dd9fc9b2087b2a6ef480b8e06df154e8bcef6f434006ebe4a
is byte-identical. Original4f205708and intervenedcfde4844initial policies/Adam/
RMS/RNG/config reconstruct exactly. Additional independent locked-environment
verification confirms all35PPOAdam states gain320steps,action timesteps+10240,
PPOepochs+80,mean-network weights changed,andfinalobsRMSmean/var/count EXACT.
Root final fullCPU1101PASS/1skip/4CARLAexcluded112.76s. Hosted9e726d7CI41288185210
SUCCESS1084pass/18skip/4excluded. New read-only verifier enhancement is exercised
on the public restored models and covered by the root fullsuite.

Official receipt:reports/native_B1_exploration_PPO_public_verification_20261101.json.
Machine recipe:protocols/public_native_B1_exploration_PPO_20261101.json.
Docs:native_B1_exploration_reproduction.md. Public assets/notes/readback/server
SHA256s all verified;release remainsprerelease andtargetba932787unchanged.
No new physical controls/optimizer updates during public reconstruction.
All native jobs and upload/download/reconstruction tool sessions are TERMINAL.
No queued next training. Root final metadata/verifier changes need commit at
this snapshot. Preserve all completed jobs;do not repeat current public checks.

Research goal ACTIVE/UNMET. Current std.10PPOpilot2/4,1bounds/1deadline despite
10240new controls/320actual Adam. Three earlier recovery hypotheses also fail.
Totalthissequence21229physical controls/1536auxiliary Adam/320actual PPOAdam.
No full-task entry/final120activation or model promotion. Do not queue another
BCweight/epoch/std-only sweep or automatically add budget to this failed version.
Next useful evidence is measured terminal overshoot/action uncertainty or
actor-vs-critic gradient analysis from already recorded TRAINrollouts,before a
separately predeclared new baseline training version. Same original goals.

# PPO pilot COMPLETE AND FAILED; all native jobs terminal — 2026-11-01 16:14

Latest286e72f is PUSHED;CI41287114006SUCCESS1084pass/18skip/4CARLAexclusions
197.68s. Independent47tests8.85s and all3recovery reports byte-identical.
Current actual-PPOreport:reports/native_B1_exploration_PPO_pilot_20261101.json.
Batch20261101T075805697090Z_native_B1_exploration_PPO completes10240fresh controls,
320actual Adam,8phases,lineage30720;model9ae73aba...normalizer1a9ded9a...
Pure4roads:bounds420,deadline900,success318,success341=2/4. GateFAILS.
All12219raw controls+4measured-start actions verified. Guardnormal0/owned empty
server stopped. Exec48771TERMINAL;no current simulator/training job active.
Do not repeat this pilot or queue more budget for the same rejected hypothesis.

All3earlier recovery fits also fail;all9010controls audited. Totalthissequence
21229new controls,1536auxiliary Adam proposals plus320actual PPOAdam. No new
final conditions. Unprojected fit changes the auxiliary TRAINMSEscreen only;
actualdrivinggate never relaxed. No failed model promoted or full-task gatewaived.

The original pre-intervention replay checkpoint is now retained inside
B1/initial_reference_B1. scripts/audit_native_exploration_initialization.py
reconstructs std.10in memory and verifies every initial policy/Adam tensor,
counter,normalizer,RNG/config byte. Rebuild proof is complete,zero new controls.
Capsule563files/56422535bytes/SHA438778411ba8cba4faf2708c3e07daa39b70c8597927f6db18fc55a1118bdbcf;
all7parts+manifest actually uploaded toexisting v0.2.0-dev.20261101,tagtarget
ba932787113168ad5781ee640c6f06d09bbee0e1unchanged. Independent public download
and reconstruction are the immediate remaining publication check at this snapshot.
Use explicit versioned endpoint figure --reference-figure;archive predates this
figure and has only the separate training plot. docs/native_B1_exploration_reproduction.md.
Raw collection/recovery capsules remain LOCAL;only this PPOpilot newly public.
GOAL ACTIVE/UNMET. Do not imply healthyB1,PPOvalue,feedback advantage or final120.

# Actual PPO exploration pilot RUNNING — 2026-11-01 16:00

GOAL ACTIVE/UNMET. Recovery-supervision path is now closed for this dataset.
All3new profiles fail pure driving gates:2/4with1bounds(new TRAINroads),
2/4with1bounds(original TRAINprojection),1/4with2bounds(unprojected matched
ablation). All9010new physical controls audited;1536new auxiliary Adam proposals,
zero PPOin those3profiles. Projected/unprojected input arrays,parent,weights,
seed and512proposal budgets match exactly. The auxiliary TRAINMSEscreen differs,
not the unchanged >=3/4ANDzero-unsafe driving gate. No failed model promoted.

ACTUAL running job:20261101T075805697090Z_native_B1_exploration_PPO.
Worker61460,creation1793519928.466;revalidate creation/executable before any
process management. Lifecycle+post-audits execute in tool session48771.
At16:00:28:1485actual new PPOcontrols,first1280rollout COMPLETE with40actual Adam
updates,actorL2change.0091584. Target10240new controls,final lineage30720,
wallcap7200s,no retries. One CARLAserver/RPC2100,reused isolated client/runtime.
Only initial2Gaussian log_std values change from~.018to.10;fixed replay4f205708
mean network/critic/PPOoptimizer/counters/RMS/RNG retained. Actual PPOuses same
3e-6rate,1280rollout,10epochs,batch320,reward,physical hold1/20Hz interface.
NOteacher or extra BCduring training/evaluation. All9010recovery controls charged.
Initial deterministic replay NOT driven again. This pilot is not yet completed
or a healthy baseline. Do not start another server or duplicate this job.

After normal owned cleanup,the existing shell AUTOMATICALLY runs frozen raw
endpoint audit,4measured-start inference and frozen prepare_native_exploration_ppo.py
--analyze into reports/native_B1_exploration_PPO_pilot_20261101.json. Preserve its
failure if any;no blind retry. Monitor B1/heartbeat.json,optimization.jsonl and
checkpoints. Initial activation receipt is a dated snapshot,not perpetual status.
All5independent basic courses must qualify before full-task entry remains allowed.
Next action is observe this actual learning endpoint and audit;do not add more
BCweight/epoch sweeps or regenerate any known public cohort.

Previous source4e2e25f is PUSHED;hostedCI41285718700SUCCESS1074pass/18skip/
4CARLAexclusions200.39s. Rootfull1091pass/1skip/4excluded156.11s at that source;
independent21targeted and first recovery report byte-identical. Current new
original-TRAIN/ablation/PPOpreparation sources and reports need the next commit.
Raw new models/traces remain local;do not commit private inference roots.

# Latest native recovery collection and pure-policy probe FAILED — 2026-11-01 15:38

All work below is actually executed,not a pending collector plan.
Collection20261101T073000482834Z_native_TRAIN_recovery_collection:1612controls,
850teacher/762learner,3success/1deadline,zero unsafe. Four new TRAINroads
indices4..7,never24VALor final conditions. All1612measured observation arrays,
learner proposals,source choices and physical control frames independently
reconstructed by scripts/audit_native_recovery_collection.py. No optimizer
updates during collection;PPO/optimizer/RMSfingerprints identical.

Fit runs/B1_seed22_executed_native_recovery_fit_20261101:3467original executed
teacher TRAINrows+850new executed teacher labels,all outcomes,episode-balanced
sampling,512new separate actor Adam proposals with prelocked four-group
retention backtracking. Gatepasses,but is not a driving result. Model2f7287e9...
retains20480PPOlineage and unchangedcritic/PPOoptimizer/counters/RMS.

Pureprobe20261101T073504221792Z_native_recovery_pure_policy_probe:2/4,
case0success350,case1deadline900,case2success318,case3bounds89,total1657.
Allrawframes+4measuredstarts verified. NOteacher in pure evaluation.
Original≥3/4ANDzero unsafe gateFAILS. ModelisNOTpromoted;do not start the
five-seed full-task phase from it. Totalnew3269controls,512auxAdam,zeroPPO.
BothlifecyclesTERMINAL/normal0/ownedemptyserverstopped. No native job running
at this snapshot. Revalidate PID identities before any future management.

Report reports/native_B1_executed_recovery_collection_and_probe_20261101.json;
docs/native_recovery_collection.md lists commands and limits. Raw collection,
models and pure traces are LOCAL,not a public reconstructed capsule. Never
commit raw measured_start_inference.json with its private absolute root.
Root full CPU:1091PASS,1skip,4CARLAdeselect,156.11s;prior75609efhostedCIalsoSUCCESS.

Do not repeat this collection,fit,probe or the already public five-pair cohort.
New-road recovery supervision did not fix original-TRAINclosed-loop coverage;
next inspect failure histories and,if collecting again,declare all four original
TRAINroads as a separate fixed recovery profile with fresh source provenance.
No further train-MSE-only weight sweeps. Original research goal ACTIVE/UNMET.

# Latest native repair FAILED / next required work — 2026-11-01 15:20

The executed-teacher-only repair native probe is COMPLETE AND FAILED the
unchanged basic gate:3/4,case1out_of_bounds at184controls. Other controls362/310/326,
total1182;0new PPO. All raw controls and4measured-start actions verified.
Raw audit SHA703b285f1481dc892c1592010e7c7994f637568504d1b5b698c73afd9f4a5615. Official report `reports/native_B1_executed_teacher_only_repair_probe_20261101.json`.
Model6d1ebeb3... is NOT promoted. Worker72792/lifecycle exec15759TERMINAL,
normal exit0/owned-empty cleanup. ALL native queues now terminal;no current
training or simulator job remains active. Revalidate any future process IDs.

DO NOT repeat closed-loop replay3/4,99correction2/4,or newclean repair3/4.
DO NOT add more offline label weighting/fits:four such hypotheses were tried,
three CPUrejected and one nativefailed. New teacher-only fit used3467actual
teacher TRAINrows,excluded29counterfactual rows(weight580),512auxiliaryAdam,
retained20,480PPOcontrol lineage/critic/PPOoptimizer/norm. Better teacher MSEin
all4groups did NOT fix closed-loop safety. Preserve this negative evidence.

Next useful step is genuine native ON-POLICY recovery data/closed-loop
competence,not another train-MSE-only fit. Use existing measured bridge,
RecordedRoadEnv and HeldTeacher(persistent2Hzplan/20HzMPC)to collect a bounded
explicit teacher/learner mixture on preselected TRAINroads only. Keep PPO/BC
updates0during collection;count actual teacher-driven and learner-driven
controls separately. Mixed success is NOT a learned-policy result. Evaluation
must remain a SEPARATE direct-model-only run. Do not feed mixture traces into
ordinary policy-inference auditing as if every action came from PPO. Need
explicit collector role/provenance and real-frame command audit before fitting.
Choose unused TRAINgeometry from existing36TRAINinventory,never the24known VAL
roads/futurefinal set. All original H/B1fairness/safety/5seed/final120goals stay.
No new collector has yet been implemented or started;do not claim it is running.

Source/current evidence pending commit at this snapshot:37targeted root tests
PASS,new data/retention guards/persistent-label tooling,completed teacher117/120
attribution and all failed B1results. Fullnew CPUCIand independent37testrun
should follow commit. Old latest full CI1036PASS. Root native source/plant did
NOTchange;only diagnostic/auxiliary-fit/preparation tools changed.
Goal ACTIVE/UNMET:healthyB1,PPOvalue,stablefeedback,fairremainingablations and
120NEWfinal paired conditions still absent. Main five-pair public reproduction
is complete and should not be repeated.

# Latest B1data repair / all attribution complete — 2026-11-01 15:12

GOAL ACTIVE/UNMET. Main research goal unchanged. Canonical/new common-harness
H/H0public cohorts are complete;do not repeat their training/download/audits.
Main40540,analysis37956,publisher54528,teacher69732,B1probe68120are TERMINAL
normal completions. No stale PID adoption/termination. New actual native job below.

Five teacher-only controls COMPLETE:117/120versus trainedH120/120,zero unsafe,
PPOmean+2.50pp,95%t[-2.1274,+7.1274]pp,p=.5. Strong B3still24/24.
42109new teacher evaluation controls,zero PPO/BCupdates. Source09ca078independent
reconstruction matches attribution report BYTE-FOR-BYTE SHA
5dfac9890683e58818eba6e0df3a0b1a60c0498356328ed1913fef6a6f32939f.
Report `native_broader_PPO_teacher_rule_attribution_20261101.json` and its
`native_broader_teacher_attribution_independent_verification_20261101.json`.
No stable added-PPObenefit is proven. These teacher raw logs are local,not yet
publicly packaged;do not claim public reconstruction of these new teacher controls.

Old99label B1probe COMPLETE NEGATIVE:
`20261101T032412886357Z_native_trained_actor_trajectory_correction`2/4,
bounds cases1/3at152/143controls,total958;replay3/4bounds case1at264controls.
Raw SHA8cf03cb1e647c31d4e992c2131e56e4637eba3702a1576c5557c4b28a7e04e7d.
Report `native_B1_three_episode_training_coverage_probe_20261101.json`.
Old52972worker terminal;do not restart. Extra coverage fit was rejected.

TRAINretention diagnosis:all3496base TRAINrows used;99correction increases
basic group0..3MSEabout2.7–4.0times versus replay. Label rate-projection mismatch
was tested and DISPROVED:all99solved labels have identical requested/projected
steering. Do not repeat that hypothesis. Three further CPUfits rejected by
prelocked gates,zero new native probes:
- `B1_seed22_retention_weight8_fit_20261101`:512new actor Adam,group0/1/3fail.
- `B1_seed22_hard_retention_fit_20261101`:512Adam proposals,all basic groups
  retained via TRAIN-only block backtracking,but query MSEonly14%better,not50%.
- `B1_seed22_persistent_teacher_matched_fit_20261101`:512new actor Adam,reuses
  already-fitted replay;new query fit improves,but every original basic group
  degrades. All three have rejected.jsonand NOcomplete.json/native registration.
Combined1536new auxiliary Adam proposals,0newPPO/controls. No relaxing gates.
Report `native_B1_retention_and_failed_gates_20261101.json`.

Actual teacher-reference context audit now COMPLETE:original teacher
`20261027T080111Z_native_rule`,same env/planning/trajectory/MPCcode as B1runtime.
All1297reference-prefix commands across4basic roads reproduce with EXACT0error.
Fresh context compared at68fixed phases(control_index mod40in0/5),with label
disagreements. This is different supervision,not proof of better driving.
`reports/native_teacher_reference_context_20261101.json` is hash-bound;keep bytes.
Initial missing-episode setup and full-fresh-query300sbudget failure preserved in
`runs/native_teacher_reference_context_audit_20261101`;no controls/fits from audit.
DO NOT rerun full-fresh-query audit. Completed source has fixed sampling.

New persistent-label generation COMPLETE on EXACTsame121previous TRAINstates:
all observations BIT-IDENTICAL;4136counterfactual teacher control queries,55.5s,
zero driving. `B1_seed22_persistent_teacher_queries_20261101`,config exactSHA7cef...
at `20261031T025409165666Z_native_supervised_ppo/H/config.json`. Its fit FAILED
retention as above;do not fit more labels or deploy it.

IMPORTANT source-data finding:base3496TRAINrows contain3467actual teacher rows
(groups0..11,unit weights)and29OLDcounterfactual rows(groups>=1024,weight580).
The CURRENT single new hypothesis is FIXEDPROVENANCEdata cleanup:from the same
actual seed22PPOparent(20480-ish controls),512actor-only Adam using only3467
physically executed teacher TRAINrows,zero new queries. Historical supervision
remains in ancestry;this does not erase it. Norm/critic/PPOoptimizer/counters
preserved. Same four basic TRAINretention ceilings pass,all errors BELOWreplay.
Fit `runs/B1_seed22_executed_teacher_only_fit_20261101` COMPLETE.

ACTUALLY RUNNING native frozen-policy probe:
`runs/20261101T071019575543Z_native_trained_actor_teacher_only_cleaning`.
Model6d1ebeb3f35959c1a2dacd0853e0342b3223bd9012a375ffbb7dc81be9a3cdb6,
normdb467fb35b0f78a989e8e042b0400036cada30cfad36b41fba727a4f432e5df5.
Worker72792,creation1793517057.939,basePython311;guard48260;revalidate exact
worker_process.json/CIMbefore management. At15:11guardhealthy,34C. Single owned
CARLA2100..2102. Existing exec session15759is native lifecycle,not functions.wait.
No PPOduring eval,all4original known basic roads,1800s/3600maxcontrols,no retries.
Fresh batch,not ActivateQueued. Model keeps20kactual PPOlineage plus512new BC;
B1runtime remains direct acceleration/steering,NOteacher/planner/MPCsubstitution.
AFTER normal lifecycle:raw endpoint audit and all4measured-start inference;
read actual evaluation before any healthy-baseline claim. Basic gate remains
>=3/4ANDzero collision/bounds. Even success would be ONEdevelopment seed,
not the qualified full five-seed/full-task baseline.

Current root source changes are diagnostic/TRAINfit/data repair only;native job
uses unchanged parent frozen source. New scripts:
`probe_native_retained_correction.py`, `query_native_persistent_teacher.py`,
`audit_native_teacher_reference_context.py`;generic fitter can reuse identical
prior replay inputs and optionally gate before preparing native probes. Native
preparer supports explicit single-arm profiles and refuses rejected gates.
37root targeted tests PASS;new-code full CI/independent tests pending after push.
Previous completed full CI1036PASS,sourceefe975d. All original research gates,
new final120,healthyB1,stablefeedback and PPOvalue remain UNMET.

# Latest complete matched cohort / public reproduction — 2026-11-01 14:15

PROGRESS:all five newly trained common-harness pairs COMPLETE and publicly
reconstructed. H120/120,H0119/120,zero unsafe;only seed44H0case15deadline.
Mean+0.8333pp,95%t[-1.4804,+3.1470]pp,exact sign-flip p1.0. Still24unique
known conditions,NOT120new final conditions. No stable feedback benefit.
Training controlsH103197/H0102711,actual Adam3270/3250;H44and55each660,
others650. All-outcome interventions2020/2730dominated by one H0timeout;
Hslightly slower on mutually successful cases in EVERY seed. Whole-project
costs/prior failures remain separate. Cost script and regression9tests added.
Root and clean independent checkout30targeted PASS,cost output byte-identical
SHA15a2a07ea1db8fac4dce23f9806ba3a76a9395e870818728438cc15f96b6fafd.

Publication54528 TERMINAL SUCCESS;analysis37956 and main40540 TERMINAL.
Do not restart them. Full public proof verifies5capsules/4652files/287811controls/
240starts/5case figures,all cohort fields,10actual Adam/frozen normalizers,
5distinct trained policies per method,and exact main figure SHA
207b650e4ddaf29269fa16e2578c95eee961ac76627c21c3fa6ba5202d322982.
Recipe `protocols/public_native_matched_broader_feedback_cohort_20261101.json`;
proof `reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json`.
Current source includes cost and metadata export fixes. Original public capsule
packager369db69/full-cohort reconstruction2b8e6ce/frozen publishera8b491eremain
accurately recorded. Prerelease tagv0.2.0-dev.20261101-feedback stays369db69.
No repeated download or model/raw replay is needed.

Found REAL Git/export line-ending binding bugs. Commit9ba95f5preserves4hash-bound
reports. Sourceefe975dd22304e20830165308d47c480eaa2bfcaadds exact-byte recipe
export (LF/CRLF,wrong digest,existing-file guards) and preserves paired manifest/
plot metadata/public recipe bytes. Root44targetedPASS;hosted1036PASS/18skip/
4CARLAdeselect/89.13s,run41279842561. Prior cost code3154e53hosted1031PASS.
Current frozen publisher was NOT modified. Its public recipe was already correct
SHA940b062fa628ef4d0b0f86061cbe899018a1f552922a132160d3b6c5cfea0a2d,
but root export reencoded it as7d5b5f34... . After normal completion,ORIGINAL
root recipe and receipt were archived under
`runs/native_matched_broader_feedback_costs_execution_20261101/pre_exact_recipe_export`;
new tested helper copied EXACT already-published recipe and left receipt bytes
UNCHANGED. Original complete.json receipt binding still holds. No reupload.
Report `native_matched_broader_exact_metadata_export_verification_20261101.json`.
Do not regenerate/normalize hash-bound files. Only remaining Git-reference rebuild
should reuse the already-public restored inputs and verify current Git refs;
do not repeat all raw/model/network reconstruction for the same concern.

Teacher69732 is LIVE,completed11/22/33,now44
`20261101T020245721142Z_native_pristine_H_baseline` asof14:14.
Read exact current worker_process.json/CIM before management;old66144terminal.
First teacher11scores23/24versus its trainedH24/24,1100interventions;
no five-seed PPOattribution yet. All5teacher controls must finish,then
B1probe68120automatically evaluates the queued99label correction,not the exact
already-completed replay policy. No new B1fits before that actual native result.
Final outputs: `reports/native_broader_PPO_teacher_rule_attribution_20261101.json`
and `reports/native_B1_three_episode_training_coverage_probe_20261101.json`.
No new final120conditions activated. Healthy B1/PPOvalue/stable feedback/final
confirmation remain UNMET. Goal ACTIVE;five public pairs do not complete it.

# Latest fourth complete public pair / timeout reconstruction — 2026-11-01 13:37

Seed44 `20261101T011953260380Z_native_matched_broader_feedback` COMPLETE:
H24/24,H023/24;H0case15(static_vehicle)deadline,no collision/bounds.
H20766controls/660actual Adam,H020540/650;do not round H's one-rollout
budget overshoot down to650. Raw total58237. Old worker68520 TERMINAL.

Public seed44 capsule COMPLETE:930files/58237controls/48measured starts,
exact figureSHA62c9f57b70d2a6f3c878112f83caa9aa7638f59b8e808ceb1af035b3d7cd41a0,
archiveSHA0465fd8620e20b5c3212d15ba143d857d2326eeeb6e8ed6e6fc84d5ff131eb23.
Publisher54528 handled it automatically with zero network failures.
First4 public totals3715files/230332controls/192starts;still24unique known
conditions. Descriptive H96/96,H095/96;no five-seed interval yet.

Existing full-decision reliance auditor ran ONCE per method against exact frozen
seed44 source and all measured states,zero driving/fitting. H16/841greedy actions
change when learned feedback is hidden;H0negative control0/881and zeroTV.
Both complete in~35s. Reports `native_matched_broader_seed44_H_feedback_reliance_20261101.json`
and `native_matched_broader_seed44_H0_feedback_reliance_20261101.json`.
Descriptive timeout report `native_matched_broader_seed44_failure_diagnostics_20261101.json`:
H0reference first12decisions,planner feasibility rejection at decision12,then
78consecutive plan_failed decisions/780braking interventions until45s deadline.
Planner rejection is NOT a physical collision. H succeeds331controls/16.55s.
Different trained weights/states and initial actions prevent causal attribution;
recorded-state feedback sensitivity is NOT closed-loop performance benefit.

Main observer40540 NOW seed55 `20261101T011957251615Z_native_matched_broader_feedback`,
actual worker66144,creation1793510816.265,basePython311;last11032Hcontrols at13:34.
Revalidate actual handle/CIM before management. Analysis37956 waits exact40540;
teacher69732 follows then B1probe68120. Publisher54528 waits seed55 and final
joint public reconstruction. Do not duplicate packages or touch frozen workers/
worktrees/expected references. No new B1fits before queued99label native result.
GoalACTIVE/UNMET;healthy B1/PPOvalue/stable feedback/final120remain missing.

# Previous third complete public pair — 2026-11-01 13:03

Previous turn PROGRESS:four factual delivery docs updated and137local links verified,
commit8af80bf1f24fab93e1f4484724d79bf5a6a92be7. Their current-result sections are
explicit12:30snapshots;update final ledgers after all five,not every heartbeat.
This turn VERIFIED WAIT then PROGRESS:waited exact worker45988to normal exit0,
owned-empty cleanup,all raw/inference checks and automatic public reconstruction.

Seed33 `20261101T011949266084Z_native_matched_broader_feedback` COMPLETE:
H24/24,H024/24,zero unsafe;20544/20583training controls,650Adam each,
57374total raw controls. Worker45988 TERMINAL;do not restart/adopt/kill that PID.
Source-only inspection found same-target witness caching already present and
batch_sat_v1already active;no profiling experiment or new optimization adopted.

Publisher54528 automatically packaged/uploaded/publicly downloaded/rebuilt33.
918files/57374controls/48start actions/exact figure,zero transfer failures.
Receipt `reports/native_matched_broader_seed33_public_verification_20261101.json`.
First3public pairs total2785files/172095controls/144starts/3exact figures,
still24unique known conditions. H/H0all3tie24/24;no five-seed interval yet.
Incremental release notes updated to describe fixed protocol/per-pair receipts
without stale claims about how many pairs are still running;tag stays369db69.
At all-five completion update notes with actual complete metrics.

Main observer40540advanced to seed44
`20261101T011953260380Z_native_matched_broader_feedback`;actual worker68520,
creation1793508925.433,basePython311,last6818Hcontrols at12:58. Read actual
worker_process.json/CIM before management. Seed55remains queued. Existing
teacher69732/B1probe68120/analysis37956still wait exact dependencies;publisher54528
waits seed44. All frozen training inputs unchanged. No new B1edits/tests/rebuilds
before queued99label driving result or a real failure requiring action.
GoalACTIVE/UNMET. Healthy B1/PPOvalue/feedback/final120still missing.

# Previous actual two-pair results / autonomous publication — 2026-11-01 12:25

Previous turn PROGRESS:first pair actually publicly reconstructed. This turn
PROGRESS:seed22native pair COMPLETE/raw+inference verified,both24/24zero unsafe,
new training20579/20539controls,650Adam each;worker50292 TERMINAL. Observer40540
advanced to seed33 `20261101T011949266084Z_native_matched_broader_feedback`,
worker45988(created1793506553.840,basePython311),currently Htraining.
Two completed pairs do NOT show a success-rate feedback benefit;no five-seed CI yet.

New source2b8e6ce2c357ba342d2b03a6a5537806e723381b adds guarded incremental
all-five publication +public reconstruction support for FIVE complete paired
capsules/24conditions. Existing canonical SIXcapsule/12condition path retained.
Separate --reference-root supports immutable future all-five reference snapshots.
Initial publisher32964 FAILED before any new package/upload/CARLAcall:Windows
GBKreader thread could not decode GHUTF8checkmark,then stdoutNonecausedTypeError.
Old execution `native_matched_broader_full_publication_execution_20261101` terminal,
original source/driver/failure receipts retained. Do not restart it.

FIXED publisher sourcea8b491e29b7d66fb4cb5479379babcd696c876cb captures raw stdout/
stderr bytes,sets child UTF8,and preserves raw logs before decoding. Root and
independent44targetedPASS;hosted1022PASS/18skip/4CARLAdeselect,job41272197243,
197.57s. Actual pending-CIcheckmark log then passed through FIXED pipeline.

NEW ACTUAL publisher54528(created1793506441.616,basePython311),execution
`runs/native_matched_broader_full_publication_execution_20261101_v2_utf8`,
already reused publicseed11,PACKAGED/UPLOADED/PUBLICLYDOWNLOADED/REBUILT seed22,
and is now WAITING exact40540forseed33. No native calls/training source changes.
Launch.json has exact identity. Artifact directory
`runs/prerelease_20261022/native_matched_broader_full_publication_20261101_v2_utf8`.
All models/normalizers/controlled logs unchanged;remote objects checked before
upload,no overwrite/delete,manifestLAST,1MiBparts,2workers,max3transferattempts,
12hglobal deadline. It handles remaining33/44/55automatically;do not package or
publish these separately. At the end it waits analysis37956,uses copied immutable
references,checks full public cohort/10distinct policies/actual Adam/frozen
normalizers/main figure,and writes recipe+verification report+release metadata.
First pair is not reuploaded or separately re-downloaded;the final joint cohort
validation necessarily reconstructs all five from exact prior PUBLIC download
caches. Current program's frozen source/plan must remain untouched.

DO NOT CHECKOUT/EDIT the two dedicated publisher worktrees:
`.../prerelease_20261022/matched_broader_packaging_checkout_20261101` pinned369db69,
`.../prerelease_20261022/matched_broader_public_reconstruction_checkout_20261101`
pinned2b8e6ce. Basic-course independent checkout is a8b491eand IDLE for later tests.
Do not edit automatic all-five reference report/figure;extra cost analyses use
separate outputs. After full5public success,scan+commit generated recipe/reports
and update prerelease notes with actual complete metrics (notes initially describe
firstpair). Public completion is NOT overall research acceptance.

Seed22archivePUBLIC COMPLETE SHA
23c15a1a5360040fcc51dd8f57759e71c5cbd2b13100064afac166b326709f73,
931files/57309controls/48start actions/exact figureSHA
c4f173e3974f6721f9846e6254cef538d8061ea6054420eaa906b2d1825ebe33.
Receipt `reports/native_matched_broader_seed22_public_verification_20261101.json`.
Together first two:1867files/114721controls/96starts,24unique known conditions.
Activation/failure provenance reports `_activation_20261101.json`and
`_activation_20261101_v2_utf8.json`(full prefixnative_matched_broader_full_publication).
Teacher69732/B1probe68120/analysis37956still wait existing exact dependencies.
GoalACTIVE/UNMET;no new B1edits before queued99label closed-loop outcome.
Do not repeat failed NullRHIprobe or completed tests/reconstructions.

# Previous complete public pair / owned resource probe — 2026-11-01 11:58

Previous turn PROGRESS:first pair completed and automatic all-five analysis
prepared;source369db6939ee67e540aa6b63c155b3eb27964f2bc hosted1007PASS,independent27PASS.
This turn PROGRESS:first COMPLETE paired model/raw archive actually packaged,
uploaded,publicly downloaded and independently reconstructed. No new method edits.

PUBLIC prerelease v0.2.0-dev.20261101-feedback targets369db6939ee67e540aa6b63c155b3eb27964f2bc.
Archive `native_matched_broader_H_H0_seed11_20261101_repro.zip`,141286566bytes,
936files,SHA536125ada8d3bf8b4a55d78505d9f53c3a257f79195646c379d4a94e2765e92f.
135x1MiBparts+manifestLAST;all remote size/digest/state checks passed,zero failures.
Unauthenticated public download/independentcheckout369 reconstruction verifies
57412controls,48start actions,exact all48case figure SHA
dd4854412258bbf0f74483e1413557149b1f411cdc79f9f1161e1a11b1fc4b1f.
Public receipt `reports/native_matched_broader_seed11_public_verification_20261101.json`.
Execution `runs/native_matched_broader_seed11_publication_execution_20261101` TERMINAL
SUCCESS;publisher34144 TERMINAL. Do not restart/reupload/rebuild this capsule.
Artifact directory `runs/prerelease_20261022/native_matched_broader_seed11_publication_20261101`.
Separate local preflight report `native_matched_broader_seed11_capsule_independent_20261101.json`.
No new training/evaluation controls from packaging or reproduction.

One isolated NullRHIempty-world startup feasibility attempt FAILED before RPC
readiness on2300..2302;owned shipping42380 exited3after19.469s,total0controls,
0actor/camera commands. Its primary-preservation proof shows main71516live;
secondary has exited,no remaining children found. No renderer/launcher/source
changes adopted,no retry. This is not evidence of physics equivalence or a
general unsupported-platform claim. SourceCLI/failed readiness/serverlogs stay
in `runs/isolated_nullrhi_readiness_probe_20261101`;sanitized report
`reports/native_isolated_nullrhi_startup_probe_20261101.json`.
Keep existing single-server native queue;do not repeat this resource experiment.

Broader observer40540LIVE,seed22batch`20261101T011945179519Z_native_matched_broader_feedback`,
worker50292(created1793504098.534,basePython311). Hcompleted20579controls/650Adam,
worker reports24/24validation (rawverification awaitsH0andcleanup). H0has
ACTUALLY started,1355controls at11:56. Do not rerun completed H.
Teacher69732/B1probe68120/analysis37956remain waiting exact existing dependencies.
Read launch.json/CIM before any management. GoalACTIVE/UNMET;healthyB1/PPO-vs-rule
and feedback contributions/final120remain missing. No new B1edits before queued
99label closed-loop outcome;no extra CPUprobes or repeated completed checks.

# Previous verified pair / automatic analysis — 2026-11-01 11:38

First matched broader pair COMPLETE and raw/inference verified. Seed11H24/24,
H024/24,all three families8/8each,zero unsafe. Actual new training20522/20529
controls,650Adam each,total57412raw controls. H470interventions vs H0390;
all24mutually successful,412.500006s vs405.550006s. No feedback success/time/
intervention advantage for this one seed. No five-seed inference from one pair.
Reports `native_matched_broader_feedback_seed11_endpoint_20261101.json`,
`native_matched_broader_feedback_seed11_pair_20261101.json`,
`native_matched_broader_feedback_seed11_costs_20261101.json`.

Observer40540 automatically advanced to seed22batch
`20261101T011945179519Z_native_matched_broader_feedback`;worker identity must be
read from its worker_process.json/CIM,not old68256(terminal firstpair worker).
Teacher69732 and B1probe68120 remain waiting;their frozen inputs unchanged.

NEW ACTUAL read-only analysis observer37956,execution
`runs/native_matched_broader_feedback_analysis_execution_20261101`, waits exact
40540beforeall5aggregate/figure. launch.json has exact creation/executable.
No CARLAcalls,0training,12hwait/1200sanalysis,no retries. All5/all24/commonFULL
source trees required. Protocol
`protocols/native_matched_broader_feedback_pairs_20261101.json`.
Fixes old plot's hardcoded12condition/canonical-cohort text ONLY for new
explicit matched-broader kind;neutral exploratory title/dynamic effect axes.
Old published figure rebuilt EXACT SHA
cafca085ed2550173858f647a6de72c4dccbcedfa6ef03497325720c8c94eb87.
27related tests PASS. New cohort result/figure are PENDING,not available yet.
Activation report `native_matched_broader_feedback_analysis_activation_20261101.json`.
Do not restart completed firstpair or duplicate either analysis observer.
No further B1edits before the already queued99label actual driving result.

# Previous continuation — 2026-11-01 11:27

PUBLIC RGB RELEASE COMPLETE, source ba932787113168ad5781ee640c6f06d09bbee0e1,
tag v0.2.0-dev.20261101. Source hosted CI passed. Both capsules downloaded without
authentication and independently reconstructed:1020files/1828controls/4start
actions/2byte-identical figures;all1832video frames decoded. Four direct MP4links
also downloaded and hash-verified.82assets,zero network failures. Receipt
`reports/native_PPO_RGB_public_verification_20261101.json`; reproduction commands
`docs/native_PPO_RGB_reproduction.md`. No further upload/rebuild required.

New B1development hypothesis implemented and ACTUALLY FIT: own first3completed
TRAIN episodes per condition, fixed stride40plus final pre-control,all outcomes;
121queries/99solved references/22wait excluded,12episodes,30.875s query wall.
Fit `runs/B1_seed22_three_episode_matched_trained_fit_20261101`:512steps each,
1024additional actor Adam,41.796s,zero new PPO/driving. TRAIN MSE decreases;
no driving benefit claimed. Do not conflate this with earlier17label negative pair.
New prepared correction batch `20261101T032412886357Z_native_trained_actor_trajectory_correction`.
New replay `20261101T032412251159Z_native_trained_actor_replay` is UNSTARTED:
its full policy/retained PPO optimizer tensors/counters and normalizer exactly
match already evaluated old replay `20261101T014911209543Z_native_trained_actor_replay`.
Reuse old3/4+1bounds result; do not repeat replay driving. ZIP hashes differ due
container metadata; exact tensor/normalizer proof is retained.

ACTUAL observer68120(created epoch1793503546.106,base Python311), execution
`runs/B1_three_episode_corrective_probe_execution_20261101`, waits exact teacher
observer69732. Its launch.json has exact process identity; revalidate before any
management. It checks all five teacher controls complete/cleanup,then drives
ONLY new corrective model on all4known cases,raw/inference checks and unchanged
zero-unsafe gate. Max3600new evalcontrols/1800s,0PPO,no retries. Auto report
`native_B1_three_episode_training_coverage_probe_20261101.json` is NOT available yet.
Activation evidence `reports/native_B1_three_episode_training_coverage_activation_20261101.json`.
Driver source is frozen/hash-bound; current running native inputs untouched.

Broader observer40540 remains LIVE. First seed11H finished20522controls,
650actual Adam,24/24known validation success. H0 is actually training (last
10828controls); remaining four pairs pending. Do not rerun first H or claim the
whole five-seed cohort complete. Teacher observer69732 waits exact40540.
GoalACTIVE/UNMET;healthy B1/strong-rule PPOvalue/feedback/final120still missing.

# Historical native recordings / corrective results — 2026-11-01 11:10

All5lower-rate B1extensions COMPLETE:3/2/2/0/2of4,9/20success,10bounds,1timeout,
only seed11qualified. 51200new training controls/1595actual Adam steps,58991raw
controls. Seed55ends2/4,315new Adam,11493rawcontrols. Reports
`native_B1_all_five_lower_rate_extensions_20261101.json`and55endpoint retained.
The full-task driver20420correctly rejected every-child entry;no full B1starts.

Camera queue42060FAILED before policy controls:subset manifest kept4original
routes but selected only0/3,violating the unchanged all-roads enumeration guard.
Old Hbatch`20261101T004032948606Z_native_PPO_camera`,stderr/failure/owned-empty
server cleanup preserved. Dependent30268/73064observers failed closed before
native children. Do not restart any of those old executions or old unstarted
B1camera inputs. Fix only prunes unused routes/remaps slot3->1,then checks actual
decoded physical-case equality with original0/3;no global guard or policy change.
48targeted/70broader camera+asset tests pass;actual frozen enumeration/restore of
BOTH corrected batches passes before native launch. Completed training receipt
hashes allow a new observer after original dependency process exits;strict
all-five gate/cleanup checks remain.

NEW camera execution`native_PPO_camera_execution_20261101_v2_subset_fix`COMPLETE.
H`20261101T024611835234Z_native_PPO_camera`:2/2success,645controls/647RGBframes.
B1`20261101T024615807894Z_native_PPO_camera`:1success/1bounds,1183controls/1185frames.
Actual4videos/1832RGBframes/1828controls,zero new PPO/Adam. Original parent cases
0/3,seed11,Hretains650Adam/B1retains9. All encode/decode/fps/hash/world-frame
fences checked;both contact sheets viewed (12samples total). Case3palm canopy
partly occludes ego;do not claim unobstructed or whole-video human inspection.
Unique IDs/model/norm/seed/parent-case bindings:
`reports/native_PPO_video_catalog_20261101.json`;full startup-fix/real-result ledger
`reports/native_PPO_actual_recordings_and_subset_fix_20261101.json`.
Camera source/meta/models are frozen;graphs in eachbatch/analysis/frozen_endpoints.png.
Optional packager --camera-media includes models/raw controls/videos/frame
manifests but explicitly omits bulky original RGBPNGs (kept local).

Matched actor probe COMPLETE:replay512and17label-correction512both3/4with one
bounds. 1275/1153audited controls;no new PPO. Original17labels do NOT improve
success/safety over replay;neither branch qualifies. Do not reinterpret the98%
TRAINfit gain as driving improvement.
`reports/native_B1_matched_trained_actor_correction_20261101.json`.
All-five early TRAINoracle queries now COMPLETE:112points/110solved references,
two wait rows;zero new controls or optimizer updates,no fit. Fifth firstTRAIN
case3is short so only20queries. All five data sources cover only4known conditions.
`reports/native_B1_all_five_early_TRAIN_oracle_coverage_20261101.json`.

ACTUAL ongoing broader observer40540,execution
`H_H0_matched_broader_with_corrective_probe_execution_20261101_v2_camera_subset_fix`,
has finished both probes and STARTED seed11H/H0pair
`20261101T011938826336Z_native_matched_broader_feedback`(Hworker68256lastseen).
The same5prepared training inputs remain unchanged;no new initialization/fits.
Teacher-only observer69732,execution
`H_pristine_baseline_execution_20261101_v2_camera_subset_fix`,waits exact40540.
Their launch.json records contain exact creation epochs/executable identities;
revalidate before managing. Old failed dependencies and supersession receipts
remain. GoalACTIVE/UNMET;healthy B1/strong-rule PPOvalue/feedback/final120missing.
At this historical timestamp public media verification was pending;now COMPLETE above.

# Historical recording-index preparation — 2026-11-01 10:37

Previous goal turn was PROGRESS:new teacher-only control queue,980full hosted
passes,seed44failure/raw-bound figure and90early TRAINoracle references.
Fifth B1worker45716(created10:21:29.204364+08:00)remains live in
`20261101T022104544331Z_native_B1_basic_lr_extension`;last10:36:30heartbeat8139
controls/6phases. Existing observers and every frozen input remain unchanged.

New helper`scripts/catalog_native_ppo_videos.py`addresses an actual metadata
ambiguity:the frozen renderer uses method folder names H/B1as legacy run_id.
After the ENTIRE camera42060queue completes,run:
`python scripts/catalog_native_ppo_videos.py --execution runs/native_PPO_camera_execution_20261101 --output runs/native_PPO_video_identity_catalog_20261101.json`.
It verifies all planned batches/videos,raw/model/normalizer/episode bindings,
actual contiguous RGBframe hashes/times,complete decode/fps receipts,then adds
batch-qualified video IDs,original parent-case IDs and unique release names.
Original videos/manifests remain unchanged. It rejects pending/partial queues;
no catalog or new camera footage is claimed yet. Actual contact-sheet visual
inspection and publication are still separate required actions.22related tests
pass3.67s. Do not modify active camera or broader observers just to adopt this
postprocessing helper. Goal remains ACTIVE/UNMET.

# Historical native checkpoint — 2026-11-01 10:30

Seed44TERMINAL:0/4success,four bounds,10240new training controls/320actual Adam,
20480retained policylineage,11813rawcontrols (1573evaluation). BasicgateFALSE.
Report`native_B1_basic_lr_extension_seed44_endpoint_20261101.json`. All four
trajectories are plotted in`reports/figures/native_B1_seed44_lower_rate_failures_20261101.png`
and visually inspected. Independent checkout3d4e992rebuilds byte-identical PNG,
SHA b0ee44bb132d5ac07875974f94a02aa2d0152ea4cb62687538718a032e897d1f.
All1573measured evaluation controls show no goal-position entry;nearest goal
distances2.4412,22.4621,1.0013,16.9503m. Case3fails at3.65s. Ten exact measured
decision reconstructions retain identical observation statistics;comparison is
with the extension's already trained parent,not a zero-PPO counterfactual run.
`native_B1_seed44_lower_rate_failure_diagnosis_20261101.json`binds all evidence.
These failures cannot be explained solely by terminal braking.

Actual fifth/last lower-rate learner:batch
`20261101T022104544331Z_native_B1_basic_lr_extension`,worker45716,
created2026-11-01T10:21:29.204364+08:00;guard22424is distinct. Last heartbeat
10:30:37shows4250controls/3phases. Driver62332and failed-all-five full-gate
observer20420unchanged. Revalidate exact creation identities before management.

New bounded offline EARLY TRAINoracle queries COMPLETE for11/22/33/44:
`runs/B1_lower_rate_seedSEED_early_training_oracle_20261101`.
Existing --coverage profile:first completed TRAINepisode/condition,regardless
outcome;fixed controls0/40/80/120/160/200 when present. 24/24/24/20queries,
92total,90solved-reference MPCcommands,2wait-only rows (seed44condition1at
160/200). 55.937s total query wall,zero new controls/PPO/supervised updates.
Twelve seed22anchors overlap the earlier full-trajectory queries:observations
and labels reproduce exactly. Repeat queries are explicitly counted. All source
and measured-history hashes retained;no model fits or changes to the original
queued17label/512step pair. These are candidate labels,not executed commands or
proof of recovery;bounded oracle search failure is not geometric impossibility.
`reports/native_B1_early_TRAIN_oracle_coverage_20261101.json`binds manifests.
Seed55must finish and be raw-audited before any TRAINquery;no fifth-data claim.

Other verified-live observers unchanged:camera42060 -> corrective probes and
broader H/H0observer30268 -> pristine teacher-only observer73064. Rootcode
3d4e992has980hosted passes/18skips/4CARLAexclusions,127.73s,independent34passes.
No new final conditions. Goal ACTIVE/UNMET. Wait for registered native results;
do not treat90offline labels,greenCI or new published figures as repaired B1.

# Historical attribution-control preparation — 2026-11-01 10:08

Previous goal turn was PROGRESS:matched B1actor fits and independent exact replay
completed,new native pair queued,code pushed3c2893250195faf7f37bd33fa28c9eb8f84da596.
This turn preserves all existing queues and adds a necessary teacher-only H
control to distinguish added PPO from the common supervised prior.

ACTUAL new WAITING observer73064(created2026-11-01T10:07:08.171144+08:00;
launch shim16492),execution`runs/H_pristine_baseline_execution_20261101`.
It holds the exact30268process handle and waits for ALL5matched broader H/H0
pairs and owned cleanup. No native launches or new optimization yet. Finite12h
wait,1800s per baseline,no retries. The earlier B1/camera/corrective/broader
workers,observers and frozen training inputs are unchanged.

Preparation`runs/H_pristine_baseline_preparation_20261101`contains5distinct
zero-PPO Hactors copied from the respective broader cohorts,896inherited
teacher-supervised updates,zero added fitting:
11`20261101T020242512174Z_native_pristine_H_baseline`,
22`20261101T020243559283Z_native_pristine_H_baseline`,
33`20261101T020244732149Z_native_pristine_H_baseline`,
44`20261101T020245721142Z_native_pristine_H_baseline`,
55`20261101T020246624428Z_native_pristine_H_baseline`.
All5actual frozen-worker CPUrestores and registered launcher validations pass;
zero counters/emptyAdam,exact initial actor hashes/normalization,one actual
teacher TRAIN-observation inference. CPUprobe v1used a wrong archive key and
failed before any driving;preserved source and corrected H_observations/TRAINmask
probe recorded. No native retry or policy result is hidden.

The completed strong B3`20261031T082841Z_native_rule`has the SAME resolved task,
36TRAIN/24known-VAL inventory/calibration and46active navstack/bridge modules.
Only6inactive diagnostic modules differ. All8033measured controls remain raw
bound;reuse this fixed deterministic reference rather than retrain/replay it.
Post-cohort report will compare ALL5trained H policies with their own pristine
controls and separately B3,with paired-seed tintervals/sign-flip descriptions.
Secondary times use explicit mutually successful case pairs;all-outcome cost
also retained. This is exploratory development,not RL-necessity proof or final
confirmation. 24cases×5seeds is not120unique conditions. Extra evaluation cost
is separate;no additional independent PPOtraining is claimed for teacher-only H.

New sources:`prepare_native_pristine_baselines.py`,`run_native_pristine_baselines.py`,
`analyze_native_pristine_baselines.py`;34related root tests pass4.90s and34
independent-checkout tests6.28s at3d4e992f91fb5a0f613120dbee4433ec1d2c78a5.
Full hosted source3d4e992passes980/18skips/4CARLAexclusions in127.73s:
https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41264048875.
Protocol,all
artifact identities/CPUpreflights/live queue:
`reports/native_PPO_teacher_only_attribution_queue_20261101.json`.
After all endpoints finish,standalone rebuild:
`python scripts/analyze_native_pristine_baselines.py --execution runs/H_pristine_baseline_execution_20261101 --output <fresh.json>`.

Actual prior-code hosted regression e2bc98aa4bff1158cb22af4eb09b5eba0e822b3a:
971passed,18skipped,4CARLAdeselected,197.12s,
https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41262967134.
Fourth B1extension remains actual worker32708(created09:55:07.805682+08:00),
batch`20261101T015454924014Z_native_B1_basic_lr_extension`;last checked10:13:09,
8264new controls/6phases. Seed55still future. GoalACTIVE/UNMET;do not fill idle
time by rerunning completed releases or fitting more architectures before the
registered closed-loop correction and broader attribution results arrive.

# Historical continuation — 2026-11-01 09:56

Seed33TERMINAL:10240new real PPOcontrols/320Adam updates,2/4success,two bounds,
basicgateFALSE,11861rawcontrols,20480lineage. Full raw/inference audits complete;
`reports/native_B1_basic_lr_extension_seed33_endpoint_20261101.json`retains it.
Failures0/3,successes1/2. Actual seed44now runs in
`20261101T015454924014Z_native_B1_basic_lr_extension`,worker32708(firstheartbeat
09:56:11,141controls). Revalidate creation identities before management.
Driver62332and full-gate observer20420remain unchanged;no failed seed omitted.

NEW retained-state actor correction diagnostic actually fit both arms:seed22
endpoint,512separate-Adam actor-only updates each,critic/retainedPPOAdam/counters/
normalizer exact. No new PPO or driving. Same3496teacher TRAINrows/weights;
correction arm adds17solved-reference labels from25outcome-independent first
TRAINepisode queries;all8wait/brake-only labels excluded. Acceleration/steering
TRAIN MSE .0308003/.00237298 becomes .0310828/.00242406with replay alone,
.000594401/.0000510366with added on-policy labels. This is not driving evidence.
Executed fit/data/oracle snapshots remain in
`runs/B1_seed22_matched_trained_trajectory_fit_20261101` (47.828s,total1024extra
supervised steps). Zero-PPO fitting guard remains intact;new trained-state API
is explicit and requires a fresh rollout before any subsequent PPO training.

Both frozen four-case probes are PREPARED,not driven:
replay `20261101T014911209543Z_native_trained_actor_replay`,
correction `20261101T014911842823Z_native_trained_actor_trajectory_correction`.
Both actual launcher input preflights passed;18related tests pass6.69s.
Both actual frozen-worker CPUrestores retain20480controls/PPO_n_updates92 and
reproduce all25saved TRAIN-state predictions with maximumerror0.0.
Independent checkout e2bc98aa4bff1158cb22af4eb09b5eba0e822b3apasses29related
tests9.10s and COMPLETES both bounded auxiliary fits in52.406s. Every input
array,loss history,all25predictions,full policy and retained optimizer tensor
matches exactly. The1024extra reproduction actor updates are separately charged;
this is not new driving or independently trained policies. Its verification is
in`runs/prerelease_20261022/basic_course_code_checkout_20261101/runs/B1_seed22_matched_trained_trajectory_independent_fit_20261101/independent_verification.json`.
Record `reports/native_B1_trained_trajectory_correction_preflight_20261101.json`.

Old broader observer72028was verified live/childless/waiting then explicitly
superseded via its creation-bound handle;receipt retained in its old directory.
No native worker or prepared H/H0training inputs changed. NEW observer30268
(actual base Python,created2026-11-01T09:50:53.932895+08:00;launch shim55872)
is actually WAITING on exact camera42060handle. New execution:
`runs/H_H0_matched_broader_with_corrective_probe_execution_20261101`.
Order:existing all-five B1extensions/full-gate resolution -> existing camera
and cleanup -> both4case corrected-actor probes with raw/inference audits ->
the SAME already prepared all-five36TRAIN/24known-VAL H/H0cohort. No retries,
no final conditions,no baseline qualification claim from offline loss.
Do not restart the superseded observer or launch duplicate native batches.

# Historical native endpoint — 2026-11-01 09:36

Seed22lower-rate extension TERMINAL:2/4success,two bounds,320new Adam updates,
10240new controls,20480policylineage,11557audited raw controls. GateFAIL.
Learning rate/update count alone is NOT a universal B1repair. Do not skip this
seed or launch full-task training from the one passing seed11. Complete the
remaining prespecified33/44/55courses;observer20420will reject full entry unless
every gate is true. Current actual child:
`20261101T013448773461Z_native_B1_basic_lr_extension`,seed33,worker38692,
driver62332. Revalidate all PIDcreation identities before process management.
Report`native_B1_basic_lr_extension_seed22_endpoint_20261101.json`is retained.

# Historical research checkpoint — 2026-11-01 09:34

Previous goal turn was PROGRESS:seed11completed its10240-control lower-rate
extension with320new Adam steps;3/4success,1deadline,zero unsafe,BASICGATEPASS.
Lineage20480controls;not full-task health. The failure never enters.8mgoal
tolerance,endsstationary2.1643mfromgoalat45s;do not hide this stall or claim
overall driving success. Reports`native_B1_basic_lr_extension_seed11_endpoint_20261101.json`
and`native_B1_first_lower_rate_basic_gate_and_stall_20261101.json`bind evidence.

ACTUALLY TRAINING:62332observer,seed22child
`20261101T011350364179Z_native_B1_basic_lr_extension`,worker41112.
Seed11child53620is terminal;revalidate creation identities before management.
All5retain the same lower-rate recipe;full observer20420still waits every gate.
Camera observer42060still waits for training/full-stage exits. No RGBframes yet.

NEW ACTUALLY WAITING:72028observer,execution
`runs/H_H0_matched_broader_execution_20261101`,after exact owned42060finishes.
Preparation`runs/H_H0_matched_broader_preparation_20261101`has ALL5pristine H/H0pairs:
11`20261101T011938826336Z_native_matched_broader_feedback`,
22`20261101T011945179519Z_native_matched_broader_feedback`,
33`20261101T011949266084Z_native_matched_broader_feedback`,
44`20261101T011953260380Z_native_matched_broader_feedback`,
55`20261101T011957251615Z_native_matched_broader_feedback`.
36TRAIN/24KNOWNdevelopment-validation,20480requested controls per method/seed,
204800total requested. No new BCfit,no final conditions,no actual broad training
yet. All5complete517-file source trees identical,both methods share current
bridge852e6e...before first PPOupdate. All10models actual frozen-runtime restored
with zeroPPO/emptyAdam,exact actor tensors/statistics and existing12teacher-case
prefix;5launcher input validations pass. Prior canonical12and mixed-harness
single-seed broader costs stay charged/historical,not pooled as this new cohort.
No pipeline observer may skip failed gates or silently retry a native job.

EXECUTION PREDICTOR REJECTED,not deployed. Pilot4160TRAINdecisions/707planning
failures gave27.3percent Brier improvement. Stricter20,800decision validation
uses205446already-audited TRAINcontrols from all10canonical H/H0policies;
trainseeds11/22/33,validation44/55,each fold withholds its complete route.
Capacity-matched MLP41inputs,32tanh16tanh1,300fixed Adam updates;geometrybaseline
zeros6past-execution inputs. Mean Brier geometry.07199,feedback.05179,
simple past-failure baseline.04632. Prespecified stronger criterion FAILS.
All4independent replay prediction arrays exactly match;33targeted independent
tests8.99s. No new native controls or policy updates. Keep raw datasets/protocols
in`runs/selected_execution_prediction_pilot_20261101`and
`runs/selected_execution_prediction_cross_seed_validation_20261101`;
production utility`scripts/probe_native_execution_prediction.py`is diagnostic only.
Reports`native_execution_prediction_cross_seed_probe_20261101.json`and
`native_execution_prediction_independent_verification_20261101.json`retain failure.

Actual hosted sourceae76d1ddd872519b7c7d0b073d59c97c30e53e81passes968tests,
18skips/4CARLAexclusions,197.52s:
https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41261183283
Independent checkout`basic_course_code_checkout_20261101`is atae76d1dand IDLE
after33tests/all4MLPfold replays. Existing full Windows976pass at8755337remains
valid for that source;do not call it a new full run at the later source.

GoalACTIVE/UNMET. Healthy all-five B1,PPOvalue,strong B3 comparison,feedback
benefit,frequency/shared-safety evidence and120NEWfinal conditions remain.
No new final conditions have been generated. Do not redo completed publications.

# Historical continuation checkpoint — 2026-11-01 08:59

Read this snapshot before the timestamped historical notes below. Revalidate live
process creation identities from local receipts;never manage a process by PID alone.
Goal remains ACTIVE/UNMET;do not mark complete because training/tests/publication finish.

All-five basic PPOcourse62176is TERMINAL:seeds11/22/33/44/55score3/2/2/0/2of4,
all qualificationFALSE. Total51,200new training controls,103actual Adam updates,
58,758audited raw controls. Only4known TRAINING conditions repeated across5policies.
`reports/native_B1_all_five_basic_PPO_course_20261101.json` rebuilds with
`scripts/analyze_native_basic_course_cohort.py`;independent checkout reproduces
every field. Do not rerun these completed courses.

ACTUALLY TRAINING:driver62332,execution
`runs/B1_basic_lower_rate_extension_execution_20261101`,seed11child
`20261101T005612994446Z_native_B1_basic_lr_extension`,worker53620.
Its first1280native controls/40new Adam updates are saved;optimizer counters49
include9retained updates,actual saved rate3e-6,obs mean/variance/count exact.
The entire continuation is not yet raw-audited or qualified. Each of all5seeds
receives the same additional10240controls with retained trained weights/Adam/RNG;
no new BCfit and no gate waiver. Parent interruption loss/unknown tail remains charged.

WAITING:full-task observer20420,execution
`runs/B1_full_after_lower_rate_extension_execution_20261101`. It may create full
children only if ALL5extensions qualify. Then each gets20480additional controls;
eventual lineage40960is not matched to existingH20480. No full child yet.

WAITING:camera observer42060,execution`runs/native_PPO_camera_execution_20261101`,
after62332and20420exit with its strict cleanup/failure barrier. Prepared H/B1
camera batches`20261101T004032948606Z_native_PPO_camera`and
`20261101T004019372592Z_native_PPO_camera`,each original cases0/3. No RGBframes
yet. Frozen CPUrestores pass;illustrations are selected known cases,not final
confirmation or matched-budget evidence. Finish actual frame/decode/visual checks
and a uniquely batch-qualified video catalog before claiming/publishing video success.
Independent public-input staging`20261101T004659118707Z_native_PPO_camera`is
CPUONLY,not registered for native launch. Do not start it as a duplicate.

Full independent Windows CPU976pass/1skip/4CARLAexcluded,193.65s at8755337.
Actual hosted Linux959pass/18skip/4excluded,196.69s:
https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41258680712
Earlier failures are retained:unsupported test action-space stand-in corrected
without weakening auditing;camera edits restored frozen cadence/partition text
anchors. Keep those anchors compatible with already-frozen preparation helpers.
`reports/native_B1_basic_cohort_independent_verification_20261101.json` binds all proof.

Exact corner-cell speed probe had1.019x median improvement and was REJECTED;
no geometry optimization was applied. First3seed landmark inference40decisions
max2.98e-8action error locates missed braking AND nonterminal lane failure;
it is descriptive same-state inference,not causal counterfactual driving.

Existing H/H058/60vs57/60and H/B258/60vs44/60/public reconstruction stay complete;
H/B2bridge initialization differs,so no isolated planner-effect claim. Strong B3
matches H24/24in the broader development validation. Healthy B1,RLvalue,feedback
benefit,final120NEWconditions and final acceptance remain unmet. Do not activate
final tests or count known12/36TRAIN/24VALas final. Do not repeat completed releases.

# Historical public failure evidence — 2026-11-01 08:27

FAILED B1seed11capsulePUBLICCOMPLETE,all96partsandmanifestuploadedwithoutfailed
networkattempts,actualunauthenticateddownloadandindependentreconstructionpassed.
Receipt native_B1_basic_seed11_failed_gate_public_verification_20261101.json;
553files,11966controls,4actions,exactfiguree2d3dcc89e41f054a42d7eceb090078e250067fb583857d6894a40e07b36a6c5.
Archivehashd13dcf3b07fe1574d8823e23f5c29abcc6e43bbdd1d68e1e5e9b506a5c0aa22a.
QualificationFALSE,3/4successand1boundsretained. Publisher916terminalsuccess;
mainnative_road_checkout285cd30mayadvanceafterconfirmingnoliveuser.
Actualtraining62176nowseed44;11/22/33allfailedbasic. Extension62332waitsall5;
conditionalfull20420waitsall5extensionqualifications. No nativeextensionchildyet.
No repeatedH/H0/B2publication or repeatedmatchedreplayneeded. GoalACTIVE/UNMET.

# Authoritative core-learning progress — 2026-11-01 08:21

CORE ROOT-CAUSE EVIDENCE:actual first1280TRAINrollout reconstructed using recorded
UNBOUNDED Gaussian actions,measured states/actuatorhistory,and originalRNG.
Originalrate3e-5produces1Adamstep,checkpointmaxparametererror5.2154e-8(tol1e-7),
maxsampleerror1.192e-7(tol1e-6). Strictbitwiseactionprobe failedat34with2.98e-8;
not hidden. Firstfullprototype failedJSONnpfloat32serialization aftertraining;
retained,not a successful result. Production replayer fixesserialization,saves
recorded_rollout.npz/before_update_rng.pklBEFOREtraining,and uses actualrecorded
raw actions rather than regenerated ones. Alternative3e-6produces40Adamsteps,
meanKL.00374859vs.0139810,valueTRAINMSE.721098vs.844533(before.891875).
ONEseed/ONErollout,not driving-improvement or independent-seed evidence.

SOURCE99479b54ddd1c36af9c4724f058cb22354b91856pushed. Independent183testsPASS12.58sec.
Independent coldZIPreproduction verifies553files/11966controls/4initialactions/
exactfigure. Independent entire1280replay again produces BITIDENTICAL dataset
arrays and all result fields. Receiptnative_B1_matched_rate_replay_independent_verification_20261101.json.
Code scripts/replay_native_first_ppo_update.py; no alternate offline model deployed.
Rootfixture actualtrained-checkpoint copy preflight also proves LRcontinuation
retainsweights/Adam9/normmoments/RNG,0reset/control/update. No extra BCfits.

ACTUAL PIPELINE (revalidatePIDs):62176originalbasiccourses,nowseed44after11/22/33
finish3/4,2/4,2/4 ALLFAILunsafe. All5stillcompleteasregistered. Lower-rate
EXTENSION62332,tool81745,executionB1_basic_lower_rate_extension_execution_20261101
waitsexact62176handle:requiresall5coursesfinishedwithfailedqualification,binds
matchedreplayreporthash,then ALL5trainedpolicies eachget10240MOREnativecontrols at
3e-6withAdam/RNGretained. GateAFTERlearning unchanged>=3/4zero unsafe.
NewconditionalFULL20420,tool19441,executionB1_full_after_lower_rate_extension_execution_20261101
waits62332ALL5qualification;adds20480onoriginal12conditions. Totalpolicylineage
40960,not equal-budgetH20480evidence. Fullfactory now charges retained cumulative
Adamsteps and bothbasicstages. Obsoletechildless66252was exactidentityverified
andintentionallyterminated;supersessionreceiptretained,no nativeworker stopped.
Newqueuesfrozen afterallchanges;DO NOTedit their snapshots/workspacebindings.

FAILUREPUBLICATION916ACTUALLYLIVE,tool38265,executionB1_failed_basic_course_publication_20261101,
scriptpublish_failed_B1_basic_course_20261101.py. Completefailedseed11archive
native_B1_basic_seed11_failed_gate_20261101_repro.zip:49871991bytes/553files,
SHAd13dcf3b07fe1574d8823e23f5c29abcc6e43bbdd1d68e1e5e9b506a5c0aa22a.
96parts512KiB;manifestLAST;finite30min/3attempts/missing-only;then actualpublic
re-download/reconstruct. Packagedfigureexists,so no repeatofB2baselinebug.
Notyetpubliccomplete. Keepmainnative_road_checkout285cd30FIXED while916usesit.
Secondarybasic_course_code_checkout_20261101at99479b5isnowIDLE(canadvance).
PublicH/H0andH/B2all5fullrebuildsCOMPLETE;doNOTrepeat. GoalACTIVE/UNMET.

# Actual comparison rebuild and basic-course outcome — 2026-11-01 07:49

PUBLIC H/B2TEN-CAPSULE RECONSTRUCTION COMPLETE at285cd308b78824d53347e25a3badd442bb5f4522.
Every original paired-report field identical,10actualAdam/normchecks,5distinct
trained tensors eachmethod,exactmainfigure81efcd7bf430d8e6e387f6e8186af2a2e82c024d8b62fa0053a7fa389ca9bb78.
Independent178tests9.69sec. Outputpublic_H_B2_five_seed_rebuild_20261101;
receipt native_public_H_B2_five_seed_independent_verification_20261101.json.
Uses10actual prior public downloads with receipt/manifest/archive hashes rechecked,
NOtrainingarchives substituted. Mainrawcontrols259138;extraH0siblings also audited.
OriginalB2executionregisterCRLFbyteshasha39dd46...are preserved using per-file
.gitattributes -text,not weakened comparison or fabricatedmetadata. Reproducer
sourcecheckoutnative_road_checkout285cd30isnowIDLE;mayadvanceafterlivejobcheck.

CORE:basic-PPOseed11COMPLETE,10240controls,9ACTUALAdamsteps dueKLearlystop,
3/4success with1out_of_bounds(case3),raw11966/4actions audited. QualificationFALSE.
Reportnative_B1_basic_PPO_seed11_endpoint_20261031.json. Fullqueue66252cannot
start from failedresult;will reject onceall5basiccoursesfinish. Othercourses
still runall5asregistered;62176nowseed22batch20261031T232842374284Z_native_B1_basic_PPO_course.
Do not call baseline healthy or pretendPPOused320Adamsteps. No droppedseed/waivedgate.

Next ROOT-CAUSE work:investigate actorKLearlystop starvingcritic/learning before
anotherexpensive nativevariant. Current8rollouts permit320Adamsteps nominal but
only9actuallyrun;criticexplainedvariance negative. These are observations,not
causalproof. Existingcritic_rate_policy.pysupportsactor1/critic30profile,doNOT
blindly stack it. A meaningful bounded matched-replay diagnostic can reconstruct
the FIRST1280TRAIN controls from measured_demo,recorded bridgebegins,original
Torchrng/numpy states;uncappedstochastic actions must reproduce recorded clipped
actions. Must reproduce actualfirst native checkpoint's weights/Adam state before
using any offline LRalternative comparison. Never treat a surrogate replay or
TRAIN fit as driving success. Final120conditions remain UNTOUCHED/goalUNMET.

# Independent next-stage verification — 2026-11-01 07:38

Source3bfeb77b93dade0f4146edb7abcf916d9d71eb1aispushed. Secondarycheckout
171relatedtestsPASS9.22sec;actualtrainedcheckpoint1280restoreforfullcaseconfig
retainsAdam/RNG/weights,normmoments,0reset/control/update. Receipt
native_B1_full_after_basic_independent_code_verification_20261101.json.
Conditionalqueue66252livewaiting62176,created[]/completed[],nofulltrainingyet.
Basicseed11at5426controls/4optimizations07:37;all5basicresultsstillpending.
B2all5publiccomponentcomplete;maintainresearchUNMETanddoNOTrepeatthispublication.

# Latest actual progress — 2026-11-01 07:37

B2PUBLICALL5COMPLETE. Inventorynative_B2_all_five_public_completion_20261101.json
verifies133687controls/60actions/allfigures. Publisher70304terminalsuccess;
original45944/47464gone. native_road_checkout6a5486amayadvanceaftercheckingno
other users. NOreupload/retrain needed. Alloriginalfailuresretained.

NEXTSTAGE ACTUALLYQUEUED:driver66252,tool31512,execution
B1_full_after_basic_PPO_execution_20261101,immutable328frozenfiles (actualmanifest
count authoritative). Exactread-onlyhandle62176. RequiresALL5trainedbasiccourses
qualified then orderlydriverexitandallnativecleanups,otherwiseNOchild starts.
Factoryretainsactualtrainedcheckpoint/optimizer/norm/RNG,restoresoriginal12
conditions,adds20480controls,policylineage30720. MoretotalbudgetthanH20480,
notanequalbudgetclaim. Every1280save. Noextraactorfit/newfinalconditions.
Actualtrainedcheckpoint1280CPURESTORE with full-caseconfigSUCCEEDED,Adam1and
numtimesteps1280retained,0resets/control/updates;queueactivationreportrecordsproof.
Root21relatedtestsPASS3.92sec. Actualnativefull-tasktransitionstillpending.
Corecourse62176/35632live,3841controls/3actualAdamobserved07:34. Futurefullstage
mustNOTusepristineprofileverifierwhichcorrectlyrejectstrainedmodels. GoalUNMET.

# Current actual evidence — 2026-11-01 07:32

Restartseed11checkpoint_000001280ACTUALLYSAVED,Adam1,all3model/norm/RNGhashes
verified. Newrawcontrols1353observed,worker35632anddriver62176live. First lost
oldAdamupdate remains lost/charged,not confusedwithnewretainedAdam. B2seed44
ACTUALPUBLICRECONSTRUCTIONCOMPLETE;receipt20261101nowexists. Remaining55is
uploading under70304,tool23727;correctobjectsnotreuploaded. Independent166tests
PASS9.83secsource284a52c. GoalstillUNMET,allnewcourseendpointsstillpending.

# Current proof — 2026-11-01 07:31

Code284a52c9b1140b044e5f5699deb204de8f47ae69pushed. Secondary independent
checkout166relatedtestsPASS9.83sec;root16newtestsPASS3.39sec. Newcourse62176
andworker35632ACTUALLYLIVE,seed11newbatch20261031T232838093068Z_native_B1_basic_PPO_course,
879controlsobserved07:31. No retained trained weights from oldinterruptedcourse.
RemainingB2publisher70304ACTUALLYLIVE,tool23727,44currentlypublicreconstruction;
next55. It now passes explicit versionedfigure;maincheckout6a5486astayspinned.
Previously recorded processhandles/status timestamps are historical,notliveness.

# Authoritative external-interruption recovery — 2026-11-01 07:30

OriginalPython/CARLAprocesses gone;causeUNKNOWN,not claimed reboot. Original
PID25348reused by svchost.exe7:22;NOTmanaged/killed. Last genuine heartbeat2408,
1actualAdamupdateat1280,butNOtrainedcheckpoint. Actualinterruption_audit.json
preserves truncated gzip originals,unknownunflushedtail,consumedlowerbound2408.
Explicit NEW pristine-initializer experiment,not trained-weight recovery.
NewprepB1_basic_PPO_interruption_restart_preparation_20261101ALL5;seed11cost+2408
and lostAdam1declared;other4zero. Source/learnerunchanged except operational
checkpoint intervalnow1280. Oldraw/batches untouched. Newdriver62176LIVE,
executionB1_basic_PPO_execution_20261101_v2_external_interruption,tool45134,
seed11batch20261031T232838093068Z_native_B1_basic_PPO_course. Read-onlyCIMchecks
rejectmatchingoldprocessnames,acceptabsent/reusedforeignname,noforeigntermination.
Require oldmaturationfailedall5andnormalcleanupsstillchecked beforedriving.

B2original45944andreferenceverifier47464also gone. Public11/22/33previously
ACTUALLYCOMPLETE;44hadpartuploadfailures before networkinventoryquery failed.
Missing-onlyremaining44/55resumerACTUALLYSTARTED tool23727,execution
B2_remaining_publication_20261101_v2,scriptpublish_remaining_B2_20261101_v2.py.
Itpreservescorrectremoteobjects,usesfreshpublicdownloaddirs,and nowpassesexplicit
versionedreferencefiguretoindependentreproducer. DONOTadvance native_road_checkout
6a5486auntilthisresumerterminal. Secondary basic_course_code_checkout_20261101
is availableforCPUtestsandmayadvance. Mainrepositoryheadbeforethisrecoverye465bc3.
GoalACTIVE/UNMET. Do notdeclareoldtrainingstilllivefromstalePID/statusfiles.

# Actual update proof — 2026-11-01 00:08 local

BasicPPOseed11first optimization ACTUALLYDONE at1280real controls:1Adamstep,
actorL2change.011022,KLearlystopping retained. Do not assume320Adamstepspercourse;
report ACTUALoptimizer counts. Root163relatedtestsPASS6.66sec. Newqueue source
frozen326files remains immutable. Newnetwork verifications11/22/33complete.

# Authoritative continuation — 2026-11-01 00:07 local

MAIN CORE ACTION:all-five basic-motion PPO course25348ACTUALLYLIVE,
runs/B1_all_five_basic_PPO_course_execution_20261101,tool89590,frozen326files.
Old60572finished all5gate audits thenFAILED normally;shared70788/cadence59456
will exit their failed dependency. Do NOTrevive old gates/PPOor waive criteria.
Old matured fullscores9/4/5/5/7;4basic gates fail;23610raw evaluation controls.
Newcourse driver waited exact owned60572handle,verified all5terminal cleanups,
bound oldfailureSHA,and now drives seed11batch20261031T155537318174Z_native_B1_basic_PPO_course,
worker55336. It is collecting its FIRST1280-action PPOrollout;no update claimed
before an actual optimization log exists. Planned5*10240controls,all4basicroads,
no additional supervised fit. Gate AFTERlearning unchanged>=3/4zero unsafe.
All5pristine original parents used,same3e-5LR/1280rollout/320batch/10epochs/.01KL/
-4std/.05reward units/frozen obs. No healthy full-task or equal-budget H/B1claim.

New scripts:prepare_native_basic_motion_ppo.py,run_native_basic_motion_ppo_cohort.py,
analyze_native_actor_maturation_fit.py. TRAIN-only actual all10before/after
inferences show lower errors for every matured seed despite bad driving;not causal
or holdout evidence. Do not add another blind actor-fit pass. Actual frozen
restores and ValidateOnlyALL5pass;5distinct policy tensor hashes,0reset/control.
Found actual audit_native_ppo_configuration.py bug:action_space.nassumedDiscrete;
fixedcontinuousBoxbounds/shape/logstd,discreteprofileunchanged. Root25relatedtests
pass3.75sec incl13newcurriculumtests;all5actual saved learner profiles match.
Prepared courses contain frozen original audit module,but newdriver's immutable
snapshot contains the fixed continuous verifier. Nativeworker doesn't callthat
initializer audit;actual restore confirms runtime compatibility.

B2PUBLICSUPPLEMENT:originalpublisher45944stilllive,sourcecheckout6a5486aMUSTSTAYFIXED.
Its public raw/model audits pass but archives lack packaged figure baseline,so
originalreproductions fail correctly. DO NOTalter ZIPs/parts or weakenfigurechecks.
All5independent versioned referencefigures nowreports/figures/native_B2_seedSEED_trained_endpoint_20261101.png.
Separate verifier47464,tool75520,executionruns/B2_versioned_reference_public_verification_20261101
reuses exact original verified PUBLICdownloads,passes --reference-figure,and
writesnative_B2_seedSEED_public_verification_20261101.jsononlyonsuccess.
11/22/33alreadyACTUALLYCOMPLETE;waiting44download,next55. No repeated experiment
or correctpublicdownload. Originalfailed dirs retained. Do notadvancecheckout
until BOTH45944and47464terminal. Both are finite andnoCARLAcontrols.

GoalACTIVE/UNMET. FinishbasicPPOlearning,then develop/verifyfull-task B1 with
allcurriculumcosts charged;feedback/RL-vs-rule value and NEWfinal120conditions
remainunmet. Do not repeat finishedH/H0five-seedpublicreconstruction/B2training.

# Latest gate result — 2026-10-31 23:45 local

B1matured seed22gate FAILED:Require75percent unobstructed-motion success
with zero basic-motion unsafe episodes. All-five driver60572continues seed33,
then44/55as promised;it will not train any PPOif one gate fails. Seed11passed
4/4basic but9/12full with3bounds. The fully healthy baseline remains UNSOLVED.
Do not describe actor-maturation/basic-gate completion as healthy baseline or
repeat this same failed version. Preserve all5outcomes and original failures.
New development must diagnose basic failure and version any necessary learning
curriculum/observations/training redesign;never waive the frozen gate or alter
started protocols. Shared/cadence downstream require the completed five trained
B1policies and will not run from these gate models.
Rootc2b0191is pushed;H/H0all-five public reconstruction is actually complete.
B2publication45944stilllive,seed11uploading;sourcecheckout6a5486amuststayfixed.

# Authoritative update — 2026-10-31 23:41 local

FULL PUBLIC H/H0FIVE-SEED RECONSTRUCTION SUCCESS at source
6a5486aa4083bc2f14f32762c1983fc0e5dfbafa:6actual prior public downloads rechecked,
all reference fields identical,10Adam/statistics proofs,5distinct policies each,
main PNGexactcafca085ed2550173858f647a6de72c4dccbcedfa6ef03497325720c8c94eb87.
Source independent150tests12.21sec,hosted41215466456SUCCESS.
Receipt reports/native_public_feedback_five_seed_independent_verification_20261031.json.
Actual output runs/prerelease_20261022/public_feedback_five_seed_rebuild_20261031_v3_verified_public_cache.
The public-cache option validates original public URL/receipt/manifest/fullarchive;
no local training archive substitution. Earlier two failed network attempts retained.

B1maturation driver60572is driving seed22gate after seed11passes4/4basic,
9/12overall. No new B1PPO yet;all5gatesmustpass before any PPOstarts.
Shared70788andcadence59456remain queued. Actual status files override this snapshot.
B2ALL5native complete44/60vsH58/60,zero unsafe. Additional bridge audit now binds
all5H/B2bridgehashes to walker+.2m pre-settlement initialization patch.
Thus numerical difference is NOTan isolated causal planner-effect proof.
Report native_H_B2_five_seed_development_harness_audit_20261031.json;figure
caption/readme disclose this. Families H/B2:curve20/19,static18/6,dynamic20/19.

B2public capsule preparation ALL5complete;finite1hour publication driver45944
ACTUALLYLIVE: runs/B2_five_seed_publication_20261031,status.json,tool16248.
Script runs/publish_B2_five_capsules_20261031.py;2upload workers,atmost3attempts
permissing part,no correctasset overwrite,manifestLAST,then actual public
re-download and independent reconstruction perseed. Perseedpublic verification
reports are written only on success. Remote prerelease had517assets beforethis
queue;all5B2manifests use512KiBparts. No B2public completion claimed yet.
IMPORTANT:the publisher also uses native_road_checkout6a5486a;DO NOTadvance that
checkout until this driver is terminal. Root may commit docs/reports normally.
Goal ACTIVE/UNMET:healthyB1,feedback/RL-vs-rule benefit,newfinal120conditions and
lightweight independent thresholds remain. Continue core research,not repeat
completed H/H0training/public reconstruction/B2training.

# Latest continuation — 2026-10-31 23:34 local

B2 ALL FIVE COMPLETE:11/22/33/44/55 score9/8/10/10/7 of12,zero unsafe,
103003training controls,3250actual Adam,133687raw controls,60initial actions.
Driver32792exited success;B1driver60572has automatically started all-five gates,
currently seed11batch20261031T131044921585Z_native_B1_actor_maturation.
Shared70788andcadence59456remain waiting. Do not restart completed B2.
H/B2known-development paired comparison58/60vs44/60:mean+23.33pp,
95%t[+3.43,+43.24]pp,exact sign-flip p=.125. Exploratory,not confirmation.
Report native_H_B2_five_seed_development_comparison_20261031.json binds exact
initial model/statistics/RNG,physical budget/source and all episodes. Five B2
capsules have been prepared,NOTpublished;receipt lists every hash/part manifest.

H22missing public capsule COMPLETE:archive e70b381b397e344f2f6354f23aa81f841762b8a6675020f5e720fc1f8be951ed,
93mixed-size parts. Public receipt native_seed22_completed_H_retained_H0_public_verification_20261031.json
verifies599files/24858controls/12actions/exactfigure;68incomplete H0files and
2original unclosed gzip prefixes retained. Exact public reconstruction at346b2c5.
The missing-only repair finished and exited;all six H/H0capsules now public.

Independent checkout f127699passed144related tests11.10sec. Its first complete
public-cohort reconstruction correctly rejected a truncated download(part012);
the second retained manifest RemoteDisconnected. Neither generated results.
Those two failed output directories are preserved. New downloader adds bounded
manifest retries;new optional --public-download-cache rechecks original public
receipt/manifest/entire archive,never a raw local-training directory. Root25tests
for downloader/cache/recipe pass5.93sec. A fresh independent all-six reconstruction
using these previously verified public transfers is about to start;do not claim
success until its reproduction.json exists. Keep checkout fixed while running.
Research goal remains ACTIVE/UNMET;healthyB1,RL-vs-rule/feedback benefit and new
final120conditions remain outstanding. Earlier snapshots below are historical.

# Current handoff — 2026-10-31 20:45 local

LATEST22:10: B2seeds11AND22COMPLETE. Seed22batch104126873363scores8/12,
4deadlines,zero unsafe;20576training controls/650Adam;all27016rawcontrols and
12initial actions verify. Native_B2_v4_seed22_endpoint_20261031.json saved.
Driver32792now truly runs seed33batch20261031T104129314279Z_native_planner_training.
No B1maturation/shared/cadence native probes have started;their queued drivers
60572/70788/59456remain waiting. Original five-seed inventories unchanged.

Codeb0f11bc842095aa7baba03217dc1c6bb917b25f8is pushed;root85related tests PASS
8.14sec,independent checkout85PASS14.06sec,hosted41208046981SUCCESS. Receipt
native_control_cadence_independent_code_verification_20261031.json.
Current independent checkout is b0f11bc;no publication jobs depend on it.

Read-only latency diagnosis on12measured B1seed22initial states:observation
2.164s total,execution feedback2.058s(~95%),analytic connections/road SATdominate.
Profile runs/B1_native_observation_cost_profile_20261031.pstats. Reconstructed
all12actions exactly. Prototype occupied-rear-axle shortcut tested236captured
geometry batches/195743poses,all booloutputs exact,BUT zero shortcut hits and
two noisy timings give no useful demonstrated improvement. NOT ADOPTED;do not
spend more goal turns repeating it or alter live workers. Local executed source
runs/probe_exact_road_interior_rejection_20261031.py andrawprobe retained.
Report native_B1_observation_cost_and_rejected_fastpath_20261031.json explicitly
limits it to local exploratory input profiling,not published perf/realtime proof.
No controls/learning/new model fitting in this profiling. Goal ACTIVE/UNMET.

AUTHORITATIVE21:59update: B2seed11COMPLETE,9/12,3deadlines,zero unsafe;
20615training controls/650Adam;all26670raw controls/12initial actions verify.
Report native_B2_v4_seed11_endpoint_20261031.json. B2driver32792actually runs
seed22batch20261031T104126873363Z_native_planner_training,worker69724;
18434controls observed21:57. Other original B2batches unchanged.

Shared-brake postprocessor had an actual schema bug:it read row.trace instead
of row.info.trace. Corrected root runner and tests;scanner actually rerun on
H4337/B15019native controls. Before ANYprobe activated,old waiting63328was
verified by exact owned process handle/no children and intentionally closed.
Original sources/plans/prepared probes retained;orchestrator_superseded.json
records closure. DO NOTrestart63328or edit any frozen source.
NEW ACTUALLY LIVE shared driver70788,execution
runs/shared_brake_five_seed_execution_20261031_v4_trace_schema_fix,tool session62229,
frozen312files. Still waits on exact B1driver60572;newstate/failurefiles authoritative.

NEW ACTUALLY LIVE control-cadence queue59456,execution
runs/control_cadence_five_seed_execution_20261031,tool session24714,frozen315files;
waits through retained read-only handle to shared70788. After shared20complete,
it runs20H/B1probes:5seeds x2methods xholds2/5. Native20Hz trained endpoints
are reused as baseline. This is deployment-only requested-accel/steer hold:
physics/actuator inner loop stay20Hz,Hpolicy2Hz,MPCprediction grid20Hz unchanged.
MPCnot retuned;B1holds its own direct action. Actual recompute flags distinguish
held frames/failed solve events;recompute-only solver timings exclude holds.
No shared/cadence driving has occurred. No safety/final superiority claim.

Hseed11frequencyprobes ARE prepared and actual frozen-code restored,not driven:
hold2=20261031T133749867856Z_native_control_hold2,
hold5=20261031T133752011973Z_native_control_hold5. Both exact same policy tensor
hashes/norm/RNG;native launcher ValidateOnlypassed. Evidence
native_control_cadence_seed11_preparation_20261031.json and
native_control_cadence_and_shared_trace_recovery_activation_20261031.json.
Root-bound navstack/control_cadence.py, scripts/prepare_native_control_cadence.py
and scripts/carla_bridge37.py MUST stay unchanged while queue59456needs them.
New cadence module/overlay/trace31tests pass;shared schema10tests pass against
actual native schema. New source/receipts need push +independent verification.
Goal ACTIVE/UNMET. Main remaining bottleneck remains healthy B1and learned
contribution;fresh final120conditions must remain untouched until development
and preregistration qualify. Missing public completeH22capsule still unresolved.

21:26continuation addition: NEW shared-brake driver63328ACTUALLY LIVE,
runs/shared_brake_five_seed_execution_20261031_v3_matured_B1,tool session40474.
Waits through retained read-only identity handle to new B1driver60572;requires
all5matured B1PPOseeds and completed independent B2before any deployment probe.
Frozen312sourcefiles,plan profile all_five_matured_v1;seed11uses NEW matured
policy,not old11. Four preprepared Hprobes retained/reused;two OLD B1seed11
probes remain unstarted/unused. No shared-brake CARLAcontrols or results.
New root cohort resolver6tests pass;old21shared-brake tests still pass.
Receipt native_shared_brake_matured_B1_queue_activation_20261031.json.
Actual B2seed11still training,15501controls observed21:25. Root bindings and
all live snapshots immutable. Code9eec's hosted41205080566nowSUCCESS;source
38ce845metadata commit is also pushed. New shared-continuation code34a7f3b
is pushed and50related tests actually pass from its independent checkout in
13.04sec;root41changed-area tests pass11.35sec. Report
native_shared_brake_matured_B1_independent_verification_20261031.json.
Hosted34a7workflow41205580510is still running at21:29. B2seed11has17850controls/
56optimization phases observed21:28. No code/snapshot/weight changes to running
jobs. Goal remains ACTIVE/UNMET;next focus is actual queued native results and
the still-missing research/confirmation components,not another installation.

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
Latest code9eec6e651e7fdd4c5d8a8edb83a4d610178a7a04is actually pushed.
44related tests pass in root7.10sec and independent checkout8.55sec;exact feedback
figure and every H/B3comparison field independently reconstruct. Receipt
native_matured_B1_independent_code_verification_20261031.json. Code9eec's hosted
workflow41205080566is still running at21:22;earlier a19workflow41203628741SUCCESS.
Current B2seed11already14391controls/45optimization phases;B1queue60572still
waits through retained handle to32792. All research
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
