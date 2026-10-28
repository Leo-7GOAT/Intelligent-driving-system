# Current handoff — 2026-10-28 10:13 local

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
