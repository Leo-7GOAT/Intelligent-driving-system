# Current handoff — 2026-10-27 20:30 local

Continue the original thesis research objective. Do not restart the project,
restore the old plant, lower acceptance gates, or claim final completion.
The prior long handoff is frozen in
../reports/handoff_before_native_twelve_case_20261027.md; current facts below
supersede its pending statuses. Never rerun an already activated batch.

## Completed actual results

- Common development inventory:4known roads ×3families,12conditions;
  configs/native_development_cases_v1.json. Static car13m differs from retained
  earlier10mfailures. Not unseen/final conditions.
- Rule080111:12/12,4,329controls,4,700world ticks,0PPO. All raw audited.
- Actor-only BC094138:H/H0each11/12,896supervised steps each,0PPO;
  4,914/4,889evalcontrols. Hcase7deadline,H0case5deadline.
- Pooled direct BC095043:4/12,1,792supervised steps,0PPO;4,459controls,
  sixunsafe episodes. NOT a healthy baseline; no PPO continuation.
- Real PPO100609:H11/12,H010/12,20,541/20,623traincontrols,
  650Adam/65phases each,51,429totalrawcontrols. Guard0/reasonNone and
  owned_empty_server_stopped VERIFIED. HretainsBCscore;H0losesonecondition.
  One seed11;1/12development gap is not independent feedback confirmation.
  ../reports/native_twelve_case_PPO_complete_20261027.json.
- Ordered direct BC102445:3/12 versus pooled4/12;curves3/4,static0/4,
  dynamic0/4;fiveunsafe episodes;5,265rawcontrols. Same4329teacherframes,
  originalmask,seed11,128epochs/1792supervised steps. Rejected;0PPOstarted.
  ../reports/native_twelve_case_ordered_B1_endpoint_20261027.json.
- Fit-split phase4BC112446:6/12,curves3/4/static2/4/dynamic1/4,
  onecollision/severaldeadlines;7,123rawcontrols. Same4329teachercontrols/
  orderedhead/seed11/1792supervised steps;differentmask and train-only norm.
  Rejected,zeroPPOstarted. ../reports/native_twelve_case_phase4_B1_endpoint_20261027.json.
- Earlier static4PPO070950:H/H0both2/4,46,493rawcontrols,normalcleanup.
  ../reports/native_shared_small_exploration_complete_20261027.json.
- Hcase7offlinefrozen-sourcequeries:beforefailurechosen2no_path;
  actions12/22produce32-node references. Afteronebrake allqueried actionsfail.
  Fresh references only; no physical-infeasibility/completion proof.
  ../reports/native_twelve_case_H_failure_queries_20261027.json.

## Actually running

115212 native_corrective_bc STOPPED with an infrastructure cleanup failure
beforecase10reset. Parent guardexit1/reasonNone; owned_empty_server_stopped
verified. Original3,578evaluationcontrols retained/charged; no complete result.
Checkpoint model/norm/RNG/config preserved byte-exact. It starts from the
phase4initializer,adds544actor-only MSE steps at1e-4with corrective weight20,
2336cumulative supervised steps,zeroPPO. All4329originalteacherrows are now
training; NO holdout claim. Normalization/RNG files copied byte-exact from
112446;critic weights unchanged. All previous failures remain.

Corrective labels:47offline production B3/planner/MPC queries on actual102445
learnerstates,not measured executed teacher commands. Offsets0/20/60/120
clipped/deduplicated acrossall12cases;193.95sunder upfront1200sbudget.
Everyinput reproduces the exact saved-policy action;physicalstates unchanged.
Exact-input dataSHAfac797a235e2e797148d3c3c922d2cccf83a0affd8a042880ec77b73d33e5919.
Local runs/ordered_B1_corrective_oracle_20261027 retains allqueries,47labels,
executed producer,original/exactdatasets and proofs. No new controls/PPO.
Actor refit charges16717physical input controls:4329teacher+5265queried learner
source+7123parent selection evaluation. Further preparation preserves ancestor
costs/deduplicates already charged inputs; do not reset to teacher cost alone.
Original launcher session47845 terminated;failure/partial traces remain.
../reports/native_corrective_B1_interruption_recovery_20261027.json.

ACTUALLY ACTIVE:122049 native_evaluation_recovery,worker69496/guard63336,
deadline epoch1793105757.817. Same379frozen source files except2harness files:
bridge actor cleanup and recovery preparer. No navstack/actuation/planning/
policy changes. ALL12conditions reevaluated,not stitched from partialoutcomes.
Source validates. Missing teacher NPZ was restored with its already-declared
SHA;protocol/source bytes were not rewritten. Launcher session80455/log
runs/native_corrective_B1_recovery_launcher_20261027.log.
Recovery input cost20,295=16,717prior+3,578interruptedcontrols; no fitting/PPO.
Cleanup accepts false-destroy only after verified worldabsence,usingthe
existing one cleanup tick as retirement fence; survivors/RPC failures stillfail.
Pure evaluation prefix audit validateszero counters,empty optimization and
exact checkpoint bytes without fabricating a training heartbeat.65targeted
cleanup/recovery/case tests pass7.81s. Original failed prepare log remains.

Bounded conditional PPO helper53976 is ACTUALLY waiting for122049normalguard/
ownedcleanup. Entry gates declared BEFORE evaluation:>=9/12,eachfamily>=3/4,
zero unsafe. If qualified,prepare/launch20,480nominal real PPO controls,
7200swall,log_std=-4,guarded_progress_v1;else record rejection/noPPO.
Script runs/continue_recovered_corrective_B1_if_qualified_20261027.py;
receipt122049/qualified_ppo_queue_execution.json. Do not launchduplicates.
Earlier helpers19628/69348 rejected102445/112446;65484stopped on115212failure.

## Publication and reproduction

Branch feature/physical-hierarchical-ppo,correctremote
https://github.com/Leo-7GOAT/Intelligent-driving-system.git;authenticatedLeo-7GOAT.
Codefce4fdc is actually pushed. HostedCI40500309114passes530/15skips/4CARLA
exclusions193.46s. Root35correctiveactor/cost checks pass9.66s;independent
checkoutfce4fdc passes35in13.19s. The subsequent cleanup/recovery fixes have
65targeted passes; hosted/independent checks must be verified separately.

Existing prereleasev0.2.0-dev.20261022 has3new12case rule/BCcapsules actually
downloaded without auth and independently reproduced:1321files/18591controls/
36policy start actions/2byte-identical repositoryfigures. See model_card.md and
../reports/native_twelve_case_public_reproduction_20261027.json.
The full12case PPO capsule is also ACTUALLY uploaded and downloaded:
native_twelve_case_PPO_seed11_20261027_repro.zip,65,018,511bytes/587files,
SHA851c26b593abcc9fe353d8994deb58538bf8fc7014f8e19a132d49e48c67be97.
Independent CLI reproduction COMPLETE under5ababeb:all51429rawcontrols,
24policy start actions,frozen norm,byte-identical packagedfigure. Receipt:
../reports/native_twelve_case_PPO_public_reproduction_20261027.json.
New ordered/split/corrective models remainlocal.

Keep disk12GiB/RAM4GiB/GPU85C floors and original wallcaps. CARLA0.9.15 onD;
main.venv;isolated.carla-python37 client. Derive ports from registered protocol
(currentactualRPC2100); never stop a foreign owner based on port number alone.
No final research acceptance:healthy sufficiently trained B1,PPOvalue vsstrong
B3,feedback contribution,5independentseeds,120newpairedfinalconditions,
unseen transfer,fullB2/frequency/shared-safety/control ablations remainrequired.
