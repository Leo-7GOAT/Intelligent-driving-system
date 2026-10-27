# Current handoff — 2026-10-27 22:35 local

Continue the original thesis objective. Do not restart the project, restore the
old plant, lower final gates, or claim completion. Never duplicate an activated
batch. Historical long handoff: ../reports/handoff_before_native_twelve_case_20261027.md.

## Actual endpoints

All twelve conditions are DEVELOPMENT: four known roads × curve/parking,
static car13m, dynamic walker. Earlier static10m failures remain separate.

| Configuration | Safe completion | Actual work / limitation |
|---|---:|---|
| B3 rule080111 |12/12|4329eval controls;4700world ticks;zeroPPO |
| H/H0 actor-only094138 |11/12each|896supervised updates each;9803eval controls;zeroPPO |
| H/H0 realPPO100609 |11/12,10/12|20541/20623train;650Adam each;51429total controls raw audited |
| B1 pooled BC095043 |4/12|1792supervised updates;4459eval;6unsafe;zeroPPO |
| B1 ordered BC102445 |3/12|same4329teacher frames;5265eval;5unsafe;zeroPPO |
| B1 phase4 BC112446 |6/12|changed fit mask/train-only norm;7123eval;1unsafe;zeroPPO |
| B1 corrective124542 |7/12|curves2/4,static2/4,dynamic3/4;4bounds;4965eval;zeroPPO |
| B2 frozen134723 |4/12|8639eval controls;866decisions;zero new training;Hparent11/12 |
| B1 realPPO130748 |4/12|20480train;640Adam;7857eval;28337current controls raw audited |

Corrective115212/122049 failed at owned actor57cleanup;3578+4030controls
remain retained/charged.124542 reran ALL12 with exact model/norm/RNG/config,
passed the former failure transition and normal owned-empty shutdown. Its total
three evaluation attempts cost12573controls.544additional actor-only updates,
2336cumulative;47counterfactual planner/MPC labels are computed commands,not
measured teacher execution. Corrective basic motion2/4unsafe is unsuitable as
stable initializer. ../reports/native_twelve_case_corrective_B1_endpoint_20261027.json.

130748 used phase4initializer with explicit basic_motion_training_v1 entry:
basic3/4zero unsafe, full-task6/12unqualified. Original strict rejections remain.
Final research gates unchanged. Frozen result4/12:curves1/4,static2/4,dynamic1/4;
two bounds,one collision.19training successes do not substitute for this frozen
result. KLmedian.023774,max.341952;45/64phases>.015. This supports a bounded
update-size diagnostic,not proof of cause. ../reports/native_B1_basic_motion_PPO_endpoint_20261027.json.

## Active sequential queue — inspect actual receipts before action

134723 native_frozen_B2 COMPLETED4/12,8639rawcontrols audited,normalguard0/ownedcleanup. Historical worker7496,
queue22776. Parent trainedH100609model/norm/RNGexact, onlyenv.method H→B2;
existing GridAStarPlanner with geometric postprocessing, same trajectory/MPC/
vehicle/task/budget. No training. Deployment planner ablation,NOTretrained B2
or independent confirmation. scripts/prepare_native_planner_ablation.py;
runs/native_frozen_B2_queue_20261027.log and batch queue_execution.json.

140111 native_supervised_ppo TRAINING COMPLETE20480controls/199Adam,now ACTUALLY FROZEN-EVALUATING all12conditions;worker57964/guard66000,
queuePID24524; runs/queue_native_KL_PPO_after_B2_20261027.ps1. Original and
new initial policy tensors,empty Adam,norm/RNG/config exact; same seed11,
20480controls,lr3e-5,std-4,7200s. ONLYlearner change is SB3target_kl=.01.
Normal per-rollout epoch early stop is not a hard bound on cumulative drift.
Same fixed endpoint/all12cases;no checkpoint selection or independent claim.
../reports/native_B1_KL_PPO_registration_20261027.json. Do not launch duplicate.
Queue requires normal guard,owned_empty_server_stopped and lifecycle lease
release. On error preserve receipts;do not bypass ownership checks.

141335 native_frozen_B2 PREPARED/VALIDATED and ACTUALLY queued after140111,queue41036. Existing body_spline_v1 uses150least-squares evaluations and swept-body/curvature certification. All8offlineopen/staticinitialqueries now reference-valid versus2/8legacy;no physical execution in queries. Both planner method and declared postprocessor change;model/norm/RNG unchanged. Runtime override is training_config ONLY,AST verified;other loop/plant code unchanged. See ../reports/native_B2_body_spline_registration_20261027.json. Grid search is finite-grid but does not enforce Hybrid max_nodes;record actual nodes/timings and do not claim equal compute.

Bounded long-training helper48844 ACTUALLY WAITING,not prepared/training. After140111and141335both normal owned cleanup,reaudit parent and continue its exact policy/Adam/norm/RNG to102400cumulative controls(+81920),10800swall,0retries;one fixed all12endpoint. Source/teacher hash changes fail closed. Charges parent eval controls and retains teacher NPZ. This is larger-budget B1tuning,NOTmatchedHcomparison or a healthy-baseline claim. See ../reports/native_B1_long_training_preregistration_20261027.json;local script runs/continue_KL_B1_after_spline_B2_20261027.ps1;parent long_training_queue_execution.json. Never launchduplicates.

## Source verification and publication

Remotehttps://github.com/Leo-7GOAT/Intelligent-driving-system.git,
branchfeature/physical-hierarchical-ppo,authenticatedLeo-7GOAT.
Source5c342ce ACTUALLY pushed. Hosted40510261079SUCCESS556passed/15skips/
4CARLA exclusions83.86s;independent38targeted pass13.58s. Previousb4134ae
hosted40506931939SUCCESS547/15/4in159.44s;independent67pass12.81s.
8f8257a ACTUALLY pushed/remote verified:hosted40517711560SUCCESS567/15/4in90.73s;independent full581/1skip/4CARLA exclusions285.46s. d974637body-spline source hosted40519686500SUCCESS570/15/4in121.18s;root49targeted6.09s/independent49in8.81s. ae3d238continuation source ACTUALLY pushed/remoteverified;hosted40521153471SUCCESS573/15/4in194.80s;root20targeted11.01s/independent20in10.96s.

Existing prereleasev0.2.0-dev.20261022 has actual publicly downloaded,
independently reproduced assets:three rule/BCcapsules1321files/18591controls/
36startactions/two byte-identical figures;realPPOcapsule587files/51429controls/
24startactions/byte-identical packaged figure. See model_card.md and reports
native_twelve_case_public_reproduction_20261027.json and
native_twelve_case_PPO_public_reproduction_20261027.json. Frozen legacyB2capsule is also actually public/downloaded/independently reproduced:433files/8639controls/12startactions/byte-identical figure. B1basic-motionPPOcapsule477files/43,808,905bytes/SHA b97dcb80ed2bf92f7898e39780c6fa52c77346304c26eaf93f2dc770b27f533c is currently UPLOADING,not yet confirmed public/downloaded/reproduced. Owned ghPID66688/toolsession3495;do not duplicate or claim uploaded before success. Its local package is runs/native_B1_basic_motion_PPO_seed11_20261027_repro.zip;all28337controls raw-audited. Other newerB1models local.
No final release or research acceptance.

## Constraints and remaining target

Disk12GiB/RAM4GiB/GPU85C floors;fixed wallcaps/no automatic retry. Existing
CARLA0.9.15 onD,main.venv and isolated.carla-python37. Derive ports from
protocol(currentRPC2100);never stop foreign processes by port alone.
Independent checkout:runs/prerelease_20261022/native_road_checkout; Python
runs/prerelease_20261022/clean_checkout/.venv/Scripts/python.exe. Never mutate
frozen batch sources/protocols; queued activation clocks follow wrapper only.

Still required:healthy adequately trained B1,PPOvalue vsstrongB3,explainable
feedback benefit,5independent trained seeds,120NEWpairedfinalconditions with
at least30each core family,unseen transfer,B2training/frequency/shared-safety/
control ablations,locked final protocol and statistical intervals. Final
targets remain light85%/75%core,CARLA80%/70%core,unsafe<=5%,H-B1>=10pp with
paired95%lower>0 (or a prelocked efficiency hypothesis before final testing).
Keep working;no guessed overall completion percentage and no false completion.
