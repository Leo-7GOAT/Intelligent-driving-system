# Shared actuator interface ready — 2026-10-25 15:14 local

Goal remains active/incomplete. Previous turn obtained terminal evidence rather
than only restating status. This turn audits and independently reproduces the
low-speed failure, implements ff_state_v1, and runs the full regression.
No CARLA worker/server is running at this snapshot; old LIVE sections below are
historical. Do not restart either completed diagnostic.

Continuation run110908Z finishes3/4,2,010controls+123reset; normalcleanup,
rawSHA114e65701e8020801416d2744e8079fa3a5ce4945aff83681c6c76a26da7851e.
Low-speed run112429Z finishes0/4,3,600controls+123reset,934.28swall, all deadline;
guardexit0/reasonnull and ownedcleanup. Its raw audit independently reproduces
at28fc9f2 in `runs/prerelease_20261022/native_road_checkout/runs/low_speed_rule_raw_audit.json`.
Both fail the unchanged4/4coverage gate. No new PPO batch/pointer is created.

New `actuator_observation=ff_state_v1` requires shared candidate/road inputs,
appends speedsetpoint/pedal/filteredacceleration/integral (4705total), and requires
the measured acceleration_ff_v2 adapter/manualfirstgear. All H/H0/B1 see them;
H0 feedbackablation retains actuatorstate. Begin/step/final acknowledgements
export state, BridgePlant validates it, raw audit requires same-frame evidence
and initial reset agreement. Both policy types normalize/consume the four
inputs. Old4701models/configserialization remain supported. Routebehavior
history is corrected to selectedtarget and separately signature-versioned.
New model/normalizer required. No frozen legacy source changed.

Root fullregression:249passed/5conditional skips/2existing warnings,101.35s;
`runs/ff_state_full_regression.xml` and `.log`. Bridge/controller Python3.7syntax
passes. Resourcecheck:36.99GBdiskfree, GPU38°C (5060Ti16GB). New prospective B3
protocol is [registered](../reports/native_actuator_state_20261025.md) before run:
same native-score/continuation/45s/fourroads, only measured FFadapter/firstgear
and its observedstate. Preparation validates physicalcalibration equality and
exact measuredfit hash/coefficients. Execute after commit/preflight. Guard30min,
12GiBdisk/4GiBRAM/GPU85°C/no-progress300s/zero retries. If audited4/4, stage fresh
H0/H/B1seed11 at20,480controls each using frozen source/calibration. Otherwise
retain failure and diagnose; never weaken gate to force training.

# Historical: route-catalog iterations; continuation rule LIVE — 2026-10-24 19:18 local

Previous goal turn was progress. This turn implements versioned route candidates,
executes route-only B3, then implements/starts a continuation-feedback B3 variant.
Goal active/incomplete. No new PPO training is running. Do not restart live B3.

LIVE `runs/20261024T110908Z_native_rule`, frozen sourcef5108d0:
worker56684 creation19:09:43.986594 (shim764), guard50760 creation19:09:54.294798,
Shipping51628 creation19:09:23.6916966. Exec sessions worker39538/guard33251/
launcher28617. Worker identity is recorded inbatch/worker_process.json.
RPC2100..2102, matching0.9.15 empty-world probe; deadline19:39:08. Limits30min,
disk12GiB/RAM4GiB/GPU85C/no-progress300s/zero retries. Guard owns cleanup.
At19:18:21 case3 had55decisions/1,660controls; same OS identities/healthyguard.
First three roads succeed; fourth has begun waiting. Final outcome remains
unrecorded/unaudited. Do not treat open-gzip partial observations as final results.

New `route_frame_terminal` activates the existing route-tangent proposal inenv:
6/9/12m lookahead,5offsets,2speeds+wait; explicit route endpoint must matchgoal.
Requires common candidate/road input; H/B1 receive identical4701inputs and B1
remains direct control. Onlyreference targets are projected; measured state never
moves. Terminal goal targets use existing stopping-reference semantics. Signature
changes prohibit old weights/normalizer reuse. Frozen old sources are untouched.

Route-only batch `runs/20261024T105443Z_native_rule` at757b08f completes3/4:
controls359/381/353/900=1,993, plus123reset=2,116worldticks. Worker24192/guard50552/
server31232 exit normally with owned cleanup. Firstwait onroad3 isdecision25,
near(19.753734,11.842357,-.965868), goal(21.645828,6,-1.047673).
Raw audit SHA fefc82ede87a5e93c9272e2b1023bcf5ac75dc4df9e8e0f95d926782e416cd87.
Independent checkoutd307fb1 reproduces raw audit byte-identically; receipt:
`runs/prerelease_20261022/native_road_checkout/runs/route_only_reproduction.json`.
It passes23targeted route/evaluator/staging tests; root757b08f fullsuite228pass/
5skip. Rule road1 fullcycleP95~2.7056s, not realtime. Actual staging against this
parent was rejected by preregistered4/4coverage gate; logs retained, no training
folder or `runs/native_route_learning.json` pointer created.

`scripts/prepare_native_route_learning.py` (d307fb1) requires normal cleanup,
rawaudit and4/4, then stages fresh H0/H/B1seed11 at20,480controls each,32actions/
rollout,batch32,2epochs,globalclip,6hguard. No recovery weights. H config must
exactly match demonstratedconfig; source/calibration copied fromcoveragebatch.
It refuses duplicate registration. Preparing is not starting.

Exploratory console queries of route-only case3 atdecisions0/1/2 select26 with
target-goal witnesses. Decisions13/17/18 select28/18/22 without such witnesses,
while alternatives5/6/15/16/25/26,5/6/15/16,5/6/15/16have them. These were offline
production queries, not a saved independently reproduced sequence diagnostic.

Continuation sourcef5108d0 replaces candidate third(timeproxy) feedback with a
sufficient target-goal connector, including exactgoalpose with a free footprint.
Oldproxy is derivable fromsharedgeometry/speed. H/B1 share the new input; H0
`continuation_ablated` zeroes all99feedback values, not just the new component.
Distinct signatures; finite5m rule penalty forabsentwitness, no action mask.
All-absent witnesses retain original ordering/fallback. Fullsuite232pass5skip/
2existingwarnings (50.03s);12targeted pass; preflight393files/no findings.
f5108d0 is pushed. Independent checkout now clean atf5108d0; new continuation
tests/raw reconstruction pending until current evaluation ends.

Next: finish/audit currentB3 and normalcleanup. If4/4, prepare new PPO batch and
actually launch frozen native_road_learning with explicit NAVSTACK_CARLA_PYTHON,
owned launcher andguard. Ifnot4/4, retain failure; do not launch gated training.
Current partialtracecase3decision37 waits near(19.590615,13.195100,-.916848).
Investigate speed and replanning separately: earlier4/4fixed-reference
native_curve_control uses1.0m/s and a retained complete registeredpath; current
rule commonly selects1.5m/s and replans every.5s. They are not matched except for
selector. Do not infer a unique root cause or add a hard connector mask.
See `reports/native_route_catalog_20261024.md`; all final research gates remain.

# Historical snapshot: six-arm rollout complete — 2026-10-24 18:48 local

Previous goal turn was a verified wait/status summary. This turn is progress:
completed and audited the last arm, reproduced the complete study, actually ran
B3, implemented its matched-task/failed-state diagnostic and reproduced it.
Research goal remains active/incomplete. No agents or frozen learner edits.

**No training or CARLA evaluation is currently running.** At18:48 an OS process
inspection found no old worker/guard/server identities and no native training,
rule-evaluation or CARLA Shipping process. Both batches completed normally:

- `runs/20261024T075110Z_native_rollout_pair`: summary complete, worker57580 exit0,
  guard35276 reason null, owned server62284 stopped after empty-world probe.
- `runs/20261024T103653Z_native_rule`: worker16572 exit0, guard25516 reason null,
  owned server59536 stopped. Matched0.9.15 initial probe; RPC2100..2102 released.

Do not resume either completed batch or interpret stale heartbeat files as live.
The old `runs/native_rollout_pair.json` pointer identifies the completed study;
there is no global B3 pointer. A new experiment needs a new versioned batch.

Final B1_256:104,863 new training controls,10,496 decisions,41 phases,656 Adam,
225,000 retained lineage/234,545 charged lower bound;9,545 inherited unretained
controls and unknown ancestor tail remain disclosed. Retained Adam1,414.
Training109,047 worldticks/4,184 reset;134 episodes=111 deadlines+23 bounds,
zero successes;210 unfinished controls. Wall1,672.64s, exactreload, actor change
1.722352. Frozen0/4 all deadlines,3,600 controls/3,724 ticks, zero interventions,
mean progress6.637067m (short2.203275m). Long-short progress+4.433792m, controls
+2,230. All108,463 controls audited;134 returns reconstruct (error<=3.50e-14).
Initial longitudinal means saturate but steering remains within bounds.

Complete six-arm analysis covers639,009 training/eval controls;617,409 new
training controls. Every arm0/4; H0 meanprogress1.840831/1.149521m, H
14.987119/14.939079m, B1 2.203275/6.637067m. All paired measured starts identical.
`analysis/rollout_pair.json` SHAf726825c5bc9d913a58f7911152bbef6d7e71a4fc5dd36610b6f99192524f466;
full24-panel `analysis/frozen_endpoints.png` SHA7daf78a8c4f560d9bb2ebc4e6f3ef31f15aa18b3d4b2c19c070df01b648cde3f,
visually inspected. Independent checkout31fc769 reproduces7files (B1 long raw,
action,returns,initial inference,fullpair,fullfigure,B3raw) byte-for-byte:
`runs/prerelease_20261022/native_road_checkout/runs/native_rollout_complete_reproduction/reproduction.json`.
No new controls or downloaded asset from reproduction.

B3 actual result3/4: roads0/1/2 succeed in368/371/338controls, road3 deadline900.
Total1,977controls/123reset/2,100worldticks,39.94s wall,zero training or optimizer
updates. No plan/MPC failures or interventions. Meanprogress22.210610m.
All measured task identities and initial poses match H32, unchanged frozen
navstack/config/calibration. This existing heuristic is not assumed strong B3.
Raw audit SHAc9df6f6765b204be4cfe30beaa36bd82ee7877ac261c319f8531ae7610611e46.
Road3:35 reference decisions then55 waits. At firstwait(decision35) distance
1.946988m, atlast1.968249m; both states have zero moving references after all30
production candidate queries. Finalyawerror.022486rad, speed~0. Search failure
does not prove initial task infeasible. Waiting yields zero interventions despite
task failure. B3 success on3roads is evidence against attributing every PPO
failure to low-level incapacity, and is not PPO success.

New source6d862b1 `scripts/diagnose_native_rule_endpoint.py` validates parent,
shared production source/config, raw traces, measured tasks, and failed-state
queries.13 targeted tests pass locally/independently. Independent checkout now
clean at6d862b1, diagnostic byte-identical SHA375c72f428bb9ddc930c07fcc756ab573bee5e88a679e61e5365a5fdd4978ccf.
Separate receipt beside the seven-file receipt: `rule_diagnostic_reproduction.json`.
Last full suite remains203passed5skipped at ac59741; no new full-suite claim.
Publication preflight389files/no findings; new documentation commit follows.

Next development should address candidate coverage/approach selection rather
than repeat the completed clipping/rollout trials. Inspect existing route-frame
candidate helpers and terminal-prefix evidence before choosing the smallest
new interface version; do not present them as untried code or a proven cure.
Any candidate/feedback change requires corresponding H0/rule capabilities,
explicit compute accounting, new configuration and retraining, with B1 task and
information parity. Do not hard-mask by bounded goal-connector failure: prior
case1 diagnostics show absent connectors even where the executed B3 succeeds.
First establish the changed catalog's real closed-loop coverage, then test
learned high-level selection; no such new batch is started yet. B1 is still not
a healthy full-task baseline. Five independent seeds, new final conditions,
remaining baselines/ablations and all research gates remain open.

# Historical snapshot: B1 short endpoint complete — 2026-10-24 18:11 local

Previous goal turn was a verified wait. This turn completes B1 short training,
evaluation, raw audit, diagnostics and independent reproduction. Goal remains
active/incomplete. No code/learner changes or agents; this turn updates reports.

Same batch `runs/20261024T075110Z_native_rollout_pair`, pointer
`runs/native_rollout_pair.json`. Actual worker57580/guard35276/CARLA62284 verified
with original creation times; B1_rollout256 automatically started after the
short evaluation and was at10,719controls/1,075actions/4phases at18:10:10.
It is the last remaining training arm; do not restart. Deadline21:51:11 unchanged.

B1_32: 102,633newcontrols (233overshoot),10,272decisions,321phases,642Adamsteps;
222,770lineage/232,315charged lower bound,9,545known inherited controls absent
from the checkpoint and unknown unflushed ancestor tail retained. Total retained
Adam1,400. Training106,631worldticks/reset3,998; wall1,762.17s; actorL2change
2.961840,exactreload. 128completeepisodes=110deadlines+18bounds,0successes;
unfinished490controls. Frozen evaluation0/4all deadlines,3,600controls/3,724ticks,
zero interventions, meanprogress2.203275m. All106,233controls raw-frame audited.

Per-case progress5.180522/0/1.965038/1.667540m; finaldistance19.451258/24.347595/
20.817321/19.056279m; yawerror.144628/1.047028/.886807/1.460454rad. No pose-qualified
control; final speeds nearlyzero. Four initial actions exactly reproduce: [1,1],
[-1,-1],[1,1],[1,1], both Gaussian means outside bounds in each case. This is
initial-action inference only, not full sequence verification or execution.
128returns reconstruct; terminal-zero PBRS identity error3.50e-14. Earlier
bounded-mean lightweight trials are not a novel untried fix or native proof.

Independent clean checkout31fc769 reproduces fivefiles byte-for-byte:
rawaudit, actiondiagnostic, episodereturns, policyinference, endpointanalysis.
Receipt checkout/runs/native_B1_rollout32_reproduction/reproduction.json.
RawSHAe23611efb5aa8fcf64a14d2002558e23f25daa49ae9789cf07aec0022f0f62a4;
analysisSHA602c1915e755afb0f1d50f8c2928ad021e8ea9eada52d6bb9a34f4183c599d56.
No extra simulator controls, new policy updates or downloaded asset. No code
change or new full-regression claim. All analysis sessions exited0.

Next: complete/audit B1_256, action/return/start-inference diagnostics, B1 pair,
full H0/H/B1 rollout report and figure. Only then normal batch/guard/owned-server
cleanup and actual native rule B3. B3 remains unprepared/unstarted. Do not confuse
four reused training roads or same-parent seed11forks with independent confirmation.
All research acceptance gates remain open.

# Historical snapshot: H continuation-witness diagnosis — 2026-10-24 17:53 local

Previous goal turn was progress (completed H pair). This turn implements and
tests an offline sequence diagnostic and executes/reproduces both H histories.
Goal active/incomplete. No navstack or frozen learner changes, agents or new
CARLA owner. Source 31fc769 is pushed; current documentation commit follows.

Same live batch/pointer and worker57580/guard35276/CARLA62284, original creation
times verified this turn. B1_rollout32 is at 60,721 controls / 6,080 decisions /
189 phases at17:53:05, normal heartbeat; B1_rollout256 queued. H/H0 arms complete.
Do not restart. Resource snapshot17:50: disk38.6GB free, GPU38C. Latest B1
checkpoint50,161 retains170,298 lineage and179,843 charged-lower-bound controls.

New `scripts/audit_native_continuation_witness.py` verifies frozen sources and
audited inputs, reconstructs all31 production references at every decision,
checks selected mode/failure reproduction, queries existing bounded analytic
connectors from measured pose and proposed targets to the task goal, and writes
all decisions. It never steps/reset a plant or queries RPC. False connector is
unknown, not unreachable; proposed target is not the next actual state.

Actual complete H32/H256 results: first no moving reference at zero-based
decisions42/37/37/22 versus40/37/never/22. Cases0/2/3 in both arms already pick
a local reference lacking a target-goal witness at decision0 although alternatives
have one. Across whole episodes such choices number15/0/16/7 versus14/0/14/6.
Current-state goal witness first absent15/0/13/0 versus14/0/12/0. Case1 has none
at all90decisions and no candidate witness initially: do not adopt a hard mask
from this. H256case2 still has references at every decision but55failed selections.
This separates earlier continuation choices from ranking; no new learned result.

Five targeted tests pass locally and independently at31fc769. Both full reports
reproduce byte-for-byte in clean independent checkout:720states/22,320action
queries. Files `{H_rollout32,H_rollout256}/continuation_witness.json`, hashes
7cca80847dcb43e382943c056c43b377b220d1bf90c187454ab8be725440cc4c and
5126beb6ec6bcfca77f14e9a6ede193bc34b4194fa83efa603b4b056e5c17a13.
Receipt: checkout/runs/native_continuation_witness_reproduction/reproduction.json.
Analysis handles37586/44246/24428/17555 all exited0; no analyses remain live.
No full-suite rerun claim (last203passed/5skipped at ac59741 remains historical).

Next remains B1 endpoints/full paired analysis and then actual unchanged-catalog
B3 after normal whole-batch completion/owned cleanup. B3 still unprepared and
unstarted. Any new continuation feedback must be versioned, shared with rule
comparators, computationally measured and retrained; this diagnostic alone does
not adopt it. Five-seed/fresh120-condition/strongB3/healthyB1 gates remain open.

# Historical snapshot: H rollout pair complete — 2026-10-24 17:40 local

Previous goal turn was a verified wait. This turn completes actual H256 training
and evaluation, raw audits, paired analysis and independent reproduction. Goal
remains active/incomplete. No learner changes, new agents or second CARLA owner.

Same batch `runs/20261024T075110Z_native_rollout_pair` and pointer
`runs/native_rollout_pair.json`. Worker 57580, guard 35276 and CARLA 62284 retain
their original 15:52:46 / 15:53:26 / 15:52:11 creation times, verified this turn.
H0_32, H0_256, H_32 and H_256 are complete. B1_32 automatically started after
H256 evaluation; B1_32 was at 15,689 controls / 49 phases at 17:40:32 and
B1_256 remains queued. Revalidate its current heartbeat and
OS identities before acting. The same deadline remains 21:51:11; do not restart.

H256: 102,400 new controls, 10,240 actions, 40 phases, 640 Adam steps; lineage
222,720 and retained Adam total 1,392. No unknown ancestor tail. 105,933 training
world ticks include 3,533 reset ticks. Wall time 1,771.89 s; actor L2 change
2.666812; exact reload. Training: 113 deadlines, zero successes, 700 unfinished
controls. Frozen evaluation: 0/4, all deadlines, 3,600 controls / 3,724 world ticks.
Raw audit verifies 106,000 controls; hash
7bc4d0b1b63bc27d80e251cf000f99c3ac5f7eed15baf915e7815b8330d97839.

`analysis/H_pair.json` is complete, hash
9ebacce2df160ae968adb5e92919be89a35e90139d5673fbbc77e486820fbba2.
Same saved parent and exactly matching measured evaluation starts. H32/H256
mean progress 14.987119 / 14.939079 m; intervention 0 / 0.483333; both 0/4.
Long-minus-short progress -0.048040 m, actual training-control difference -313.
This does not support adopting longer rollouts for H. Keep B1 running to finish
the registered complete comparison; no independent-seed interval from these forks.

H256 final moving-reference counts are 0/0/2/0. Cases 0/2/3 repeatedly choose
action 30 and fail planning (50/55/69 of 90 decisions); case 1 waits 53 times.
Final goal distances 4.413147 / 3.188710 / 7.476224 / 17.651154 m. All 360 policy
decisions reproduce; final selected probabilities .03710/.04138/.03607/.03719.
Case 2 reference probability mass .06903. No complete goal-pose-qualified control
in any case. All 113 episode returns reconstruct, PBRS identity error <1.99e-14.
Local reference availability is not an execution/parking proof or global no-path
proof. The result retains both catalog-continuation and action-ranking failures.

Independent clean checkout at 362a11a reproduces five H256 diagnostics plus the
complete H pair byte-for-byte; receipt in that checkout:
`runs/native_H_rollout256_reproduction/reproduction.json`. No new controls or
updates. Existing full regression remains 203 passed / 5 skipped at ac59741;
this turn changes documentation only and does not claim a new full-suite run.

Next: B1 endpoints, raw audits, action/return diagnostics and B1 pair, then full
three-method report and all-arm figure. After entire batch normal completion and
owned-server cleanup, prepare and execute native B3 using the existing runner;
it remains unprepared/unstarted. All formal five-seed/fresh-120-condition,
strong-B3/healthy-B1 and acceptance requirements remain open.

# Historical snapshot: partial figure reproduced — 2026-10-24 17:18 local

Previous turn was progress (H32 actual endpoint/audits and B3 evaluator). This
turn adds native rule frame verification and produces/reproduces the complete
12-case figure for the three finished arms. Goal remains active/incomplete;
no agents, no changed frozen learner and no extra CARLA owner.

Same live batch/pointer and processes. OS worker57580/guard35276/CARLA62284
revalidated17:18:58 with original creation times15:52:46/15:53:26/15:52:11.
H_rollout256=45,350controls/4,535decisions/17phases at17:18:57; guardhealthy,
deadline21:51:11. H0_32/H0_256/H_32complete, bothB1armsqueued. Do not restart.

Source362a11a adds `audit_native_rule.py` and freezes it with future rule batches.
Checks one complete bridge stream, every applied frame/state/command/IMU and
episode, all cases/routes, terminal metrics, source/calibration/config hashes,
zero training/update costs and timing summaries. Eight verifier/evaluator tests
pass locally and independently; synthetic tests, no actual B3 result. **B3 is
still not prepared or launched.** Parent completion/cleanup gate remains.

Plotter supports explicit partial arms, retains all cases per arm and lists
omissions. Default still requires all registered arms and preserves published
fullfigure rendering: clippingfigure byte-identicalSHA847e27ff7bf5a5e7136117f2e83eb71ce38b43f7284811ce8a471ef701b00014.
Livebatch `analysis/first_three_endpoints_v2.png` visually checked; all12failures
included, labeled PARTIAL3/6. Independentcheckoutcleanat362a11a reproducesimage
byte-identicallySHA589f39c1b7e592c48b1939171d08e5edcc39309da556389ec45f91c44de98937;
receipt `runs/native_partial_plot_reproduction/reproduction.json`. Earlier
first_three_endpoints.png preserved; v2 has final source-hash provenance.

Next remains actual H256/B1 completion and audits/fullpairedanalysis, then
ownedservercleanup and actual same-interface B3 run usingevaluate_native_rule.py,
followed bynewrawverifier. Do not repeat finished H0/H32analyses or releases.
No research acceptance gate is satisfied by the new figure or synthetic tests.

# Historical snapshot: H short endpoint — 2026-10-24 17:11 local

Previous turn was progress (complete H0 sequence reconstruction and independent
reproduction). This turn finishes/audits actual H32 training/evaluation, reproduces
six endpoint diagnostics independently, and implements a guarded native B3 runner.
Goal remains active/incomplete. No agents, no changes to the active frozen learner.

**Same live batch `20261024T075110Z_native_rollout_pair` and pointer
`runs/native_rollout_pair.json`; do not restart.** OS worker57580 created15:52:46,
guard35276 created15:53:26 and CARLA62284 created15:52:11 revalidated17:10.
H_rollout256 heartbeat14,830controls/1,483actions/5phases at17:10:20, guardhealthy,
deadline21:51:11. H0_32/H0_256/H_32 complete; B1_32/B1_256 queued. Handles
worker84287/guard18326/server69058 remain; revalidate live identities/progress.

H32 finishes102,713newcontrols (313overshoot),10,272actions/321phases/642Adamsteps,
223,033lineagecontrols, exactreload, actorL2change3.534195,2249.82s. Training
115episodes=110deadlines+5successes,490unfinishedcontrols. All5successes on
road7d46c3cc23, with100/30/30/10/10interventioncontrols and10/3/3/1/1failedplans.
Frozen probes0/4all deadlines, meanprogress14.987119m, zero interventions/plan or
MPCfailures. Raw audit verifies106,313controls; SHA
23a96a82f0e4e61f103f8e61acca2c9f3008d4d2d140db090a9c28ac2b587b7b.

Unlike H0, H32 finalstates have zero available moving references for all30actions.
All currentfootprintsfree; cases0/1/3have0freecandidateendpoints, case2has2clear
endpointswhose searches exhaustopen set. Goal distances4.484/3.250/7.193/17.610m;
waitcounts48/53/53/68 of90. All360policydecisionsreproduced; finalwaitprobabilities
.08790/.13544/.13432/.15486. Treat as continuation/catalog deadend evidence under
this planner, not global no-path proof or a reason to penalize wait blindly.
Prior route-frame proposal diagnostics also failed at terminalstates: see historical
handoff/audit_route_candidates.py before repeating that idea.

Independent checkout clean atbee41c4 reproduces sixfiles byte-for-byte:
rawaudit, endpointanalysis, actiondiagnostic, availability, sequence,115episode
returns. Receipt `runs/native_H_rollout32_reproduction/reproduction.json` inside
checkout. No newcontrols/updates or downloadedreleaseclaim. H256pair not complete.

New `scripts/evaluate_native_rule.py` requires normally complete parent plus
guard0 and ownedemptyserverstop before even preparing. Copies exact Hconfig,
routes, calibration and frozen source; selects existing rule_action while retaining
H environment semantics, logs rule_only_B3 and selection/fullcycle timing,0training.
watch_existing_pair now accepts this worker with existing guards/ownedcleanup.
Five runner/cleanup tests pass locally+independentcheckout. Actual preparation
against the live parent correctly rejected without creating a run directory.
**B3 is NOT prepared or started.** Protocol/commands in
reports/native_rule_protocol_20261024.md. Existing ego-arc heuristic is not yet
established as strong B3; evaluate then strengthen if needed, with costs disclosed.

Next: finish H256 and bothB1 arms; audit/endpoints/fullpairedanalysis. Once entire
batch is normally terminal and ownedservercleaned, prepare/run actual native rule
diagnostic with current runner and existing ownedlauncher/guard. Never introduce
second tickowner. All5seed/120freshconditions/healthyB1/RLvalue gates stillopen.

# Historical snapshot: H0 sequence reconstruction — 2026-10-24 16:52 local

Previous turn was progress (availability diagnostics, independent reproduction
and push). This turn adds full observation-history reconstruction: all 720 H0
evaluation decisions match frozen policies; independent checkout reproduces
both reports byte-for-byte. Goal remains active/incomplete. No agents or new
CARLA owner, no change to the live frozen learner.

Same batch/pointer and worker57580/guard35276/CARLA62284 remain live. Actual OS
identities were revalidated this turn at16:47; heartbeat16:52:35 records
H_rollout32=61,410 new controls/6,141 decisions/191 phases. Deadline21:51:11.
Both H0 arms complete; H_256/B1_32/B1_256 queued. Revalidate process/progress.

Source6cc39cd adds `scripts/audit_native_policy_sequence.py`, limited to
hierarchical ego-arcs, empty input queue and no moving actors. A read-only state
holder reconstructs prior commands/behavior/execution outcomes, with source,
model, normalizer, config, raw audit and availability binding. Each H0 arm
reproduces 360 decisions with unchanged normalization. All eight final argmax
actions fail planning despite substantial probability on reference alternatives;
see report table. Both arms'113 completed stochastic training episodes timed out,
so this does not justify a sampling/masking fix or changing deterministic evaluation.

Three new tests pass locally and independently, comparing replayed raw
observations with production reference/failure/wait transitions under enabled
and ablated feedback. Independent checkout clean at6cc39cd reproduces both
`policy_sequence.json` files; receipt is
`runs/native_policy_sequence_reproduction/reproduction.json` inside checkout.
No new controls/updates or downloaded-release claim. Last full suite remains
203passed/5skips at ac59741; six subsequent diagnostic tests passed targeted only.

Next: let current H/B1 pairs complete; raw-audit and analyze endpoints. Run
availability then sequence reconstruction for completed H endpoints to distinguish
voluntary wait from unavailable choices. Do not repeat completed H0 audits or
releases. Full five-seed/fresh120conditions/strongB3/RL-value/healthyB1 gates remain.

# Historical snapshot: H0 action availability — 2026-10-24 16:45 local

Previous summary turn was a verified wait: actual worker, guard and CARLA
processes were checked. This continuation adds a production-reference diagnostic,
executes 496 offline action queries at 16 recorded states, independently
reproduces them, and updates the completed H0 pair/reward evidence. Goal remains
active and incomplete; no agents used and no live frozen learner changed.

**Do not restart the live batch `20261024T075110Z_native_rollout_pair`.**
Pointer remains `runs/native_rollout_pair.json`. OS revalidated16:45: worker57580
created15:52:46.740937, guard35276 created15:53:26.451462, CARLA62284
created15:52:11.130147. Guard healthy; deadline21:51:11. H_rollout32 is actually
training,38,460newcontrols/3,846actions/120phases at16:45:40. H0_32 andH0_256
are complete; H_256/B1_32/B1_256 remain queued. Handlesworker84287/guard18326/
server69058. Revalidate identities/progress; no other tick owner.

Completed H0 pair: each102,400newcontrols/10,240decisions/640Adamsteps,
222,720lineagecontrols,113trainingdeadlines,0/4evaluation.32vs256 collection
has320vs40phases; progress1.840831vs1.149521m,intervention91.3889%vs96.3889%.
All measured paired starts match exactly. Partialpair `analysis/H0_pair.json`;
no completed H/B1 pair or independent-seed/held-out claim. Sourceac59741's
independentcheckout reproduces pair+two physical reward diagnostics exactly in
`runs/native_rollout_H0_reproduction/reproduction.json`:226completedepisodes,
700unfinishedcontrols perarm,maximum PBRSidentityerror1.19e-14. Same-road,
same-length deadline return variation is risk/comfort, not a discovered shaping
implementation bug. Earlier curriculum/progressreward failures remain relevant.

ac59741 also fixes missing hashed RNG state and verifies recovery copies before
parentclaims. Fullsuite203passed/5skips;17targetedtests pass independently.
These changes affect future restoration, not this batch's frozen source.

New `scripts/audit_native_action_availability.py` uses production decoding,
planning and timed-reference checks without reset/step/RPC. Current/frozen source
must match modulo Git newlines; archived bytes retain exact hashes. Dynamic
scenes without per-frame reconstruction are rejected. All8final H0 decisions
reproduce plan_failed, despite26/10/26/22(short) and26/6/26/22(long) available
reference alternatives endingcloser. This narrows these failures to action
selection rather than absence of local references, without proving eventual
parking or safe tracking. All16states/496actions retained in eacharm's
`action_availability_v2.json`; v1results preserved. Independentcheckout at530d168
passes3tests and reproducesbothv2reports byte-for-byte in
`runs/native_action_availability_v2_reproduction/reproduction.json`. First
byte-onlysourcecheck failure was CRLF/LF; historical H_global query was rejected
for changed independent_clip_ppo.py, so no historicalH result is claimed.

Code530d168 pushed to correctrepo; docsupdatedinthisturn. Next: continue current
H/B1 arms; audit/analyze completed pairs with existingrollout analyzer. Apply new
availability diagnostic to completedH endpoints to distinguish voluntary waiting
from missingreferences. Do not alter/restart frozenqueue or claim offline
references establish closed-loop catalog competence. All five-seed,baseline,
fresh120conditions,RL-value and research acceptance requirements still apply.

# Historical snapshot: rollout schedule — 2026-10-24 16:17 local

Previous goal turn was progress (finished/audited clipping study, implemented
rollout intervention, started actualCARLA continuation). This turn adds and
tests the correct rollout analyzer, audits the first completed endpoint,
verifies actual256-decision updates, and publishes/downloads/reproduces the
complete negative clipping asset. Goal remains active/incomplete. No agents.

**Same live batch `20261024T075110Z_native_rollout_pair`, pointer
`runs/native_rollout_pair.json`; do not restart.** Actual OS identities were
revalidated16:15: worker57580 created15:52:46.740937, guard35276 created
15:53:26.451462, CARLA62284 created15:52:11.130147. Handles remainworker84287,
guard18326,server69058. Guard healthy at16:16:53; deadline21:51:11. The current
arm is H0_rollout256,39,220controls/3,922decisions/15phases at16:16:56; four H/B1
arms remain queued. Revalidate actual handles and current progress.

H0_rollout32 completed102,400newcontrols,10,240decisions,320phases,640Adamsteps,
222,720lineage controls;105,933trainingworldticks (3,533reset),1032.81s. All113
training episodes aredeadline. Frozen eval0/4all deadline, meanprogress
1.84083058m/intervention0.91388889. Raw audit verifies106,000controls including
3,600eval controls; evaluationworldticks3,724. Modelreloadexact, fourstartactions
reproduce, action_learning_diagnostic.json complete. Startmaxprobabilities
.065007/.035431/.056283/.062852; still no task competence. Its endpoint analysis
is analysis/H0_rollout32_endpoint_analysis.json. This is only one completed arm,
not evidence for/against the32vs256hypothesis until the pair finishes.

New scripts/analyze_native_rollout_pair.py uses rebuilt raw frame audits and
checks actualcollection/optimizer boundaries, unchanged saved learner parameters,
budgetcrossing/overshoot, inheritedcheckpoint bytes/costs, sameconfigs/tasks and
measuredstartdifferences. It correctly does not demand clippinggradient records
absentfromthisprotocol. Defaultrequiresallthreepairs; --methodH0/H/B1 explicitly
partial. Six tests pass locally and in the independent checkout, including real
32/256PPO collection and2/16Adamsteps, plus changedparams/parents/roads rejection.
Source d35572bba8f4ce320e6838d32c0f66d1d13bd79e pushed/remoteverified. Lastfullsuite
remains193passed/5skips at92eee2b; no newfullsuiteclaim. Independentcheckoutclean
detachedd35572b, fixedseparateenvironment. TestXMLinsidecheckout:
runs/native_rollout_analysis_regression.xml.

Actualrunning-checkpoint schedule evidence in currentbatch/analysis:
- H0_rollout32_checkpoint_000051200_schedule.json:51,200controls/5,120decisions/
  160phases/320Adamsteps; also independently reproduced at
  checkout/runs/native_rollout_checkpoint_schedule_reproduction.json.
- H0_rollout256_checkpoint_000010000_schedule.json:10,240controls/1,024decisions/
  4phases/64Adamsteps; savedbuffer256, sameparentparams/epochs2/minibatch32/global.
These are scheduleaudits, not completedrawendpointaudits or improvedperformance.

**Clipping asset publication/reproduction is DONE; do not redo it.**
native_clipping_seed11_20261024.zip,110,064,724bytes/639manifestfiles, SHA256
bbea000c380e202c1801770679413dcf616b59dfc593423a5e57a31005f2bcfc. Exportedclean
d35572b, GitHubasset671661062 uploadedstate/size/digestverified. Existingprerelease
v0.2.0-dev.20261022 andhistoricaltag/earlierassetsunchanged.12auxiliarymetadata
redactions withhashreceipts; all6models/norm/resume/raw bytes unchanged. Retains
142,446controls/all24failures; noenv/CARLAbinaries. Upstreamrawnotduplicated,
butancestorcosts/unknowntailskept. Actualdownloadisruns/native_clipping_download.
Independentcheckout/runs/native_clipping_downloaded_reproduction verifies639files
beforeloading, reproducesall6rawaudits/24actions/fullpairreport/24-panelfigure.
reproduction.json andpair_reproduction.json storeactualreceipts;0newcontrols/
updates. FigureSHA847e27ff7bf5a5e7136117f2e83eb71ce38b43f7284811ce8a471ef701b00014;
pairreportSHAe3200867495c4d3c4bcee317f0b039f97b151d012fe68f8f57f8061d88c2ce4c.
Release notes nowcontain the completed negative result and verifieddownload.
Old100k asset alsoalreadyverified; noneofthis isresearchacceptance.

Next concrete work: continue current256arm/queuedH/B1withoutchangingfrozenlearner;
auditcompletedendpoints, run newrolloutpairedanalyzer as pairsfinish, compare
actualtask/progress/safety outcomes. No paired result existsyet. Keep all five
independentseed/fairbaseline/fresh120conditions/RLvalue/releaseacceptance goals.

# Historical snapshot: rollout launch — 2026-10-24 15:54 local

Previous summary turn was a verified wait (OS-confirmed live worker, guard and
server). This continuation is concrete progress: all six clipping arms finish,
all raw/action/inference audits and the full paired comparison complete; the
next single-factor rollout study is implemented, tested, registered and really
training. Full research goal remains active/incomplete; no agents used, no hard
blocker. Read the user objective attachment before continuing.

**Use the new pointer `runs/native_rollout_pair.json`.** The old
`native_recovery.json` points to a completed clipping batch and must not be
restarted. New batch `runs/20261024T075110Z_native_rollout_pair` has six jobs:
H0_rollout32, H0_rollout256, H_rollout32, H_rollout256, B1_rollout32,
B1_rollout256. Each receives nominal102,400 additional applied controls; actual
whole-rollout overshoot and all ancestor costs remain visible. Same-parent
seed11 forks are not independent seeds. Global clipping, minibatch32, epochs2,
reward/observations/actions/low level/roads remain unchanged. Global parents
were fixed for all methods, not selected by which clipping arm wins.

At15:54 actual OS worker57580 (created15:52:46.740937), guard35276
(15:53:26.451462), CARLA Shipping62284 (15:52:11.130147) are live. Exec handles:
worker84287, guard18326, server69058. Venv shim worker26100/guard61296 are not
the actual guarded identities. Worker identity is in worker_process.json,
server identity in infrastructure/server_start.json. RPC2100–2102; matching
0.9.15 read-only start probe confirmed an empty world. Guard healthy, deadline
21:51:11 local, no-progress300s, disk12GiB/RAM4GiB/GPU85C, zero automatic retries.
Resource snapshot saved; roughly39GiB disk/35GiB RAM free, GPU42C at launch.
H0_rollout32 heartbeat8,380controls/838actions/26phases; last completed update
8,320controls/52optimizersteps. Resume/architecture receipts confirm32/32/2,
globalPPO and4701-input RoadCandidateActorCriticPolicy. All frozen source hashes
match. Revalidate current processes/progress instead of assuming these counts
are still current. **Set NAVSTACK_CARLA_PYTHON in every worker shell**, not just
the server shell; use the existing .carla-python37/runtime/python.exe.

Completed clipping batch `20261024T064940Z_native_recovery`: workerexit0,
guardexit0/reasonnull, owned-empty-server stopped and all RPC ports released.
H0/H four arms each20,160controls; B1 global20,087/independent20,119. All63phases
and126optimizersteps. Every arm0/4, all24evaluation cases deadline. Mean progress
global/independent: H0 10.3751/1.4325m, H16.7472/14.8990m,
B1 9.8687/0.7841m. Independent clipping increases gradients/KL without driving
benefit, so is not adopted. B1 global training has1success/14bounds/20deadline;
independent0success/10bounds/20deadline. Keep those separate from evaluation.
Six audits verify142,446appliedcontrols. All24initial actions reproduce exactly;
all6action_learning_diagnostic.json files complete. analysis/complete_pair.json
and all_endpoints.png cover all failures; the figure has been visually checked.
Independent checkout reproduces complete_pair.json byte-for-byte, SHA
e3200867495c4d3c4bcee317f0b039f97b151d012fe68f8f57f8061d88c2ce4c.
Receipt: checkout/runs/native_complete_clipping_pair_reproduction.json.
See reports/native_clipping_pair_20261024.md for accounting and limits.

New rollout code is source92eee2b (pushed and remote hash verified). Full suite
193passed/5conditional skips/2existingSB3warnings,107.96s; receipt
runs/native_rollout_regression_20261024.xml. Independent checkout is clean
detached92eee2b;19targeted tests pass there and all three100k horizon diagnostics
rebuild byte-for-byte (runs/native_rollout_horizon_reproduction/reproduction.json
inside that checkout). No new release-download claim. Published100k asset and
previous downloaded-model reproduction remain intact and need no repeat work.
New reports/protocol explain the GAE sensitivity hypothesis and its limits;
offline regrouping does not measure actual critic error or driving improvement.

Continue the actual new batch; inspect real256-action collection/update counts
when that arm starts, then audit endpoints and analyze all paired outcomes with
the same physical-accounting and matched-start requirements. Existing clipping
analyzer intentionally rejects rollout-study protocols; implement a correctly
labelled rollout comparison rather than changing the data's experiment flag.
Do not restart completed arms, mutate frozen source, or start another tick owner.
The broader five-seed, fair-baseline,120-condition,RL-value and publication
acceptance gates remain unmet; the live batch is development work only.

# Historical snapshot: H0/H clipping pairs audited — 2026-10-24 15:26 local

Previous goal turn was progress (actual release/download reproduction and new
CARLA paired training). This turn adds a hash-bound paired analyzer and three
meaningful tests, completes/audits the H pair, diagnoses all four hierarchical
arms, and reproduces both partial reports in the independent checkout. Goal
remains active/incomplete. No performance improvement is established.

Same active batch `runs/20261024T064940Z_native_recovery`; do not restart. At15:25:47
the actual OS processes are still worker49692 (14:50:22), guard61204 (14:50:48),
CARLA45556 (14:49:56); worker handle5858, guard9014, server61330. Guard healthy,
deadline16:49:41. B1_global is really training:6,048controls/18optimizationphases
at15:25:42. B1_independent remains queued. Existing source/protocol unchanged.

H0_global/H0_independent/H_global/H_independent all finished20,160new controls,
126optimizersteps,120,320lineage; all22training episodes and all4evaluation cases
deadline in every arm. Each endpoint raw audit verifies23,760controls. All four
initial_inference_audit.json and action_learning_diagnostic.json files now exist;
16measured-startactions reproduce exactly with frozen normalization.

H0 mean progress global10.3751m / independent1.4325m; mean intervention rates
0.78333/0.95. H0 independent repeats30/30/30/20 with85/82/90/85planfails.
H mean progress global16.7472m / independent14.8990m; both intervention0 and
no plan/controllerfailures. They instead wait too early: global53/53/55/64waits,
independent51/54/53/69. H finaldistances2.1086/3.2338/5.1649/14.9892m and
4.3987/3.4359/7.2694/17.7855m. Both pairs have identical measured startposes.
H actorgradient medians0.019069/0.093473 and KL1.4901e-7/1.6429e-6: the intended
mechanism changed, driving did not improve. Do not promote separate clipping.
B1 pairing must still finish before a complete six-arm conclusion.

New `scripts/analyze_native_gradient_pair.py` defaults to allH0/H/B1 and fails
if an arm/audit is missing. `--method H0` or `--method H` explicitly emits partial
evidence. Checks all frozen source, audited input/trace/bridge/model hashes,
same inherited checkpoint bytes/accounting, budgets/configs, road/task identities;
reports actual start-state differences and preserves all failed outcomes. No CI
from four roads/correlatedforks. Outputs:
`analysis/H0_pair_v2.json` (olderH0_pair.json preserved) and `analysis/H_pair.json`.

Sourcec9273ea is pushed. Three tests pass in both worktrees. Independent checkout
now clean detachedc9273ea and rebuilds both partial reports byte-for-byte:
`runs/native_H0_pair_reproduction.json` SHA660565c45137449e78bb5c315c76008b6f57b1d3fa6bce45918bf5cf82a20272;
`runs/native_H_pair_reproduction.json` SHAfb34148fc228f595d1cf59f6b954e2fc891c89729210f964db2c4d3d06f7015b.
These use local current-batch artifacts, not downloaded release claims. Last full
suite remains183passed/5skips at2040d3f; no unnecessary full rerun.

Next: finish/audit B1jobs, run measured inference/goaldiagnostics and analyzer
defaultall, verify guard/servercleanup, plotall6arms, then choose the next
development change using actual complete results. Keep all inherited/lost costs
and failedstartup recorded. Current protocol retains32-actionrollouts/2epochs;
do not quietly change the frozen learner or declare adequate convergence from
these short additional20kforks. Published100kasset needs no repeat work.

# Native 100k release reproduced; clipping pair running — 2026-10-24 15:10 local

This goal turn made concrete progress: B1 completed, all three raw endpoints were
audited, all twelve trajectories plotted, the full native asset uploaded and
downloaded/reproduced in an independent checkout, and the six-arm clipping study
started. Full research goal remains active and incomplete. Read the latest goal
attachment before continuing; do not substitute publication for research gates.

## Current live work: DO NOT restart

Active batch `runs/20261024T064940Z_native_recovery`, pointer
`runs/native_recovery.json`; this is the one explicit recovery of the registered
six-arm `20261024T064701Z_native_gradient_pair`. The latter's initial worker
failed before bridge creation because NAVSTACK_CARLA_PYTHON was omitted from
the worker shell. Failure/config/logs are preserved, no training controls were
applied, and its owned empty server was stopped. The new prefix-audit test handles
only this verified pre-bridge case, rejecting unexplained missing heartbeats.
No frozen source in either batch was edited.

The recovery preserves all six jobs, parent model/Adam/RNG/normalization bytes,
clipping modes and 20,000 additional controls per arm. Each of the three pairs'
five inherited checkpoint files matches byte-for-byte. Original paired-study
pointer still identifies the failed parent; follow its recovery_registration or
the native_recovery pointer, never launch it again.

Actual OS identities revalidated at15:08; heartbeat/guard still advance at15:10:
worker49692, created14:50:22.965068, executablePython311, venv shim parent61500;
guard61204, created14:50:48; CARLA Shipping45556, created14:49:56, RPC2100–2102.
Exec handles: worker5858, guard9014, owned foreground server61330.
`worker_process.json`/`infrastructure/server_start.json` retain exact identities.
Deadline16:49:41 local, disk>=12GiB/RAM>=4GiB/GPU<85C/no-progress300s/0automatic
retries. Initial free disk43.24GB. Only manage this task's owned processes.

Completed/audited jobs: H0_global, H0_independent, H_global. Each adds20,160
training controls,126 optimizer steps, reaches120,320 lineage controls, scores
0/4 (all deadlines), and its raw audit verifies23,760 training/evaluation controls.
H0 progress means: global10.3751m versus independent1.4325m; median actor gradient
norms0.016289 versus0.062073. More actor gradient has NOT repaired driving.
H_global per-road progress22.52/21.11/17.62/5.73m, all intervention rates0.
H_independent is actually training (1,190controls/3phases at15:10); B1_global
and B1_independent remain queued. H/B1 pairing is incomplete; no branch selection.

Next: finish/audit remaining jobs, inspect their actual learning and frozen-road
outcomes, compare every registered arm, and plot only after all endpoint audits
exist. Per-job audit CLI accepts job names. Preserve the negative H0 result.
Do not promote independent clipping based on gradient/KL alone. Current completed
job inference/goal diagnostics have not yet been run. Do not repeatedly rerun
passed full tests or the already-reproduced100k asset without a new reason.

Every future frozen worker launch must explicitly set
`$env:NAVSTACK_CARLA_PYTHON=(Resolve-Path .carla-python37/runtime/python.exe).Path`
in THAT shell. Environment variables from the server-launch shell do not carry
into a separate exec shell. Existing owned foreground launcher, readiness probe,
worker identity capture and watch_existing_pair guard are the verified path.

## Completed100k batch and real published evidence

`runs/20261024T050724Z_native_continuation` normally completed at14:43; worker
and guard exit0, empty-world cleanup succeeds, owned server/ports released.
H0/H retained100,160 each, B1 retained100,050 and charged lower bound109,595
plus unknown ancestor tail. All0/4: H0/H four deadlines, B1 three deadlines and
one bounds failure. B1 new79,989controls/87,459worldticks/506optimizersteps;
training212bounds/27deadlines/1success. That training success is not evaluation.
Raw audits verify83,600+83,600+82,819=250,019controls. B1 raw audit SHA
d634b3c9a1354ab92d04d41f88dcc1265d8dc7552af2653d4c4ecdd3fc42660d.
All four B1 initial actions reproduce; action_learning_diagnostic.json records
final distances17.3514/24.3476/8.7342/5.5366m. Full prefix audit is also complete.

Actual uploaded asset on prerelease v0.2.0-dev.20261022:
`native_structured_100k_seed11_20261024.zip`,175,728,217bytes,895manifestfiles,
SHA94ef2147966e191d89aa4d42fba9efa9e2d053fee3b863a31421bc31c207305b.
GitHub API confirms uploaded state,size,digest. Export source clean1494af3;
all frozen source/model/norm/raw evidence preserved;25auxiliarymetadatafiles
have explicit hash-tracked path redactions. All12trajectorypanels are included.
Release notes distinguish these failures from final research acceptance.

Independent checkout `runs/prerelease_20261022/native_road_checkout` is clean
detached2040d3f4b560125cb7ff1001a35739ee1e400578, with separate pinned interpreter
`../clean_checkout/.venv/Scripts/python.exe`. Actual full regression:
183passed,5conditional skips,2existingSB3warnings,98.62s; XML
`runs/native_100k_release_regression.xml` in that checkout.

GitHub download under checkout `runs/native_100k_download_proxy` ended with EOF;
the verifier rejected it before extraction/loading. A GitHub HTTP206 request for
the missing65,433bytes assembled a complete correct-hash copy under
`runs/native_100k_download_ranged`, with download_receipt.json. No local original
asset bytes used. A separate full retry also succeeded under
`runs/native_100k_download_proxy_v2`, same published hash. Slower initial direct
download62912 was ownership-checked and stopped; its partial file remains.

`scripts/reproduce_native_asset.py` at2040d3f verified every downloaded file,
rebuilt all three raw audits byte-for-byte, loaded all three models and matched
12measured-startactions with frozen normalization, and rebuilt the twelve-panel
figure byte-for-byte (SHAbbe75e47b0e43adbc154b0e6b3ff8c3e6d888bc18d76e85ec883813a5c5e8e8b).
Receipt: checkout `runs/native_100k_downloaded_reproduction/reproduction.json`.
Zero new controls/optimizerupdates; this is NOT a new live drive. Three negative
asset-integrity tests verify rejection before extraction/model execution.

Source commits e37f515(plot),1494af3(endpoint/startup),7eeae99(reproducer),
2040d3f(integrity tests) are pushed to the existing correct branch/remote. Latest
documentation commit follows this handoff. No force push or unrelated changes.

# Native evidence export validated; B1 continues — 2026-10-24 14:23 local

Previous goal turn was concrete progress (H endpoint, goal diagnostics, published
reproduction). This turn adds/tests a complete native-segment exporter, then
actually packages/unpacks the completed B1 recovery as a validation case. No
new learning settings or active frozen source were changed. Full goal active.

New `scripts/package_native_continuation.py` and3tests export only all registered
completed methods after guard/owned-server cleanup and fresh raw audits. Critical
inputs/model/norm/raw bytes stay unchanged; auxiliary JSON paths, embedded JSON
stdout and quoted process commands are redacted with original/export hashes.
ZIP/gzip/base64 content is scanned. Failed .partial exports are retained; no
existing archive is overwritten. All segment checkpoints and failures included;
ancestor raw traces not duplicated, but charged costs/tails remain in protocol.

Actual local validation archive `runs/native_recovery_export_validation_v4.zip`:
303files,11,263,723bytes,10,500B1controls,7metadata files transformed; SHA256
4b81e9a3e43a774dad65764ca82138945bcedd98a6387dd9743ad23e8c45fdb3.
Extracted under `runs/native_recovery_export_validation_v4/`; raw audit rebuild
`runs/native_recovery_export_v4_raw_audit.json` equals original byte-for-byte
(60b6404673d8f0a8b5bad87f098d318de96018b8a1f8f4ecd4b1db72096945a5).
`runs/native_recovery_export_v4_inference.json` loads exported model/norm and
matches4starting actions, zero new controls. This archive is NOT UPLOADED and
was built during development; a later README-only exporter edit adds exact CLI
instructions. Use a new asset filename on rebuilding after commit.

B1 was live at14:23 with32,666 new controls/103phases; same worker7156,
guard30068/CARLA60980 and batch below. Continue without restarting. After B1
normal completion: raw audit, inference and action diagnostic; verify guard and
cleanup; package the whole100k batch, upload to the existing development release
with explicit negative-results scope, download/reproduce in independent checkout,
and start the registered six-arm clipping experiment. Publishing source alone
does not substitute for publishing/reproducing the data/model asset.

# H100k complete/audited, B1 live — 2026-10-24 14:10 local

This goal turn made concrete progress: extended exact rollout replay to in-run
checkpoints, verified ordinary/independent PPO recovery, reproduced H40k and
B1-recovery diagnostics in the independent environment, and completed/audited the
actual H100k endpoint. Full goal remains active and incomplete; no research gate
is met. Read the latest objective attachment before continuing.

Same batch `runs/20261024T050724Z_native_continuation`, worker7156/guard30068 and
CARLA60980 were OS-revalidated at14:10, guard healthy. H completes80,000 new,
100,160 cumulative controls,500 new/626 retained optimizer steps,exact reload,
actorL2change1.5831946. Evaluation0/4,all deadlines; raw audit verifies83,600
controls, SHA256cb65c637dec4509a399bf7350a41512eb2ae29719b6e28f19f98c6a4ceb82842.
H0 is also complete at100,160/0of4. B1 is now actually training (8,370 new controls,
26 phases at14:10), not queued. Deadline remains19:07:25. Do not restart.

H/H0 `action_learning_diagnostic_v3.json` adds measured terminal goal criteria,
minimum/final distances and action/mode counts, bound to raw audit hashes. H roads
0/1/3 stop short while repeatedly choosing wait; road2 reachesposition but misses
yaw tolerance (final0.4076rad versus0.35), with35plan failures. Tolerances and
failure outcomes remain unchanged. Latest goal/recovery targeted tests6pass;
last full173/5 belongs to the previous source snapshot.

Source `d15b40b2f70c45cc862e6bac2da8ca964864a01f` is pushed/remote-matched for
mid-run replay and recovery tests. Independent checkout at that commit passes
the five recovery tests and rebuilds H40k/B1-parent replay outputs byte-for-byte:
`runs/prerelease_20261022/native_road_checkout/runs/native_midrollout_reproduction.json`.
Goal-error source38dc67353fcf0d619849bee667eabece7d9c9017 is pushed and
remote-matched. Its independent checkout passes six targeted tests and rebuilds
H's83,600-control raw audit and action/goal diagnostic byte-for-byte; receipt:
`runs/prerelease_20261022/native_road_checkout/runs/native_H100k_reproduction.json`.
B1 advanced to13,406 new controls/42phases at14:13, guard healthy, disk40.48GiB.
Both `rollout_update_after_40k.json`/`after_70k.json` probes for H0/H reproduce
exact observations,32actions and original updates. H late KL gain is only~1.7x,
so do not generalize the initial297x result. Current B1 first-rollout replay also
matches exactly (320controls); no offline alternative counts as driving.

Next: finish B1, raw endpoint/inference/action audits, normal guard completion and
owned-server cleanup. Only then prepare/start the registered single-factor
20k-per-arm six-job clipping experiment. No pair has been prepared or started.
Keep original baseline source frozen. Prior snapshots below retain provenance.

# Clipping diagnostic published/reproduced; training continues — 2026-10-24 13:53 local

The previous summary turn was a verified wait: worker, guard and CARLA identities
were live. This turn is concrete progress: paired-stage/registration tests,
current-code first-rollout replays and final regression completed. Full research
goal remains active and incomplete. Read the latest objective attachment first.

Root changes add `navstack/independent_clip_ppo.py`,
`scripts/audit_native_rollout_update.py`, `scripts/prepare_native_gradient_pair.py`
and opt-in named worker jobs, checkpoint-mode loading and per-group gradient logs.
Full regression: 173 passed, 5 conditional skips, 2 existing SB3 warnings;
receipt `runs/native_gradient_final_regression.xml`. Source changes are not in
the currently running frozen batch. Source commit
`c5f82852653a91bc677cb0a5a8259a8f77dd9879` is pushed and remote-matched. The clean
independent checkout with its separate pinned environment also passes173/5 and
reproduces both first-rollout receipts byte-for-byte (64 actions/640 reconstructed
controls, zero new controls). Receipt/recipe:
`runs/prerelease_20261022/native_road_checkout/runs/native_gradient_reproduction.json`
and `verify_native_gradient_reproduction.py`. This reuses local raw artifacts and
frozen observation source; it does not claim release download or new driving.

Exact measured-state/RNG replay reproduces 32 actions and the original first PPO
update for H0 and H; latest receipts are each method's
`rollout_update_audit_v4.json`. Global clipping factors are .04114/.04246;
independent clipping increases exact distribution KL from 2.1603e-9/1.4008e-8 to
6.4071e-7/1.7676e-6. Alternative branches are discarded offline diagnostics,
not new driving or performance evidence. Current implementation hash:
`36cf8cad2831fabacc4b5939bc7e0ec4d143aff3f12e320e130df542841f63fd`.
The report documents limits and executable reproduction commands.

Worker7156, guard30068 and CARLA60980 remain the same owned processes listed
below; all three OS identities were revalidated at13:53, H had56,520 new controls
and176 optimization phases, guard healthy; H0 complete, B1 queued. Revalidate
OS identities/log growth before action. The six-arm 20k-per-arm clipping study
is implemented but NOT PREPARED/STARTED. Finish and audit H/B1, verify normal
guard completion and owned server cleanup, then run the pair preparer. Its
pointer is `runs/native_gradient_pair.json`; standard worker/owned server/guard
launchers apply. Parent checkpoints/Adam/RNG/norm are identical within each pair;
all inherited costs retained. Do not count forks as independent seeds or use KL
alone as success. The active baseline source and learning settings stay frozen.

# Active combined native continuation — 2026-10-24 13:22 local

H0 is now complete: 80,000 new controls, 100,160 cumulative lineage/charged
controls, 500 new / 626 cumulative optimizer steps, exact reload. Raw audit
verifies all 83,600 training/evaluation controls. All 88 new completed training
episodes and all four frozen evaluation episodes are deadlines (0/4). Progress
20.8904/1.5977/20.1850/10.8140 m; initial maximum action probabilities only
0.03252–0.03307. See H0/endpoint_raw_audit.json, action_learning_diagnostic.json
and initial_inference_audit.json. H has started real optimization; B1 is queued.
No source or learning hyperparameter was changed inside this frozen batch.

Current batch: `runs/20261024T050724Z_native_continuation`, pointer
`runs/native_continuation.json`. H0/H/B1 are continuing compatible completed
structured-policy endpoints to approximately 100k lineage controls each. Seed 11,
four training roads, same learning semantics and low-level code; not final testing.
Additional budgets are 79,840/79,840/79,939. Deadline is 19:07:25 local; guard stops
on disk below 12 GiB, low RAM, GPU temperature >=85 C or no progress for 300 s.

Actual worker PID 7156, creation epoch 1792818490.558, guard PID 30068; CARLA
Shipping PID 60980, parent 60748, port 2100. Worker/guard/server were OS-confirmed
at this snapshot; recheck identity and progress before any action. Do not launch
a duplicate. Sessions: server 24387, worker 82406, guard 79209. The guard uses
`worker_process.json` and attempts verified empty-server cleanup after worker exit.
Read `guard_final.json`, `server_cleanup.json` and `infrastructure/server_stop.json`
to distinguish completed cleanup from an attempt. Frozen source must not change.

Prior structured endpoints: H0/H each 20,160 controls and 126 optimizer steps;
B1 recovery adds 9,870, reaching lineage 20,061 and charged lower bound 29,606,
with unknown ancestor tail retained. All three score 0/4. Strong per-method raw
audits and combined manifest v2 are complete; manifest ID is recorded in the
road-candidate report. The older non-v2 manifest is superseded and was not used.
The normalization pair is complete at 302,080 each: updating 4/10, frozen 3/10.

Current root regression: 163 passed, 5 conditional skips, 2 existing warnings.
PowerShell lifecycle scripts parse. Source 4b7db60 is pushed and remote-matched;
the independent pinned checkout also passes 163/5 and rebuilds all three parent
raw audits byte-for-byte (58,020 controls). Its receipt is
`runs/prerelease_20261022/native_road_checkout/runs/native_continuation_reproduction.json`.
Next: inspect real learning progress, audit
completed endpoints, diagnose persistent near-uniform/ineffective policies, then
update source publication and independent-checkout reproduction. Full goal
remains active and incomplete; this turn made concrete progress.

All entries below are historical snapshots, not current liveness assertions.

# Verified publication, H0 failure and live H — 2026-10-23 14:38 local

Source3c9e6f2b0f48fc15755b479d9010c6e13f0c0364 pushed and remote-matched.
Root and independent checkout-local full regression146passed5skips2warnings.
Clone is clean/detached at3c9e6f2, using existing separate pinned environment.
H0 has finished20160controls/20872worldticks/126optimizersteps,exactreload,
actorL2.21155748. Training22deadlines; frozen evaluation0/4,all deadlines,
346invalid endpoint plans. Independent clone matches all23760H0controls to raw
CARLAframes; reconstructs four measured starting observations and exactly
reproduces29/30/21/29. Maximum probability.03249–.03263,nearuniform,despite real
updates. Receipts/recipes inclone/runs/road_candidate_h0_frame_audit and
road_candidate_model_reproduction (.json/.py); not whole-trajectory re-execution.
See reports/road_candidate_development_20261023.md for hashes and limitations.

Same nativeworker18088/guard12856 and lightweightworker50724/guard55448 are
OS-confirmed live at14:38. NativeH past9600controls, B1pending. Lightweight
updating_stats reached~301k and is evaluating; frozen300k stillpending.
Do not restart either. Continue endpoint audits, independent artifact rebuilds,
and inspect server cleanup receipts after native completion. The running native
source predates only root guard timeout-receipt patch; never edit frozen source.
Full objective active/incomplete, no blocker. This turn is concrete progress.

# Active native representation experiment — 2026-10-23 14:30 local

The preceding summary turn was a verified wait: actual lightweight worker and
guard identities were inspected. This continuation is progress: native readiness
verified, real CARLA PPO launched, both 200k normalization endpoints raw-audited,
and cleanup failure receipts added with fault-injection tests. Full goal remains
active and incomplete. Read the original objective attachment before continuing.

NEW native batch `20261023T062141Z_native_road_learning`, pointer
`runs/native_road_learning.json`: H0/H/B1 fresh structured road/candidate policies,
seed11,20k controls each,45s episodes,2h window ending16:21:41. No reward/lowlevel/
action/teacher changes. Representation bundle versus old MLP, not a single-factor
architecture claim; H/H0 inside batch isolates feedback. Four training-road
endpoint probes, no heldout claim. Full details in
`reports/road_candidate_development_20261023.md`.
Actual worker18088 creation1792736844.863,launcher54140; actualguard12856,
launcher48688. Server shipping52312,parent19372,started14:21:44. Client/server
0.9.15 and emptyworld confirmed beforelaunch. Atlastinspection H0 had7360controls,
46optimizersteps,actorL2.06184. CheckOS+heartbeats now; never duplicate owners.
Frozen guard owns lifecycle and attempts empty-server shutdown. Inspect both
server_cleanup.json andinfrastructure/server_stop.json aftercompletion. The
later root timeout-receipt patch does not alter this running frozen source.

Existing lightweightpair `20261023T060219Z_normalization_retention_pair` remains
live underworker50724/guard55448. Both200kendpoints audited:202240cumulative,
99840newcontrols each. Updating0/10(6deadline4collision),parking0/1/0/0;
frozen4/10(5deadline1collision),parking0/2/2/2. Addedoptimizersteps1707/3062,
39phases each. All20normalization checkpoints checked. Partialreceipt at
continuation_verification.json. Updating branch now~245k enroute300k, thenfrozen.
Keep inheritedBCcosts andonecorrelatedseed explicit. Noformal gate achieved.

Current edits include structuredpolicy/tests, nativeworker, guard andPowerShell
cleanup. Full root regression now146passed5skips2existingSB3warnings in89.16s;
receipt runs/road_candidate_lifecycle_regression.xml includes latestguardtests.
Do not treat uncommitted source as uploaded. Preserve allfrozen sources.

# Verified continuation source — 2026-10-23 14:09 local

Sourcec60e0846d2cd6f97fcbe972d33db5ccf00deec40 pushed and remote-matched.
Root and clean independent checkout both pass140tests5skips2existingwarnings.
Clone receipt runs/normalization_continuation_source_verification.json records
the test XML hash and unchanged raw-reconstruction scripts relative to917d29e;
prior complete audit/figure/terminal reproduction remains valid and retained.
Same worker50724/guard55448 OS-confirmed live and healthy; updating_stats at
175847 cumulativecontrols/68phases. Frozen_stats still waits its200kturn.
No new endpoint has completed yet. Batch/deadline/ownership details below.

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
