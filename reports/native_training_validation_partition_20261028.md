# Separate native development training and validation

Latest execution readiness (31 October): the v4 batch
`20261031T025445532080Z_native_supervised_ppo` has passed read-only launcher
validation of all 389 frozen files and its unchanged registered protocol.
It retains 36 training/24 geometry-separated development validation conditions,
20,480 controls each for H/H0, and is waiting behind the actually running
lower-rate B1 probe. Neither native validation results nor final confirmation
are available yet. See the [actual prelaunch receipt](native_v4_broader_prelaunch_readiness_20261031.json).
The earlier batch descriptions below are historical execution snapshots.

The registered inventory is
[native_development_train36_validation24_v1.json](../experiments/native_development_train36_validation24_v1.json),
SHA256 `07d0724d87f4a3bb117265656e2a0d1296c02a749b1bd37186401d8229800961`.
It contains36training and24validation conditions,balanced12/8per core family.
These are development partitions; no final confirmation conditions are generated
or claimed. Previously inspected12conditions remain development evidence.

The334-route saved CARLA map survey supplies geometry,not policy scores. Four
validation roads are chosen by proximity to signed heading targets -30,-60,30,60,
with geometry-id tie breaks. Their sampled centrelines remain at least13.25m
from all routes in132historical top-level run protocols and each previously
selected validation road. The retained maximum segment spacing of1.25m makes
12m centreline clearance conservative between samples. Historical exclusion is
based on registered route inventories; it does not certify that no historical
vehicle ever visited a nearby location. The complete protocol hashes and route
identities are retained in the inventory. No policy results select these roads.

The original12teacher conditions retain exactly their original indices. The
other24training conditions add shorter goal-parking prefixes,static distances
11/15m and different crossing times/speeds. Validation uses distinct roads,
short/full parking goals,static distances12/14m and two crossing specifications.
No new teacher labels or supervised updates are created. Physics,low-level
planning/control,rewards,weights and normalization are unchanged by registration.

The worker now selects training cases only for on-policy collection and the
separate validation inventory only after freezing the endpoint. Raw audits bind
training and evaluation resets to their respective geometry inventories. Wrong
roads,overlap,duplicate geometry,sparse sampling,unknown fields,ambiguous
partitions and use of a final-test label are rejected. The strong-rule worker
and auditor resolve the same validation inventory. This adds partition handling;
it is not a claim of measured validation performance.

Batch `20261028T014801Z_native_supervised_ppo` is actually PREPARED and its388
frozen source files pass launcher ValidateOnly. It reuses fresh seed11H/H0
actor-only initialization,20480controls per method and7200s bounded wallclock.
Only explicit partition-handling edits are applied to the parent's worker;
the installer rejects unexpected worker structure. The original12-condition
initialization gate is not qualification on the new36/24conditions. Activation
and any measured results must be reported separately.

After current-source CPU checks passed,this batch was ACTUALLY QUEUED after
shared-mask seed22PPO013440. The bounded local queue checks normal completion,
owned-empty server stop,released lifecycle lease and unchanged child protocol
before activation. No training or validation result is claimed for014801 yet.

Validation:48case/partition/raw-frame checks pass5.60s;39partition-preparation/
PPO/cost checks pass5.82s;50partition/rule/case checks pass3.89s. After adding two
rule-partition fixtures,a fixture scene-copy mismatch was found and corrected;
all17partition tests then pass3.61s. The earlier full partition regression
passes625/1skip/4CARLA deselections161.95s before the last small additions;
do not attribute that count to later source changes.

The subsequent full run collected the old rule fixture before its correction
and ended628passed/1failed/1skip/4deselections186.53s. The corrected17partition
fixtures and both critic tests then pass19/19in8.86s. The critic optimizer now
inherits Adam.step and applies/restores rates with hooks,avoiding nested PyTorch
post-hook dispatch. A complete current-source rerun is recorded separately.
The frozen active013034source is preserved; its doubled optimizer hook counters
must be corrected from saved per-parameter Adam state,not silently rewritten.
[Actual2560-control rate/count verification](native_B1_critic_rate_live_verification_20261028.json)
records22actual Adam updates versus44post-hook calls,with unchanged actor/critic
rates3e-5/9e-4. No other earlier B1 profile uses this custom optimizer.

The final complete current-source rerun passes **629tests,1skip,4CARLA
deselections in211.28s**. The completed B1 training checkpoint now binds
20480actual controls and281actual Adam steps,versus562raw hook calls;
[training-state receipt](native_B1_critic_rate_training_20261028.json). Its critic
absolute output bound is14.4825versus unsafe terminal reward-90. The rate change
increased the bound but does not establish adequate value fitting or driving
competence. Frozen driving evaluation is still in progress at this snapshot.

Preceding shared-mask/critic source345b1ac is pushed and independently verifies
32targeted tests in8.44s. Its hosted [run40616036339](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40616036339)
passes598/15skips/4CARLA deselections194.63s. Those are CPU verification,
not CARLA performance or research acceptance.
