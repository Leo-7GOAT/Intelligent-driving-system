# Broader development validation: walker initialization recovery

The original v4 H/H0 batch `20261031T025445532080Z_native_supervised_ppo`
stops during H validation condition 17 because CARLA rejects the crossing
walker's registered spawn position. H training is complete at **20,521**
controls/650 Adam steps. H0 has not started. The original failure, partial
validation, checkpoints and consumed evaluation controls are retained.
The guard exits with worker code 1 and the owned empty server is stopped.
No completed H training is restarted.

An isolated owned-server probe repeats the exact failed XY/yaw. At the original
0.6 m world Z, spawning fails; at 0.8/1.0/1.2 m, spawning succeeds and all three
walkers settle to the same Z (approximately 0.9517 m) and XY. A vertical ray
identifies the road surface near Z=0.0002 m. This supports a capsule clearance
initialization fix, without shifting the crossing position or removing map
collisions. The probe creates/destroys walkers and advances settlement ticks,
but executes no ego control or PPO update. Final actor inventory and owned
shutdown probe are empty. [Actual probe](native_walker_spawn_surface_probe_20261031.json).

The native bridge now raises only the initial walker transform by 0.2 m, keeps
the existing thirty settlement ticks, and records requested/settled Z and capsule
half-height in native hazard evidence. XY, yaw, actor blueprint, walking speed,
delay, policy control, vehicle actuation, road bounds and success gates are
unchanged. This operational initialization change is explicitly versioned in
the source/scene evidence; old snapshots remain intact.

Recovery recognition accepts only the exact walker-spawn collision error with
a finite three-value local pose, worker exit 1, dead matching owned lifecycle,
empty ports and verified server shutdown. It rejects arbitrary learner errors,
live processes, mismatched ownership and malformed poses. Seventy-two targeted
recovery, reused-PID, hazard and preparation checks pass. An initial synthetic
test insertion misplaced the preceding test's assertions; that fixture mistake
was corrected before preparing or launching recovery.

The saved H endpoint is prepared for a full frozen 24-condition reevaluation as
`20261031T070638Z_native_evaluation_recovery`, with exactly copied model,
normalization and RNG and zero new training. Original interrupted evaluation
work remains charged. This is a new operational recovery evaluation, not an
uninterrupted original run, a new independent training seed or final confirmation.
The untouched H0 initialization must train in its own subsequent lifecycle;
the blocked original queue is preserved until that lifecycle is registered.

Recovery has now completed all **24/24** geometry-separated development
validation conditions, with **8/8 per core family**, zero unsafe episodes and
8,089 audited controls/24 reproduced initial actions. Model, normalization and
RNG bytes match the original final checkpoint. The original 5,367 interrupted
evaluation controls remain charged. See
`native_broader_H_spawn_recovery_complete_20261031.json` and
`native_H_v4_broader_spawn_recovered_endpoint_20261031.json`.

All 8,089 controls include 7,687 reference-tracking, 390 dynamic-brake and
12 terminal-brake controls; there are no policy-wait or failed-plan brake
controls. The environment-decision P95/P99/max are 0.210/0.359/1.171 seconds,
with one of823decisions above the0.5-second period. This timer excludes policy
prediction, normalization and runner serialization, so it is not a full-runner
realtime certificate. [Complete runtime audit](native_H_v4_broader_recovered_runtime_20261031.json).

H0 has completed as `20261031T071026696107Z_native_H0_spawn_recovery`,
using the untouched original zero-PPO model/normalization/RNG/configuration.
AST comparison finds only `Bridge.begin` changed; step/read/actuation and all
vehicle, reward and planning sources remain unchanged. Its separate 20,480-
control budget and all original training/validation conditions are preserved.
After normal owned cleanup, a registered queue reconnects the unchanged seed33,
44 and55 initialization gates and existing independent-PPO driver. The old
failed queue and preparation failure before any native lifecycle remain intact.
The H0 helper's missing vehicle argument was corrected before registration;
the failed preparation directory contains no native controls or optimizer steps.

H0 completes20,561training controls/650Adam steps and scores21/24:8/8curve,
5/8static,8/8dynamic,zero collision/bounds. Cases9,13,15 hit deadlines.
All30,292training/evaluation controls and24initial actions verify. The first
read-only inference audit failed because training scene files omit validation
roads. The corrected auditor reads the actual reset scene payloads from the
method's hash-bound bridge response stream;there are zero new physical controls
or updates. The original failure and both audit records remain intact.
[Complete H0 endpoint](native_H0_v4_broader_spawn_recovered_endpoint_20261031.json).

The24paired conditions yield three H-only wins and no H0-only wins:12.5pp,
exact two-sided p=.25. This one-seed development comparison establishes no
statistical feedback benefit or PPO-vs-rule contribution.
[Complete paired report](native_v4_broader_feedback_pair_20261031.json).

Seed33/44/55H/H0initialization gates have all completed. Their remaining matched
PPO program now waits for B1's registered36train/24validation phase to complete
normally. Only the prior idle waiting driver was superseded with a retained,
creation-time/executable-verified handle;no CARLA worker/server was interrupted.
The replacement freezes the preparer source and keeps all original seeds,
budgets,physical profiles and gates. B1 is actual worker4840;do not infer process
ownership from this stale text without checking the live lifecycle receipts.
