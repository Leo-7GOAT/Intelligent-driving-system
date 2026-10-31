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
