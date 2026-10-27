# CARLA actor retirement diagnostic, 2026-10-27

Two frozen B1 evaluation attempts stopped when starting condition10 after
condition9. The bridge reported `destroy actor 57: destroy returned false`.
Both workers exited1; their owned servers were verified empty and stopped.
The readable3,578/4,030controls remain recorded and charged,not scored as
completed twelve-condition endpoints.

The first repair checked actor descriptor lookup and waited the existing
cleanup tick. That actual replay still failed, so it does not establish a fix.
CARLA0.9.15's [Actor::Destroy implementation](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/Actor.cpp)
can return false without a destroy RPC when its cached actor state is invalid.
A retained descriptor is therefore insufficient to decide whether another
server-side deletion is needed. The current hypothesis is stale actor-handle
state; the precise live failure cause remains subject to the replay below.

The new cleanup check uses the snapshot after the one already scheduled cleanup
tick. Its frame must match that tick exactly. A false-destroy actor is retired
only if absent from that snapshot. A surviving actor, snapshot RPC error or
frame mismatch still fails. No additional tick, foreign actor deletion or
control/physics setting is introduced. Synthetic tests cover stale descriptors,
retirement, surviving actors, RPC failure and a wrong snapshot frame.

Recovery also authenticates pure evaluation without inventing a training
heartbeat: zero counters, empty optimization/training traces and exact saved
model, normalizer, RNG and configuration are required. All prior partial
evaluation costs are retained. The registered teacher NPZ is preserved by hash.

The actual snapshot-based replay is
`runs/20261027T124542Z_native_evaluation_recovery`,all12conditions,same saved
policy,zero fitting/PPO. Its inherited input cost is24,325controls,including
both interrupted attempts. Its full outcome is pending. This is a harness
diagnostic,not policy improvement,another independent seed or final confirmation.
