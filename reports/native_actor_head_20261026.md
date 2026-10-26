# Matched native H decision-head experiment — 2026-10-26

Status: actual CARLA training started, not yet a completed comparison. The
original research objective, healthy-B1 requirement, independent seeds and
fresh final conditions remain unchanged.

## Specific architectural hypothesis

The original shared scoring network receives a separate candidate embedding
for each action. If those embeddings are identical, all logits are identical
and their shared dependence on global state cancels in softmax. The policy
still has nonlinear state/candidate interactions when embeddings differ;
this is **not** proof that all global-state gradients vanish in real episodes
or that this architecture explains every waiting failure.

The optional `UnsharedRoadCandidatePolicy` retains the exact existing road,
state and candidate encoders and value network. The head is precisely `Linear(31*32,31)`,30,783parameters versus33in the shared
head, an increase of30,750. This directly supplies independent logits from all
candidate embeddings. It sacrifices permutation equivariance, so fixed catalog
slot semantics are disclosed. There is no teacher, action mask, rule selector,
new candidate generator, goal-snapping state update or planner/controller change.

Construction first creates the original network, then replaces only its head
inside a saved/restored CPU RNG context and rebuilds the fresh optimizer.
Tests prove identical seeded encoder/critic tensors and post-construction RNG
between the two arms, genuine common-state sensitivity with identical candidate
embeddings, actual synthetic PPO updates and exact save/reload. Fourteen
targeted policy/protocol/preparation tests pass in3.80s. Synthetic gradients and
updates are plumbing/capacity evidence, not a driving improvement.

## Registered actual comparison

Batch `runs/20261026T064621Z_native_actor_head`; protocol SHA256
`d0e5e4a67208a0f731e5524cc2a10348c3c2e7e08297ccb123f6d0fd6a1813df`.
Code commit `e0f9099`; exact executed sources are individually hashed inside
the frozen batch. Both arms start fresh at seed11 with no inherited weights or
normalization, on all four ORIGINAL full development roads (not6m prefixes).

`H_shared` and `H_unshared_v1` use the same original task reward,45s episodes,
4705inputs,32-decision rollouts, batch32,10epochs,global clipping and Adam,
4,096requested training controls per arm and all four deterministic endpoint
probes. Complete-rollout overshoot is charged. Both use `batch_sat_v1` from
the start, the same7200node/Dubins planner, trajectory/MPC and calibrated
actuator. These common settings differ from the original20k scalar/two-epoch
parent; that historical parent is not presented as the matched control.

One1,800s wall window, existing disk/RAM/GPU/no-progress guards, zero retries,
all failures retained. Only a complete audited pair supports comparison.
Network capacity and full-cycle costs must be reported with outcomes; one
seed/four reused roads cannot establish feedback benefit or generalization.

At14:50local, the original worker64952(created14:48:15.869366+08:00) is still
training the shared arm; its guard is healthy. OS process/creation identity,
heartbeats and retained handles remain authoritative over this timestamped
snapshot. Do not start another CARLA tick owner or silently repeat a partial
batch. Record both endpoints, all raw audits and normal cleanup after they occur.
