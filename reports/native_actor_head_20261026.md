# Matched native H decision-head experiment — 2026-10-26

Status: both actual CARLA arms and raw audits are complete: shared1/4 versus
unshared3/4. The
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

## Completed endpoints

| Arm | Training controls | Evaluation controls | Successes | Parameters | Training seconds |
|---|---:|---:|---:|---:|---:|
| H_shared | 4,145 | 3,017 | 1/4 | 208,034 | 237.061 |
| H_unshared_v1 | 4,128 | 1,969 | 3/4 | 238,784 | 301.323 |

Each performs416sampled decisions/130Adam steps. All13,259controls match closed
original CARLA streams; all500frozen decisions (302/198) reproduce exactly
from measured feedback with frozen normalizers and zero new driving/updates.
All four paired initial states match exactly in x/y/yaw/speed/steering/
acceleration. Full metrics/hashes: [summary](native_actor_head_summary_20261026.json).

Shared succeeds only on road2; unshared succeeds on roads0/2/3. Unshared road1
has90failed plans/900intervened controls, zero progress and full-cycle
P95/P99/max7.998/8.589/8.951s; every cycle exceeds0.5s. Road3includes one failed
plan/10intervened controls despite eventual success. This is not realtime.

Read-only reconstruction of the first road1decision finds shared action6,
witness[1,1,0], with an actual local reference; unshared action8 has
witness[1,0,0] and exhausts7200nodes. No moving action has a positive sufficient
goal-continuation witness there. Missing sufficient evidence is not proof of
infeasibility: filtering by the goal witness would remove useful actions.
No mask or modified-policy execution was performed in this diagnostic.

Worker/guard exit0, reasonnull; matching empty-world probe and
owned_empty_server_stopped verify cleanup. The earlier live paragraph is
historical. This single-seed result supports further development, not formal
hierarchy, feedback, PPO-value or unseen-generalization claims.

At sourcee0f9099, independent Windows CPU regression passes304/1skip/4CARLA
deselections in168.97s. [Actual hosted CI](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40281946359)
passes302/3skips/4deselections in145.38s. CPU verification is separate from the
actual CARLA outcomes. All original research gates remain unchanged.

## Actual publication and independent public reproduction

The [complete model/raw-evidence asset](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_actor_head_seed11_20261026_full.zip)
is uploaded,402files/16,826,597bytes, SHA256
`8e4dac0babed85e9365158ef75200634968acfe24cbee58c3ad91f7095a37170`.
Export sourcef38f38a; historical prerelease tag unchanged. An independent
checkout actually downloads it publicly and verifies all402filehashes,
both raw audits/13,259controls,8initial model actions/frozen normalization,
and a byte-identical eight-case plot, with zero additional driving or updates.
[Receipt](native_actor_head_public_reproduction_20261026.json). All outcomes
and six auxiliary path-redaction receipts are retained; protected bytes remain
unchanged. Publication closes evidence delivery, not research acceptance.
