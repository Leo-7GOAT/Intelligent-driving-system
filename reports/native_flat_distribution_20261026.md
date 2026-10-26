# B1 bounded sampled-action sanity contrast — 2026-10-26

This is a new development experiment, not a change to the research goal or a
claim that action clipping is the established cause of failure. The full
five-seed, fresh-condition, H/H0/B1/B2/B3 acceptance gates remain open.

## Evidence and hypothesis

The completed fresh B1 parent has 2,080 sampled training decisions. In
1,107 (53.22%), at least one Gaussian component exceeds the existing actuator
range [-1,1]; component counts are650acceleration/665steering. Maximum absolute
sampled values are3.90605/3.70345. These counts come from its closed, raw-audited
`transitions_*.jsonl.gz` streams, with no new execution. This is a many-to-one
actuator mapping, not proof that the Gaussian policy likelihood is wrong.

The new `SquashedFlatRoadPolicy` keeps the same road/candidate encoder, critic,
initial tensors and Gaussian latent parameters, but samples tanh(Gaussian).
SB3's existing squashed distribution supplies inverse tanh and the Jacobian
correction during PPO action evaluation. It is not a bounded mean with
unbounded samples, an added planner, teacher, rule selector or safety wrapper.

Hypothesis: the bounded sampled distribution may improve basic direct-control
learning by avoiding actuator clipping redundancy. A negative result remains
negative; neither gradient plumbing nor a bounded command proves driving skill.

## Registered protocol

Actual batch: `runs/20261026T060439Z_native_flat_distribution`, protocol SHA256
`b6fff37e77548ce80781e757a1211fcc7561db4e1972da264a920b89c7effd87`.

- Fresh seed11 B1 clipped-Gaussian and B1 tanh-Gaussian arms; no inherited
  models, statistics, teacher labels or optimizer states.
- Same four measured native-road prefixes of approximately6m,12s episode limit,
  same original reward,4705inputs, direct acceleration/steering,0.5s action hold.
- Same32-decision PPO rollouts, batch32,10epochs, Adam/global clipping and
  updating observation normalization. More epochs than the original parent
  apply equally to both arms, not solely to the proposed distribution.
- Both use the same `batch_sat_v1` low-level version, from the start. This is
  not a direct performance comparison with the original scalar-trained parent.
- Each arm has4,096requested training controls; complete-rollout overshoot is
  recorded and charged. All four frozen deterministic short-route probes are
  mandatory. No endpoint or case selection by outcome.
- One1,800s wall window, existing12GiB disk/4GiB RAM/85C GPU/300s no-progress
  guards and zero automatic retries. Success requires4/4safe short-task probes;
  it does not close full-route baseline competence or unseen/generalization gates.

Preparation initially copied parent route-length labels into the prefix
metadata. That first batch (`20261026T060324Z_native_flat_distribution`) was
aborted before any execution; its protocol and registration are preserved.
The replacement above stores actual prefix lengths/headings and parent labels
separately. This is not a discarded driving failure or an automatic rerun.

## Verification and execution status

The initial targeted suite passes18tests in60.86s, including corrected density,
identical initial tensors, bounded sampled/received actions, actual synthetic
PPO updates, exact model reload, protocol isolation, route-prefix preservation
and existing road-policy/rollout preparation regressions. Synthetic probe
updates are not CARLA training or performance evidence.

Both arms completed normally, each with416sampled decisions,13optimization
phases and130Adam steps. The clipped arm uses4,150training controls/64.179s;
the tanh arm4,154controls/61.455s. Each evaluates960controls over all4cases.
Both score **0/4**, all deadlines, zero deterministic progress. Each has18
completed training episodes:15deadlines/3bounds. Training mean progress is
2.6866/2.4022m, maximum5.5923/5.3413m; these are not successful parking.
The bounded distribution alone did not repair basic learning in this budget;
it is not adopted as the final baseline and no larger repeat is justified.

All **10,224**actual training/evaluation controls match closed original CARLA
streams (5,110/5,114per arm). Per-method raw audits are in each arm's
`endpoint_raw_audit.json`. Reset ticks and rollout overshoot remain charged.
Worker12624and the replacement guard complete with exit0/reasonnull; the
owned empty server is stopped and ports are clear. The initial guard refused
relative --batch spelling before adoption; a tested explicit-argument resolver
allowed a replacement guard to adopt the SAME continuously running worker.
This guard-only override has a separate hash receipt; frozen training/low-level
source and the original rejected-guard log remain unchanged. No worker restart.

The local full CPU suite before the later guard/reward changes passes299tests,
1skip/4CARLA deselections in192.18s. The guard correction passes4targeted tests;
the subsequent combined policy/guard/shared-reward selection passes15tests in
4.65s. These scopes are retained separately rather than claiming an unexecuted
full-suite result for newer code.

## Next bounded hypothesis: explicit physical progress reward

The failed arms did move stochastically but had no successful training episode,
then chose no motion deterministically. Existing terminal-zero potential-based
shaping supplies intermediate distance differences whose discounted sum
cancels on terminal failure; it does not retain a failed episode's progress as
an independent task benefit. This motivates testing a disclosed base-task
reward change, not a claim that PBRS or PPO is incorrectly implemented.

`runs/20261026T061501Z_native_progress_sanity` starts ONE fresh clipped-Gaussian
B1 against the completed clipped control above. Only task progress weight
changes0to1; same seed11,32-action/10epoch schedule,4,096requested controls,
6m routes,12s deadline,observations,normalization and batchSAT low level.
The new term is signed measured Euclidean remaining-distance reduction each
physical tick, without a planner target. It is available equally to H/H0/B1;
this single B1 diagnostic cannot support a hierarchy comparison. Protocol SHA256
`ede5a2195f0860b3f9861f94797b5cad46d0b5b0847e593fe7bd0f0f80df8fee`.
One900s window, all4endpoint probes, same guards and zero retries. At this
update it is actually running; record observed outcomes before any expansion.
Neither a revised reward nor completion of this short test closes the original
full-task baseline-health, five-seed or independent-confirmation requirements.
