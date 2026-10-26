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

The owned server launch has begun. Actual completion, all outcomes, interaction
accounting, audits and cleanup must be recorded from the batch; none is inferred
from this prepared protocol. A partial batch cannot be reported as a completed
paired result. Original training failures and the separate interrupted weight
contrast remain preserved.
