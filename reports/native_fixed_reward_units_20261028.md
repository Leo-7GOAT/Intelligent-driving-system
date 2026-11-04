# Fixed PPO reward units development profile

The completed faster-critic B1 result is **4/12with seven unsafe episodes**,
compared with4/12and three unsafe for KL control140111. This is a negative
result,not a healthy baseline. [Endpoint and raw audit](native_B1_critic_rate_endpoint_20261028.json)
bind20480real training controls,4128evaluation controls and281actual Adam
updates. The original562post-hook counter is retained and its double dispatch
is explicitly corrected using all saved per-parameter Adam states.

The faster critic's absolute output bound grows to14.4825but remains below the
raw unsafe terminal target -90. The next single-variable profile uses fixed
learner reward units: every PPO reward is multiplied by0.05,so that this target
becomes -4.5. There is no reward clipping or adaptive return normalization.
Physical environment rewards,reward components,success rules,unsafe rules and
evaluation metrics retain their original units. Positive constant scaling
preserves the underlying discounted reward objective; its effect on finite PPO
optimization and driving performance must be measured.

[Initialization equality receipt](native_B1_fixed_reward_units_initialization_20261028.json)
binds batch020728to standard-Adam KL control140111. Network tensors,optimizer
class,actor and critic rate3e-5,configuration,RNG,observation RMS,normalized
observations,rollout320,epochs10,KL.01,teacher data and physical runtime source
are identical. It uses the original12conditions and20480controls/7200s budget.
This profile is separate from both the faster critic and the36/24partition.
Initial critic tensors are untouched and have never undergone a PPO update.

The custom road normalizer applies the constant only to rewards entering PPO.
Its original reward accessor and environment info remain in physical units.
The new class is saved with its frozen implementation and restored through the
existing SB3 checkpoint loader. Observation statistics do not change. Five new
tests cover attached/detached serialization,frozen RMS,raw reward preservation,
actual SB3 rollout rewards and rejection of adaptive/unregistered scaling.
The30related tests pass6.85s. This is CPU verification,not CARLA learning.

First preparation020529failed while serializing a detached SB3normalizer before
registration or execution;its partial child and preparation failure receipt
remain. It has zero physical controls/PPO updates and is never activated. The
replacement020728is actually PREPARED,passes ValidateOnly369source files,and is
ACTUALLY QUEUED after train36/validation24batch014801,queue41236. The existing
sequential cleanup,resource limits,wallcap and no-automatic-retry rules remain.

Current actual order:shared-mask seed11PPO013316is training,then seed22PPO013440,
then new train36/validation24seed11PPO014801,then fixed reward-unit B1020728.
These are local queued programs. Scores for unfinished phases are unknown.
No research acceptance gate or final hypothesis has been changed.
