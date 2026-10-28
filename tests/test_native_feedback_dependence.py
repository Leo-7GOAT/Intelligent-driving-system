import numpy as np
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO
from navstack.road_candidate_policy import EndpointClearRoadPolicy
from scripts.audit_native_feedback_dependence import ablate,probabilities,VARIANTS


def test_feedback_ablation_preserves_state_geometry_road_actuator_and_input():
    raw=np.arange(4705,dtype=np.float32);before=raw.copy()
    indices=set(range(599,605))|{351+8*a+j for a in range(31) for j in (5,6,7)}
    changed=ablate(raw,'zero_all_feedback')
    assert np.array_equal(raw,before)
    assert all(changed[i]==0 if i in indices else changed[i]==raw[i] for i in range(4705))
    continuation=ablate(raw,'zero_continuation')
    assert set(np.flatnonzero(continuation!=raw))=={351+8*a+7 for a in range(31)}


def test_actual_endpoint_policy_fixed_mask_matches_distribution_and_unknown_feedback_control():
    class NoPlant(Env):
        observation_space=spaces.Box(-1.,1.,(4701,),dtype=np.float32)
        action_space=spaces.Discrete(31)
        def reset(self,**kwargs):raise AssertionError('No physical execution permitted')
    torch.set_num_threads(1);model=PPO(EndpointClearRoadPolicy,NoPlant(),n_steps=8,batch_size=8,seed=11,device='cpu')
    raw=np.zeros(4701,dtype=np.float32);raw[351+5]=1;raw[351+8+5]=1
    with torch.no_grad():
        obs=torch.as_tensor(raw[None]);latent=model.policy.mlp_extractor.forward_actor(model.policy.extract_features(obs));mask=latent[:,-31:]>.5
        assert int(mask.sum())==2
        torch.testing.assert_close(probabilities(model.policy,obs,mask),model.policy.get_distribution(obs).distribution.probs,rtol=0,atol=0)
        unknown=np.zeros(4701,dtype=np.float32);mask=torch.ones((1,31),dtype=torch.bool)
        for v in VARIANTS:
            torch.testing.assert_close(probabilities(model.policy,ablate(unknown,v)[None],mask),model.policy.get_distribution(torch.as_tensor(unknown[None])).distribution.probs,rtol=0,atol=0)
    assert not model.num_timesteps and not model.policy.optimizer.state
