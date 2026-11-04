import numpy as np
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO

from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy,OrderedFlatRoadCandidatePolicy


class NoExecution(Env):
    observation_space=spaces.Box(-1.,1.,(4705,),dtype=np.float32)
    action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
    def reset(self,**kwargs):raise AssertionError('Architecture probe must not execute a plant')


def test_ordered_actor_preserves_original_encoder_critic_head_and_rng_except_fusion(tmp_path):
    torch.set_num_threads(1)
    old=PPO(FlatRoadCandidateActorCriticPolicy,NoExecution(),n_steps=8,batch_size=8,seed=11,device='cpu')
    expected_rng=torch.get_rng_state().clone()
    new=PPO(OrderedFlatRoadCandidatePolicy,NoExecution(),n_steps=8,batch_size=8,seed=11,device='cpu')
    assert torch.equal(torch.get_rng_state(),expected_rng)
    old_parameters=old.policy.state_dict()
    for name,value in new.policy.state_dict().items():
        if not name.startswith('mlp_extractor.policy_net_fusion.'):
            assert torch.equal(value,old_parameters[name]),name
    assert not new.policy.optimizer.state and new.num_timesteps==new._n_updates==0
    assert {id(p) for g in new.policy.optimizer.param_groups for p in g['params']}=={id(p) for p in new.policy.parameters()}
    assert sum(p.numel() for p in new.policy.parameters())>sum(p.numel() for p in old.policy.parameters())
    x=np.zeros((2,4705),dtype=np.float32)
    candidates=x[:,351:599].reshape(2,31,8)
    candidates[0,0]=.25;candidates[0,1]=-.5;candidates[1]=candidates[0];candidates[1,[0,1]]=candidates[1,[1,0]]
    with torch.no_grad():
        pooled=old.policy.mlp_extractor.forward_actor(torch.as_tensor(x))
        ordered=new.policy.mlp_extractor.forward_actor(torch.as_tensor(x))
    torch.testing.assert_close(pooled[0],pooled[1],rtol=0,atol=1e-7)
    assert not torch.allclose(ordered[0],ordered[1],atol=1e-6,rtol=0)
    expected=new.predict(x,deterministic=True)[0]
    new.save(tmp_path/'ordered.zip');loaded=PPO.load(tmp_path/'ordered.zip',device='cpu')
    assert isinstance(loaded.policy,OrderedFlatRoadCandidatePolicy)
    np.testing.assert_array_equal(loaded.predict(x,deterministic=True)[0],expected)
