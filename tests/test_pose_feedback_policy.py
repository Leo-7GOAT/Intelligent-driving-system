from copy import deepcopy
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from navstack.independent_clip_ppo import actor_critic_parameters
from navstack.pose_feedback_policy import promote,PoseFeedbackFlatPolicy,pose_features
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy


def test_pose_residual_preserves_base_and_optimizer_and_reloads_for_ppo(tmp_path):
    torch.set_num_threads(1)
    model=PPO(OrderedFlatRoadCandidatePolicy,None,_init_setup_model=False)
    model.observation_space=spaces.Box(-np.inf,np.inf,shape=(4705,),dtype=np.float32)
    model.action_space=spaces.Box(-1,1,shape=(2,),dtype=np.float32)
    model.n_envs=1
    model._setup_model()
    policy=model.policy
    x=torch.randn(3,4705)
    with torch.no_grad():
        before=policy.get_distribution(x).distribution.mean.clone();value=policy.predict_values(x).clone()
    old={k:v.clone() for k,v in policy.state_dict().items()}
    # Nonempty optimizer state must survive promotion and serialization.
    loss=policy.get_distribution(x).distribution.mean.square().sum()+policy.predict_values(x).square().sum()
    loss.backward();policy.optimizer.step();policy.optimizer.zero_grad()
    old={k:v.clone() for k,v in policy.state_dict().items()}
    state=deepcopy(policy.optimizer.state_dict())
    with torch.no_grad():before=policy.get_distribution(x).distribution.mean.clone();value=policy.predict_values(x).clone()
    promote(model)
    assert len(policy.optimizer.param_groups)==2
    for k,v in old.items():assert torch.equal(v,policy.state_dict()[k])
    assert policy.optimizer.state_dict()['param_groups'][0]==state['param_groups'][0]
    for key,values in state['state'].items():
        for name,v in values.items():assert torch.equal(v,policy.optimizer.state_dict()['state'][key][name])
    with torch.no_grad():
        assert torch.equal(before,policy.get_distribution(x).distribution.mean)
        assert torch.equal(value,policy.predict_values(x))
        policy.action_net.policy_net_pose[0]=.1
        after=policy.get_distribution(x).distribution.mean
        torch.testing.assert_close(after[:,1],before[:,1]+.1)
        assert torch.equal(after[:,0],before[:,0]);assert torch.equal(value,policy.predict_values(x))
    model.save(tmp_path/'policy.zip');reloaded=PPO.load(tmp_path/'policy.zip',device='cpu')
    assert type(reloaded.policy) is PoseFeedbackFlatPolicy
    with torch.no_grad():assert torch.equal(after,reloaded.policy.get_distribution(x).distribution.mean)
    actor,critic=actor_critic_parameters(reloaded.policy)
    assert any(p is reloaded.policy.action_net.policy_net_pose for p in actor)
    distribution=reloaded.policy.get_distribution(x)
    actions=distribution.sample();values,logprob,entropy=reloaded.policy.evaluate_actions(x,actions)
    reloaded.policy.optimizer.zero_grad();(-logprob.mean()+values.square().mean()).backward()
    assert reloaded.policy.action_net.policy_net_pose.grad is not None
    assert torch.isfinite(reloaded.policy.action_net.policy_net_pose.grad).all()
    reloaded.policy.optimizer.step()


def test_pose_features_use_only_declared_shared_geometry():
    x=torch.arange(2*4705,dtype=torch.float32).reshape(2,4705)
    y=pose_features(x)
    assert y.shape==(2,10);assert torch.equal(y[:,0],torch.ones(2))
    expected=torch.cat([x[:,351+i*8:354+i*8] for i in (5,15,25)],dim=1)
    assert torch.equal(y[:,1:],expected)
