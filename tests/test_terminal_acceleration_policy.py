from copy import deepcopy
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy
from navstack.terminal_acceleration_policy import promote,TerminalAccelerationFlatPolicy


def test_terminal_head_preserves_steering_critic_and_old_adam_through_training_and_reload(tmp_path):
    torch.set_num_threads(1)
    model=PPO(OrderedFlatRoadCandidatePolicy,None,_init_setup_model=False)
    model.observation_space=spaces.Box(-np.inf,np.inf,(4705,),dtype=np.float32)
    model.action_space=spaces.Box(-1,1,(2,),dtype=np.float32);model.n_envs=1;model._setup_model()
    x=torch.randn(4,4705);policy=model.policy
    (policy.get_distribution(x).distribution.mean.square().sum()+policy.predict_values(x).square().sum()).backward()
    policy.optimizer.step();policy.optimizer.zero_grad();old=deepcopy(policy.optimizer.state_dict())
    with torch.no_grad():means=policy.get_distribution(x).distribution.mean.clone();values=policy.predict_values(x).clone()
    promote(model)
    with torch.no_grad():assert torch.equal(means,policy.get_distribution(x).distribution.mean)
    head=policy.action_net.policy_net_terminal
    assert sum(p.numel() for p in head.parameters())==177
    opt=torch.optim.Adam(head.parameters(),lr=.01)
    for _ in range(3):
        loss=(head(x[:,[0,16,17,20,3,4701,4702,4703,4704]])+1).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        after=policy.get_distribution(x).distribution.mean
        assert not torch.equal(after[:,0],means[:,0]);assert torch.equal(after[:,1],means[:,1])
        assert torch.equal(policy.predict_values(x),values)
    state=policy.optimizer.state_dict()
    assert state['param_groups'][0]==old['param_groups'][0]
    for key,record in old['state'].items():
        for field,value in record.items():assert torch.equal(value,state['state'][key][field])
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert type(loaded.policy) is TerminalAccelerationFlatPolicy
    with torch.no_grad():assert torch.equal(after,loaded.policy.get_distribution(x).distribution.mean)
    actions=loaded.policy.get_distribution(x).sample();values,lp,entropy=loaded.policy.evaluate_actions(x,actions)
    loaded.policy.optimizer.zero_grad();(-lp.mean()+values.square().mean()).backward();loaded.policy.optimizer.step()
    assert loaded.policy.action_net.policy_net_terminal[-1].weight.grad is not None
