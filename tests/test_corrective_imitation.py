from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO

from navstack.corrective_imitation import fit_corrective_actor
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy


class NoPlant(Env):
    observation_space=spaces.Box(-1.,1.,(4705,),dtype=np.float32)
    action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
    def reset(self,**kwargs):raise AssertionError('Offline correction must never execute a plant')


def setup_model():
    torch.set_num_threads(1)
    model=PPO(OrderedFlatRoadCandidatePolicy,NoPlant(),n_steps=8,batch_size=8,seed=11,device='cpu')
    norm=SimpleNamespace(obs_rms=SimpleNamespace(mean=np.zeros(4705),var=np.ones(4705),count=10.),
                         normalize_obs=lambda x:x)
    return model,norm


def test_correction_changes_actor_only_without_plant_PPO_or_normalizer_updates():
    model,norm=setup_model();before={k:v.clone() for k,v in model.policy.state_dict().items()}
    report=fit_corrective_actor(model,norm,np.zeros((8,4705),dtype=np.float32),
        np.full((8,2),.25,dtype=np.float32),np.array([1.]*7+[20.]),steps=3)
    assert report['additional_supervised_steps']==3 and report['PPO_updates']==0
    assert report['critic_exact'] and report['normalization_statistics_exact']
    assert any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items() if 'policy_net' in k or 'action_net' in k)
    assert not model.policy.optimizer.state


@pytest.mark.parametrize('bad',['action','weight','PPO_state'])
def test_invalid_correction_rejected_before_fitting(bad):
    model,norm=setup_model();x=np.zeros((4,4705),dtype=np.float32);y=np.zeros((4,2),dtype=np.float32);w=np.ones(4)
    if bad=='action':y[0,0]=1.1
    if bad=='weight':w[0]=0
    if bad=='PPO_state':model.num_timesteps=1
    with pytest.raises(ValueError):fit_corrective_actor(model,norm,x,y,w,steps=2)
