import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from navstack.config import VehicleConfig
from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
from navstack.experiments import signature
from scripts.native_road_learning import training_config,learning_schedule
from test_discrete_flat_policy import Probe


def test_cadence_matches_physical_rollout_discount_batch_and_seeded_initialization():
    torch.set_num_threads(1);settings=[];states=[];rngs=[];configs=[]
    for hold in (10,2):
        protocol={'episode_seconds':12.,'flat_hold':hold,'flat_longitudinal_mode':'speed_target_v1',
                  'flat_action_catalog':'speed_steer_3x7_v1','ppo_epochs':10,'ppo_batch_size':320//hold}
        config=training_config(protocol,'B1',VehicleConfig());configs.append(config)
        schedule=learning_schedule(protocol,config,320//hold);settings.append(schedule)
        model=PPO(DiscreteFlatRoadPolicy,Probe(True),**schedule,seed=11)
        states.append({k:v.clone() for k,v in model.policy.state_dict().items()});rngs.append(torch.get_rng_state().clone())
        assert schedule['n_steps']*config.substeps==320
        assert schedule['batch_size']==schedule['n_steps'] and schedule['n_epochs']==10
    assert signature(configs[0])!=signature(configs[1])
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items()) and torch.equal(*rngs)
    for name in ('gamma','gae_lambda'):
        assert settings[0][name]**2==pytest.approx(settings[1][name]**10,abs=1e-14)
    assert settings[0]['gae_lambda']==.95**.5
    with pytest.raises(ValueError):learning_schedule({'ppo_epochs':10,'ppo_batch_size':31},configs[0],32)
