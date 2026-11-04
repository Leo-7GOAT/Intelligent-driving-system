import json,pickle,random,time
from pathlib import Path
import numpy as np
import torch
import pytest
from stable_baselines3 import PPO
from navstack.config import EnvConfig,PlannerConfig
from navstack.research import cumulative_train,preserve_global_rng


@pytest.mark.parametrize("use_sde", [False, True])
def test_save_verification_preserves_rng_and_flat_resume_matches_uninterrupted(tmp_path, use_sde):
    config=EnvConfig(method="B1",scenario="empty",deadline=2,planner=PlannerConfig(max_seconds=None))
    options={"use_sde":True,"sde_sample_freq":10,"log_std_init":-2,"forward_mean_bias":.2,"entropy":.001,"target_kl":.03} if use_sde else {}
    whole=cumulative_train(tmp_path/"whole",config,19,256,time.time()+90,rollout_physics=64,epochs=2,ppo_options=options)
    with (Path(whole["checkpoint"])/"resume.pkl").open("rb") as stream:snapshot=pickle.load(stream)
    assert random.getstate()==snapshot["random"]
    assert np.array_equal(np.random.get_state()[1],snapshot["numpy_random"][1])
    assert torch.equal(torch.get_rng_state(),snapshot["torch_random"])
    cumulative_train(tmp_path/"split",config,19,128,time.time()+90,rollout_physics=64,epochs=2,ppo_options=options)
    resumed=cumulative_train(tmp_path/"split",config,19,256,time.time()+90,rollout_physics=64,epochs=2,ppo_options=options)
    with preserve_global_rng():
        a=PPO.load(Path(whole["checkpoint"])/"model.zip",device="cpu")
        b=PPO.load(Path(resumed["checkpoint"])/"model.zip",device="cpu")
    assert whole["physics_steps"]==resumed["physics_steps"]==256
    assert whole["optimizer_steps"]==resumed["optimizer_steps"]
    assert all(torch.equal(value,b.policy.state_dict()[key]) for key,value in a.policy.state_dict().items())
