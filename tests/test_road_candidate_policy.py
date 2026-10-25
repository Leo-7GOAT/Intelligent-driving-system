import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from navstack.config import EnvConfig,PlannerConfig,VehicleConfig
from navstack.env import NavigationEnv
from navstack.experiments import actor_vector,signature
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant,VehicleState
from navstack.world import World
from navstack.road_candidate_policy import (RoadCandidateVecNormalize,
    RoadCandidateActorCriticPolicy,FlatRoadCandidateActorCriticPolicy)
from navstack.independent_clip_ppo import IndependentClipPPO,load_ppo
from scripts.native_road_learning import training_config


class RoadPlantEnv(NavigationEnv):
    """Real lightweight integration on a synthetic road; never CARLA evidence."""
    def _create_episode(self, options=None):
        grid=RoadGrid(sample_road_grid({'width':34.,'height':24.,'geometry_id':'test'},
                                      lambda x,y:2<y<22))
        plant=VehiclePlant(self.config.vehicle,VehicleState(4,12))
        # Fabricated adapter feedback tests input plumbing only, never CARLA dynamics.
        plant.evidence={'actuator_observation':'ff_state_v1','speed_setpoint':.8,
            'actuation_state':{'signed_pedal':.2,'filtered_acceleration':.1,'feedback_integral':.03}}
        return World(34.,24.,road=grid),plant,(27,12,0),'test_road'


def config(method='H',feedback='enabled',actuator='off'):
    return EnvConfig(method=method,flat_hold=10,deadline=2,road_observation=True,
        candidate_observation=True,execution_feedback=feedback,
        actuator_observation=actuator,
        planner=PlannerConfig(max_seconds=None,max_nodes=64,analytic_expansion=True,vectorized_primitives=True))


def test_shared_native_road_candidate_input_and_exact_ablation():
    values=[]
    for method in ('H','B1','B2','B3'):
        env=RoadPlantEnv(config(method));obs,_=env.reset(seed=11)
        assert obs.shape==(4701,) and signature(env.config)['observation_shape']==[4701]
        values.append(obs);env.close()
    for obs in values[1:]:np.testing.assert_array_equal(values[0],obs)
    env=RoadPlantEnv(config(feedback='ablated'));ablated,_=env.reset(seed=11);env.close()
    np.testing.assert_array_equal(ablated[:351],values[0][:351])
    np.testing.assert_array_equal(ablated[605:],values[0][605:])
    np.testing.assert_array_equal(ablated[351:599].reshape(31,8)[:,:5],values[0][351:599].reshape(31,8)[:,:5])
    assert not ablated[351:599].reshape(31,8)[:,5:].any() and not ablated[599:605].any()
    protocol={'feedback_ablation':True,'episode_seconds':45,'road_geometry_version':'carla_driving_indexed_union_64_v3'}
    old=training_config(protocol,'H',VehicleConfig())
    new=training_config({**protocol,'road_candidate_policy':True},'H',VehicleConfig())
    assert {**old.to_dict(),'candidate_observation':True}==new.to_dict()
    assert training_config({**protocol,'road_candidate_policy':True},'H0',VehicleConfig()).execution_feedback=='ablated'


@pytest.mark.parametrize('method,policy',[('H',RoadCandidateActorCriticPolicy),('B1',FlatRoadCandidateActorCriticPolicy)])
@pytest.mark.parametrize('algorithm',[PPO,IndependentClipPPO])
@pytest.mark.parametrize('actuator',['off','ff_state_v1'])
def test_road_policy_geometry_invariance_real_updates_and_reload(tmp_path,method,policy,algorithm,actuator):
    torch.set_num_threads(1)
    vec=RoadCandidateVecNormalize(DummyVecEnv([lambda:RoadPlantEnv(config(method,actuator=actuator))]),norm_reward=False)
    try:
        obs=vec.reset();raw=vec.get_original_obs()
        for region in (slice(351,599),slice(605,4701)):
            np.testing.assert_array_equal(obs[:,region],raw[:,region])
            np.testing.assert_array_equal(vec.unnormalize_obs(obs)[:,region],raw[:,region])
        model=algorithm(policy,vec,n_steps=8,batch_size=8,n_epochs=2,seed=11,device='cpu')
        permutation=np.random.default_rng(3).permutation(31)
        changed=obs.copy()
        changed[:,351:599]=obs[:,351:599].reshape(-1,31,8)[:,permutation].reshape(-1,248)
        with torch.no_grad():
            a=model.policy.get_distribution(torch.as_tensor(obs)).distribution
            b=model.policy.get_distribution(torch.as_tensor(changed)).distribution
            if method=='H':
                torch.testing.assert_close(b.probs,a.probs[:,permutation]);assert (a.probs>0).all()
            else:
                torch.testing.assert_close(b.mean,a.mean);torch.testing.assert_close(b.stddev,a.stddev)
            torch.testing.assert_close(model.policy.predict_values(torch.as_tensor(obs)),model.policy.predict_values(torch.as_tensor(changed)))
        tensor=torch.as_tensor(obs).requires_grad_(True)
        distribution=model.policy.get_distribution(tensor).distribution
        score=distribution.logits[:,4].sum() if method=='H' else distribution.mean.sum()
        score.backward()
        assert tensor.grad[:,605:].abs().sum()>0 and tensor.grad[:,351:599].abs().sum()>0
        if actuator != 'off':
            assert obs.shape[-1] == 4705 and tensor.grad[:,4701:].abs().sum()>0
            assert not np.array_equal(obs[:,4701:],raw[:,4701:])
            np.testing.assert_allclose(vec.unnormalize_obs(obs)[:,4701:],raw[:,4701:],atol=1e-6)
        before={k:v.clone() for k,v in model.policy.state_dict().items()}
        actor_before=actor_vector(model).clone()
        model.learn(16)
        assert model._n_updates==4 and torch.linalg.vector_norm(actor_vector(model)-actor_before)>0
        assert any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items() if 'policy_net_road' in k)
        model.save(tmp_path/'model.zip');vec.save(tmp_path/'normalization.pkl')
        loaded=load_ppo(tmp_path/'model.zip',device='cpu')
        assert type(loaded) is algorithm
        assert isinstance(loaded.policy,policy)
        assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
        restored=VecNormalize.load(tmp_path/'normalization.pkl',DummyVecEnv([lambda:RoadPlantEnv(config(method,actuator=actuator))]))
        try:
            assert isinstance(restored,RoadCandidateVecNormalize)
            np.testing.assert_array_equal(restored.normalize_obs(raw)[:,605:4701],raw[:,605:4701])
            np.testing.assert_array_equal(restored.normalize_obs(raw),vec.normalize_obs(raw))
        finally:restored.close()
    finally:vec.close()
