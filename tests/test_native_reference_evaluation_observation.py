import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import EnvConfig
from navstack.reference_context_policy import ReferenceContextPolicy,ReferenceContextVecNormalize
from navstack.road_candidate_policy import EndpointClearRoadPolicy,RoadCandidateVecNormalize
from scripts import native_road_learning as learning
from test_route_catalog_environment import RouteFixture,config
from test_actuator_observation import evidence


class ObservationFixture(RouteFixture):
    def _create_episode(self,options=None):
        world,plant,goal,name=super()._create_episode(options)
        plant.evidence=evidence()
        return world,plant,goal,name


@pytest.mark.parametrize('recorded',[False,True])
@pytest.mark.parametrize('context',[False,True])
@pytest.mark.parametrize('actuator',['off','ff_state_v1'])
def test_evaluation_factory_observation_can_be_consumed_by_the_registered_policy(monkeypatch,recorded,context,actuator):
    torch.set_num_threads(1)
    monkeypatch.setattr(learning,'RecordedRoadEnv',ObservationFixture)
    monkeypatch.setattr(learning,'CarlaNavigationEnv',ObservationFixture)
    cfg=EnvConfig.from_dict({**config(feedback='continuation').to_dict(),
                            'route_goal_candidate':True,'reference_reuse':True,'actuator_observation':actuator})
    cls=learning.native_environment_class(context,recorded)
    raw=DummyVecEnv([lambda:cls(cfg)])
    norm=(ReferenceContextVecNormalize if context else RoadCandidateVecNormalize)(raw,norm_reward=False)
    model=PPO(ReferenceContextPolicy if context else EndpointClearRoadPolicy,norm,
              n_steps=2,batch_size=2,n_epochs=1,seed=11,device='cpu')
    try:
        obs=norm.reset()
        assert obs.shape==(1,4701+(4 if actuator!='off' else 0)+(31 if context else 0))
        action,_=model.predict(obs,deterministic=True)
        assert action.shape==(1,)
    finally:norm.close()
