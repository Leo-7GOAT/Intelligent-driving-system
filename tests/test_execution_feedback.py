import numpy as np
from navstack.config import EnvConfig,PlannerConfig
from navstack.env import NavigationEnv


def test_feedback_information_is_common_and_ablation_keeps_network_shape():
    observations=[]
    for method in ("H","B1","B2","B3"):
        env=NavigationEnv(EnvConfig(method=method,scenario="empty",subgoal_mode="goal_frame",execution_feedback="enabled"))
        obs,_=env.reset(seed=7301)
        assert obs.shape == (450,) and env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        observations.append(obs)
    for obs in observations[1:]: np.testing.assert_array_equal(obs,observations[0])
    ablated=NavigationEnv(EnvConfig(scenario="empty",subgoal_mode="goal_frame",execution_feedback="ablated"))
    obs,_=ablated.reset(seed=7301)
    np.testing.assert_array_equal(obs[:351],observations[0][:351])
    assert not obs[351:].any()


def test_feedback_records_failed_choice_without_overriding_it():
    env=NavigationEnv(EnvConfig(scenario="empty",subgoal_mode="goal_frame",execution_feedback="enabled",
                               planner=PlannerConfig(max_seconds=None,analytic_expansion=True)))
    env.reset(seed=7301)
    target,_=env.decode_action(6)
    env.world.static=[(target[0]-1,target[1]-1,target[0]+1,target[1]+1)]
    _,_,_,_,info=env.step(6)
    assert info["mode"] == "plan_failed"
    assert env.execution_outcome[1] == 1
    assert info["executed_substeps"] == 10
    assert env.state.x == 4
    assert env.state.time > 0
