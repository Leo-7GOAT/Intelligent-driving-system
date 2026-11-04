import numpy as np
import pytest
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


def test_curriculum_same_geometry_both_methods_and_full_validation():
    for count,expected in ((0,10),(50000,20),(150000,27)):
        for seed in range(10):
            h=NavigationEnv(EnvConfig(method="H",curriculum=True))
            b=NavigationEnv(EnvConfig(method="B1",curriculum=True))
            h.total_physics_steps=b.total_physics_steps=count
            ho,_=h.reset(seed=seed);bo,_=b.reset(seed=seed)
            np.testing.assert_array_equal(ho,bo)
            assert h.goal[0]==b.goal[0]==expected
            assert h.world.free(h.goal,h.config.vehicle)
    evaluation=NavigationEnv(EnvConfig(curriculum=True,split="validation"))
    evaluation.reset(seed=11)
    assert evaluation.goal[0]==27


def test_full_length_task_staging_is_common_and_never_applies_to_validation():
    names = set()
    for count in (0, 299999, 300000):
        for seed in range(12):
            observations = []
            for method in ("H", "B1", "B2", "B3"):
                env = NavigationEnv(EnvConfig(method=method, empty_pretraining_steps=300000, obstacle_num=0))
                env.total_physics_steps = count
                observation, _ = env.reset(seed=seed)
                assert env.goal[0] == 27
                assert env.config.scenario == "mixed"
                if count < 300000:
                    assert env.scenario_name == "empty" and not env.world.static and not env.world.dynamic
                else:
                    names.add(env.scenario_name)
                observations.append(observation)
                env.close()
            for obs in observations[1:]:
                np.testing.assert_array_equal(obs, observations[0])
    assert names == {"empty", "static", "narrow", "dynamic"}
    env = NavigationEnv(EnvConfig(empty_pretraining_steps=300000, scenario="static", split="validation", obstacle_num=0))
    env.reset(seed=11)
    assert env.scenario_name == "static" and env.world.static and env.goal[0] == 27
    env.close()


def test_fixed_distance_stages_share_geometry_use_physical_counters_and_leave_validation_full():
    stages = [{"until_physics_steps": 10, "route_m": 6.}, {"until_physics_steps": 20, "route_m": 12.}]
    for count, distance in ((0, 6.), (9, 6.), (10, 12.), (19, 12.), (20, 23.)):
        observations = []
        for method in ("H", "B1", "B2", "B3"):
            env = NavigationEnv(EnvConfig(method=method, route_stages=stages))
            env.total_physics_steps = count
            observation, _ = env.reset(seed=11)
            assert env.goal[0]-env.initial_scenario["initial"]["x"] == distance
            if count < 20:
                assert env.scenario_name == "empty" and not env.world.static and not env.world.dynamic
            observations.append(observation)
            env.close()
        for observation in observations[1:]:
            np.testing.assert_array_equal(observation, observations[0])
    env = NavigationEnv(EnvConfig(method="B1", flat_hold=10, route_stages=stages))
    env.reset(seed=11)
    initial_goal = env.goal
    env.step(np.array([0., 0.], dtype=np.float32))
    assert env.total_physics_steps == 10 and env.goal == initial_goal
    env.reset(seed=11)
    assert env.goal[0] == 16.
    env.close()
    env = NavigationEnv(EnvConfig(route_stages=stages, split="validation"))
    env.reset(seed=11)
    assert env.goal[0] == 27.
    env.close()
    with pytest.raises(ValueError, match="another curriculum"):
        EnvConfig(route_stages=stages, empty_pretraining_steps=10)
    with pytest.raises(ValueError, match="stage"):
        EnvConfig(route_stages=list(reversed(stages)))


def test_rehearsal_retains_completed_routes_only_and_shared_validation_stays_full():
    stages = [{"until_physics_steps":10,"route_m":6.}, {"until_physics_steps":20,"route_m":12.}]
    for count, allowed in ((0,{6.}), (10,{6.,12.}), (20,{6.,12.,23.})):
        seen = set()
        for seed in range(24):
            observations = []
            for method in ("H","B1","B2","B3"):
                env = NavigationEnv(EnvConfig(method=method, route_stages=stages, route_rehearsal_probability=.5, obstacle_num=0))
                env.total_physics_steps = count
                observation,_ = env.reset(seed=seed)
                distance = env.goal[0]-env.initial_scenario["initial"]["x"]
                seen.add(distance)
                assert distance in allowed
                if distance != 23.:
                    assert env.scenario_name == "empty" and not env.world.static and not env.world.dynamic
                observations.append(observation)
                env.close()
            for obs in observations[1:]:
                np.testing.assert_array_equal(obs, observations[0])
        assert seen == allowed
    env = NavigationEnv(EnvConfig(route_stages=stages, route_rehearsal_probability=.5, split="validation", scenario="static", obstacle_num=0))
    env.total_physics_steps = 30
    env.reset(seed=11)
    assert env.goal[0] == 27. and env.world.static
    env.close()
    for bad in (-.1, 1., float("nan")):
        with pytest.raises(ValueError, match="rehearsal"):
            EnvConfig(route_stages=stages, route_rehearsal_probability=bad)
    with pytest.raises(ValueError, match="rehearsal"):
        EnvConfig(route_rehearsal_probability=.25)
