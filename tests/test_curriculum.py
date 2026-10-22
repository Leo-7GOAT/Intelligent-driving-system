import numpy as np
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
