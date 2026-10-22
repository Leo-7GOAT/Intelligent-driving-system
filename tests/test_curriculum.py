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
