import json
from pathlib import Path
import numpy as np
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


def test_density_cases_really_add_obstacles_with_equal_policy_information():
    cases=json.loads((Path(__file__).resolve().parents[1]/"configs/generalization_cases.json").read_text(encoding="utf-8"))
    dense=[case for case in cases if "obstacle_num" in case.get("overrides",{})]
    assert len(dense)==6
    for case in dense:
        environments=[];observations=[]
        for method in ("H","B1"):
            config=EnvConfig(method=method,scenario=case["scenario"],split="test",obstacle_num=case["overrides"]["obstacle_num"])
            env=NavigationEnv(config);obs,_=env.reset(seed=case["seed"])
            assert case["scenario"]=="unseen" and len(env.world.static)==8
            environments.append(env);observations.append(obs)
        assert environments[0].initial_scenario==environments[1].initial_scenario
        np.testing.assert_array_equal(observations[0],observations[1])
        standard=NavigationEnv(EnvConfig(scenario="unseen",split="test",obstacle_num=2));standard.reset(seed=case["seed"])
        assert len(standard.world.static)==4
        assert standard.world.static==environments[0].world.static[:4]
        for env in [*environments,standard]:env.close()
