import pytest
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


def test_short_learning_route_never_silently_shortens_validation():
    for method in ("H", "B1", "B2", "B3"):
        config = EnvConfig(method=method, scenario="empty", training_route_length=6)
        env = NavigationEnv(config)
        env.reset(seed=1)
        assert env.start_distance == pytest.approx(6)
        config.split = "validation"
        env.reset(seed=1)
        assert env.start_distance == pytest.approx(23)
        env.reset(seed=2, options={"diagnostic_route_length": 6})
        assert env.start_distance == pytest.approx(6)


def test_variable_training_distances_reproduce_without_changing_validation():
    config = EnvConfig(method="B1", scenario="empty", training_route_range=[6, 23])
    env = NavigationEnv(config)
    distances = []
    for seed in range(10):
        env.reset(seed=seed)
        distances.append(env.start_distance)
        env.reset(seed=seed)
        assert env.start_distance == distances[-1]
    assert all(6 <= d <= 23 for d in distances)
    assert max(distances)-min(distances) > 5
    for split in ("validation", "test"):
        config.split = split
        env.reset(seed=1)
        assert env.start_distance == pytest.approx(23)
    with pytest.raises(ValueError):
        EnvConfig(training_route_range=[23, 6])
    with pytest.raises(ValueError):
        EnvConfig(training_route_range=[6, 23], training_route_length=6)
