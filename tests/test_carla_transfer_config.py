import json
import pytest
from navstack.config import EnvConfig, VehicleConfig, PlannerConfig
from navstack.env import NavigationEnv
from scripts.carla_diagnostic import calibrated_config


def test_calibration_preserves_learned_action_observation_and_planner(tmp_path):
    (tmp_path / "calibration").mkdir()
    vehicle = VehicleConfig(wheelbase=3.0)
    from dataclasses import asdict
    (tmp_path / "calibration/calibration.json").write_text(json.dumps({"vehicle": asdict(vehicle)}))
    for method in ("H", "B1", "B2"):
        original = EnvConfig(method=method, subgoal_mode="goal_frame_terminal", execution_feedback="enabled",
                             deadline=45, terminal_goal_prefix=True,
                             planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))
        before = original.to_dict()
        transferred = calibrated_config(tmp_path, method, source_config=original)
        expected = {**before, "vehicle": asdict(vehicle)}
        assert transferred.to_dict() == expected
        assert original.to_dict() == before
        assert NavigationEnv(transferred).observation_space.shape == (450,)
    with pytest.raises(ValueError):
        calibrated_config(tmp_path, "B3", source_config=original)
