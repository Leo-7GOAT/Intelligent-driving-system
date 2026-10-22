import sys
from pathlib import Path
from types import SimpleNamespace


def test_resumed_world_ticks_include_previous_reset_cost(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from carla_finetune_pair import CarlaRecorder

    counts = {
        "physics_steps": 2652,
        "actions": 2560,
        "episodes": 2,
        "optimization_phases": 1,
        "reward_components": {},
        "modes": {},
        "action_histogram": {},
        "controlled_physics_steps": 2560,
        "reset_physics_steps": 92,
    }
    recorder = CarlaRecorder(tmp_path, 1e20, SimpleNamespace(total_ticks=31), counts)
    recorder.locals = {
        "actions": [[0, 0]],
        "infos": [
            {
                "executed_substeps": 1,
                "mode": "flat",
                "trace": [],
                "plan": {},
                "termination_reason": None,
            }
        ],
    }
    try:
        recorder._on_step()
        assert recorder.counts["physics_steps"] == 2683
        assert recorder.counts["controlled_physics_steps"] == 2561
        assert recorder.counts["reset_physics_steps"] == 122
    finally:
        recorder.close()
