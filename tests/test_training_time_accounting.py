import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "training_time_accounting",
    Path(__file__).parents[1] / "scripts/training_time_accounting.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_slow_first_rollout_of_resumed_session_is_not_mistaken_for_continuation():
    # A new session's first rollout is longer than the entire previous short session.
    # There is no decrease in the elapsed field, but there are still two sessions.
    rows = [
        {"epoch": 105, "session_wall_seconds": 5},
        {"epoch": 110, "session_wall_seconds": 10},
        {"epoch": 220, "session_wall_seconds": 20},
        {"epoch": 230.000001, "session_wall_seconds": 30},
    ]
    result = module.session_training_seconds(rows)
    assert result["seconds"] == 40
    assert len(result["sessions"]) == 2
    with pytest.raises(ValueError):
        module.session_training_seconds([{"session_wall_seconds": 5}])
