import json
import pytest
from navstack.selection import select_checkpoints
from navstack.selection import development_rank


def test_selection_does_not_prefer_permanent_parking_over_safe_completion():
    def row(event):
        return {"termination_reason": event, "success": event == "success", "progress": 23 if event == "success" else 0,
                "intervention_rate": 0, "travel_time_success_only": 20 if event == "success" else None}
    stopped = [row("deadline") for _ in range(40)]
    productive = [row("success") for _ in range(39)] + [row("collision")]
    unsafe = [row("success") for _ in range(35)] + [row("collision") for _ in range(5)]
    assert development_rank(productive) < development_rank(stopped) < development_rank(unsafe)


def test_selection_refuses_best_seed_cherry_picking(tmp_path):
    paths = []
    for seed in (11, 22):
        path = tmp_path / str(seed)
        path.mkdir()
        paths.append(path)
        (path / "config.json").write_text(
            json.dumps({"seed": seed, "env": {}}), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="best training seed"):
        select_checkpoints(paths)
