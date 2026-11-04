import json, sys
from pathlib import Path
import pytest


def test_carla_milestone_evaluation_never_substitutes_later_model(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from carla_formal_pipeline import checkpoint_for_fine_budget

    for requested, actual in ((50000, 51200), (100000, 102400)):
        path = tmp_path / f"checkpoint_{actual}"
        path.mkdir()
        (path / "training_summary.json").write_text(
            json.dumps(
                {"requested_physics_budget": requested, "physics_steps": actual}
            ),
            encoding="utf-8",
        )
    assert Path(checkpoint_for_fine_budget(tmp_path, 50000)).name == "checkpoint_51200"
    assert (
        Path(checkpoint_for_fine_budget(tmp_path, 100000)).name == "checkpoint_102400"
    )
    with pytest.raises(RuntimeError, match="No completed"):
        checkpoint_for_fine_budget(tmp_path, 200000)
