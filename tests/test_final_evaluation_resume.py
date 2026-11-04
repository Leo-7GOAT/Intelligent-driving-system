import hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from navstack.config import EnvConfig


def test_held_out_resume_skips_completed_cases_and_refuses_repeat(
    tmp_path, monkeypatch
):
    from scripts.evaluate_manifest import main
    import navstack.env, navstack.experiments

    batch = tmp_path / "batch"
    checkpoint = batch / "formal" / "H_seed11" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"env": EnvConfig().to_dict(), "seed": 11}), encoding="utf-8"
    )
    (checkpoint / "training_summary.json").write_text(
        json.dumps({"physics_steps": 1000000}), encoding="utf-8"
    )
    (batch / "batch.json").write_text(
        json.dumps({"deadline_epoch": time.time() + 1000}), encoding="utf-8"
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"scenario": "empty", "seed": s} for s in (1, 2, 3)]),
        encoding="utf-8",
    )
    (batch / "development_freeze.json").write_text(
        json.dumps(
            {
                "registered_source_versions": ["formal"],
                "manifest_hashes": {
                    "cases": hashlib.sha256(cases.read_bytes()).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    seen = []

    class FakeEnv:
        def __init__(self, c):
            self.done = False
            self.initial_scenario = {}
            self.episode = {"physics_steps": 1}

        def reset(self, seed):
            seen.append(seed)
            self.seed = seed
            return np.zeros(2), {}

        def step(self, a):
            self.done = True
            if self.seed == 1:
                (batch / "STOP").write_text("test interruption", encoding="utf-8")
            info = {
                "mode": "reference",
                "plan": {},
                "trace": [],
                "episode_metrics": {
                    "success": True,
                    "termination_reason": "success",
                    "progress": 23.0,
                    "physics_steps": 1,
                },
            }
            return np.zeros(2), 0.0, True, False, info

        def close(self):
            pass

    class Normalizer:
        obs_rms = SimpleNamespace(mean=np.zeros(2))

        def normalize_obs(self, obs):
            return obs

        def close(self):
            pass

    model = SimpleNamespace(predict=lambda obs, deterministic: (0, None))
    monkeypatch.setattr(navstack.env, "NavigationEnv", FakeEnv)
    monkeypatch.setattr(
        navstack.experiments, "load_run", lambda path: (model, Normalizer())
    )
    args = SimpleNamespace(
        source_root=str(Path(__file__).resolve().parents[1]),
        checkpoint=str(checkpoint),
        batch=str(batch),
        label="primary",
        split="test",
        cases=str(cases),
    )
    main(args)
    marker = checkpoint.parent / "final_primary.json"
    assert not json.loads(marker.read_text(encoding="utf-8"))["complete"]
    (batch / "STOP").unlink()
    main(args)
    result = json.loads(marker.read_text(encoding="utf-8"))
    assert (
        result["complete"]
        and result["completed_cases"] == 3
        and result["normalization_frozen"]
    )
    assert seen == [1, 2, 3]
    with pytest.raises(RuntimeError, match="completed result marker"):
        main(args)
    assert seen == [1, 2, 3]
