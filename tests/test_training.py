import json
import numpy as np
import pytest
from navstack.config import EnvConfig
from navstack.experiments import train, load_run


@pytest.mark.training
@pytest.mark.parametrize("method", ["H", "B1"])
def test_real_ppo_updates_roundtrip(tmp_path, method):
    config = EnvConfig(method=method, scenario="empty", deadline=2.0)
    run, summary = train(config, seed=19, root=tmp_path, physics_budget=640)
    assert summary["optimization_phases"] >= 2
    assert (
        summary["actor_l2_change"] > 0
        and summary["reload_exact"]
        and summary["training_pass"]
    )
    assert summary["physics_steps"] >= 640
    model, norm = load_run(run)
    original = norm.obs_rms.mean.copy()
    model.predict(
        norm.normalize_obs(np.zeros(351, dtype=np.float32)), deterministic=True
    )
    np.testing.assert_array_equal(norm.obs_rms.mean, original)
    assert not norm.training
    norm.close()
    incompatible = EnvConfig(method="B2" if method == "H" else "H")
    with pytest.raises(ValueError, match="incompatible policy"):
        load_run(run, incompatible)
