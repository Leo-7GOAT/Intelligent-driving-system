"""Opt-in real CARLA PPO continuation, including optimizer and reset accounting."""

import json, os, sys, time, pickle, random
from pathlib import Path
import numpy as np
import torch
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_CARLA_TRAINING_RESUME") != "1",
    reason="Requires exclusive live CARLA owner and explicit opt-in",
)
def test_actual_carla_optimizer_resume():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from carla_finetune_pair import finetune

    batch = Path(os.environ["CARLA_BATCH"]).resolve()
    origin = json.loads(
        (batch / "development_v1/B1_seed11/latest.json").read_text(encoding="utf-8")
    )["checkpoint"]
    run = batch / "carla_resume_verification" / ("B1_seed11_" + str(time.time_ns()))
    first = finetune(batch, origin, 1, 11, run)
    with (Path(first["checkpoint"]) / "resume_stats.pkl").open("rb") as stream:
        saved_rng = pickle.load(stream)
    assert random.getstate() == saved_rng["random"]
    assert np.array_equal(np.random.get_state()[1], saved_rng["numpy_random"][1])
    assert torch.equal(torch.get_rng_state(), saved_rng["torch_random"])
    second = finetune(batch, origin, first["physics_steps"] + 1, 11, run)
    assert second["physics_steps"] > first["physics_steps"]
    assert second["controlled_physics_steps"] == 2 * first["controlled_physics_steps"]
    assert second["reset_physics_steps"] > first["reset_physics_steps"]
    assert second["optimizer_steps"] > first["optimizer_steps"] > 0
    assert second["ppo_epoch_updates_added"] == 2 * first["ppo_epoch_updates_added"]
    assert first["reload_exact"] and second["reload_exact"]
    assert second["actor_l2_change"] > 0
