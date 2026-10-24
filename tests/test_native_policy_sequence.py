"""History reconstruction is checked against production observation updates."""
from dataclasses import asdict

import numpy as np
import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World
from scripts.audit_native_policy_sequence import observation_env, advance_observation


class FixtureEnv(NavigationEnv):
    def _create_episode(self, options=None):
        self.grid = sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'sequence'},
                                    lambda x, y: 12 < y < 20)
        return World(32., 32., road=RoadGrid(self.grid)), VehiclePlant(
            self.config.vehicle, VehicleState(6., 16.)), (24., 16., 0.), 'fixture'


@pytest.mark.parametrize('feedback', ['enabled', 'ablated'])
def test_reconstructed_history_matches_every_production_observation(feedback):
    config = EnvConfig(execution_feedback=feedback, candidate_observation=True,
                       road_observation=True, deadline=5., planner=PlannerConfig(
                           max_seconds=None, max_nodes=32, analytic_expansion=True))
    env = FixtureEnv(config)
    obs, _ = env.reset(seed=11)
    scene = {'width': 32., 'height': 32., 'goal': list(env.goal), 'static': [], 'road_grid': env.grid}
    initial = {'initial': asdict(env.state), 'goal': list(env.goal), 'static': [],
               'dynamic': [], 'road_sha256': env.world.road.sha256}
    replay = observation_env(config, scene, initial)
    assert not hasattr(replay.plant, 'step')
    np.testing.assert_array_equal(replay._observation(), obs)
    modes = set()
    try:
        for action in (5, 30, 0, 6):
            obs, _, done, _, info = env.step(action)
            modes.add(info['mode'])
            advance_observation(replay, {'policy_action': action, 'info': info})
            np.testing.assert_array_equal(replay._observation(), obs)
            assert replay.state == env.state
            if done:
                break
        assert {'wait', 'reference', 'plan_failed'} <= modes
    finally:
        env.close()


def test_reconstruction_rejects_unrecorded_input_queue():
    config = EnvConfig()
    config.vehicle.input_delay_steps = 1
    with pytest.raises(ValueError, match='no input queue'):
        observation_env(config, {}, {})
