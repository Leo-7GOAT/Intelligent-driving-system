from dataclasses import asdict

import numpy as np
import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv, rule_action
from navstack.experiments import signature
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World


class RouteFixture(NavigationEnv):
    def _create_episode(self, options=None):
        road = sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'route-fixture'}, lambda x,y: True)
        self.scene_evidence = {'native_layout': {'path': [[6.,16.,0.], [24.,16.,0.]]}}
        start = (options or {}).get('start_x', 6.)
        return World(32.,32.,road=RoadGrid(road)), VehiclePlant(self.config.vehicle, VehicleState(start,16.)), (24.,16.,0.), 'fixture'


def config(method='H', feedback='enabled', mode='route_frame_terminal'):
    return EnvConfig(method=method, subgoal_mode=mode, execution_feedback=feedback,
        candidate_observation=True, road_observation=True, flat_hold=10,
        planner=PlannerConfig(max_seconds=None, analytic_expansion=True))


def test_route_information_parity_flat_stays_direct_and_signatures_change():
    h, b = RouteFixture(config()), RouteFixture(config('B1'))
    try:
        ho, _ = h.reset(seed=11)
        bo, _ = b.reset(seed=11)
        np.testing.assert_array_equal(ho,bo)
        assert ho.shape == (4701,)
        before = asdict(h.state)
        assert h.decode_action(5)[0] == (12.,16.,0.)
        assert asdict(h.state) == before
        def forbidden(*args):
            raise AssertionError('Flat control called the planner')
        b.planner.plan = forbidden
        b.step(np.array([.1,0.], dtype=np.float32))
        for method in ('H','B1'):
            assert signature(config(method)) != signature(config(method, mode='ego_arcs'))
    finally:
        h.close(); b.close()


def test_terminal_route_action_produces_stopping_reference_without_moving_state():
    env = RouteFixture(config())
    try:
        env.reset(options={'start_x':20.})
        before = asdict(env.state)
        assert all(env.decode_action(i)[0] == env.goal for i in range(1,31))
        action = rule_action(env)
        assert action != 0
        mode, detail = env._prepare_reference(action)
        assert mode == 'reference' and detail['terminal_goal_prefix']
        assert env.reference.v[-1] == 0
        assert asdict(env.state) == before
    finally:
        env.close()


def test_route_feedback_ablation_keeps_geometry_and_rejects_wrong_goal():
    env = RouteFixture(config(feedback='ablated'))
    try:
        obs, _ = env.reset()
        slots = obs[351:599].reshape(31,8)
        assert np.any(slots[:,:5]) and not np.any(slots[:,5:])
        env.navigation_route = ((6.,16.,0.),(23.,16.,0.))
        with pytest.raises(ValueError,match='endpoint differs'):
            env.decode_action(1)
        env.navigation_route = None
        with pytest.raises(ValueError,match='explicit known'):
            env.decode_action(1)
    finally:
        env.close()


def test_route_catalog_requires_shared_information():
    with pytest.raises(ValueError,match='shared candidate'):
        EnvConfig(subgoal_mode='route_frame_terminal')


def test_continuation_feedback_is_shared_and_terminal_target_is_a_witness():
    from navstack.env import candidate_goal_witness
    h, b = RouteFixture(config(feedback='continuation')), RouteFixture(config('B1',feedback='continuation'))
    try:
        ho, _ = h.reset(options={'start_x':20.})
        bo, _ = b.reset(options={'start_x':20.})
        np.testing.assert_array_equal(ho,bo)
        assert candidate_goal_witness(h.goal,h.goal,h.world,h.config.vehicle)
        assert np.all(ho[351:599].reshape(31,8)[1:,7] == 1)
        assert signature(config(feedback='continuation')) != signature(config())
    finally:
        h.close(); b.close()


def test_absent_continuation_witness_does_not_mask_moving_actions(monkeypatch):
    import navstack.env as module
    monkeypatch.setattr(module,'candidate_goal_witness',lambda *args:False)
    env=RouteFixture(config(feedback='continuation'))
    try:
        obs,_=env.reset()
        assert not np.any(obs[351:599].reshape(31,8)[:,7])
        action=rule_action(env)
        assert action != 0
        assert env._prepare_reference(action)[0]=='reference'
    finally:
        env.close()
