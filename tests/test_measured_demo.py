from copy import deepcopy
import numpy as np
import pytest

from navstack.config import EnvConfig
from navstack.measured_demo import measured_flat_env, measured_label, advance_measured_flat
from test_route_catalog_environment import RouteFixture, config


def fixture():
    cfg = EnvConfig.from_dict({**config('B1').to_dict(), 'flat_hold': 1})
    live = RouteFixture(cfg)
    first, _ = live.reset(seed=11)
    scene = {'width': live.world.width, 'height': live.world.height, 'static': live.initial_scenario['static'],
             'goal': live.initial_scenario['goal'], 'road_grid': live.world.road.payload,
             'native_layout': live.scene_evidence['native_layout']}
    view = measured_flat_env(cfg, scene, live.initial_scenario)
    return live, view, first


def test_measured_views_reconstruct_direct_history_without_integrating_or_calling_planner(monkeypatch):
    live, view, first = fixture()
    def forbidden(*args):
        raise AssertionError('Measured view must not execute a plant or planner')
    monkeypatch.setattr(view.planner, 'plan', forbidden)
    try:
        assert not hasattr(view.plant, 'step')
        np.testing.assert_array_equal(view._observation(), first)
        for action in (np.array([.3, .2], dtype=np.float32), np.array([-.4, -.1], dtype=np.float32)):
            expected, _, _, _, info = live.step(action)
            tick = info['trace'][0]
            label = measured_label(view, tick)
            np.testing.assert_array_equal(label, action)
            advance_measured_flat(view, tick, label)
            np.testing.assert_array_equal(view._observation(), expected)
    finally:
        live.close()
        view.close()


@pytest.mark.parametrize('fault', ['time', 'command', 'label'])
def test_measured_view_rejects_unaccounted_controls_and_changed_labels(fault):
    live, view, _ = fixture()
    try:
        _, _, _, _, info = live.step(np.array([.2, 0.], dtype=np.float32))
        tick = deepcopy(info['trace'][0])
        if fault == 'time': tick['state']['time'] += .05
        elif fault == 'command': tick['requested_command']['acceleration'] = 100.
        if fault == 'label':
            with pytest.raises(ValueError): advance_measured_flat(view, tick, np.array([0., 0.]))
        else:
            with pytest.raises(ValueError): measured_label(view, tick)
    finally:
        live.close()
        view.close()


def test_small_qp_residual_uses_production_clipping_without_changing_raw_request():
    live, view, _ = fixture()
    try:
        _, _, _, _, info = live.step(np.array([1., 0.], dtype=np.float32))
        tick = deepcopy(info['trace'][0])
        tick['requested_command']['acceleration'] = live.config.vehicle.max_accel+.0002
        before = deepcopy(tick)
        label = measured_label(view, tick)
        np.testing.assert_array_equal(label, [1., 0.])
        assert tick == before
    finally:
        live.close()
        view.close()
