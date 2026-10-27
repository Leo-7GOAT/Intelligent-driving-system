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


def dynamic_fixture():
    from dataclasses import asdict
    from navstack.world import MovingObstacle
    live, old_view, _ = fixture(); old_view.close()
    live.world.dynamic = [MovingObstacle(12., 14., 0., .8, .5)]
    initial = deepcopy(live.initial_scenario)
    initial['dynamic'] = [asdict(o) for o in live.world.dynamic]
    scene = {'width': live.world.width, 'height': live.world.height, 'static': initial['static'],
             'goal': initial['goal'], 'road_grid': live.world.road.payload,
             'native_layout': live.scene_evidence['native_layout']}
    begin = {'scene': scene, 'state': initial['initial'], 'dynamic': initial['dynamic'],
             'frame': 10, 'episode_id': 'measured-fixture',
             'native_dynamic_actors': [{'actor_id': 42, 'actor_type': 'walker.pedestrian.0001',
                                       'state': initial['dynamic'][0]}]}
    view = measured_flat_env(live.config, scene, initial, begin=begin)
    _, _, _, _, info = live.step(np.array([.2, 0.], dtype=np.float32))
    tick = deepcopy(info['trace'][0])
    # Deliberately not v*dt: the view must copy the given measured snapshot,
    # rather than advancing a Python obstacle or inventing its next position.
    live.world.dynamic[0].y = 14.1237
    state = asdict(live.world.dynamic[0])
    tick['backend_frame'] = 11
    tick['backend_evidence'] = {'frame': 11, 'episode_id': 'measured-fixture',
                               'native_dynamic_actors': [{'actor_id': 42, 'actor_type': 'walker.pedestrian.0001', 'state': state}]}
    return live, view, tick


def test_dynamic_view_uses_exact_measured_snapshot_without_world_advance(monkeypatch):
    live, view, tick = dynamic_fixture()
    def forbidden(*args): raise AssertionError('Measured observation attempted simulation or planning')
    monkeypatch.setattr(view.world, 'advance', forbidden)
    monkeypatch.setattr(view.planner, 'plan', forbidden)
    try:
        advance_measured_flat(view, tick, measured_label(view, tick))
        assert view.world.dynamic[0].y == 14.1237 and not hasattr(view.plant, 'step')
        np.testing.assert_array_equal(view._observation(), live._observation())
    finally: live.close(); view.close()


@pytest.mark.parametrize('fault', ['actor', 'missing', 'episode', 'frame', 'nonfinite'])
def test_dynamic_view_rejects_inconsistent_measured_snapshots_before_mutation(fault):
    live, view, tick = dynamic_fixture()
    try:
        from dataclasses import asdict
        before = (asdict(view.state), asdict(view.world.dynamic[0]))
        evidence = tick['backend_evidence']
        if fault == 'actor': evidence['native_dynamic_actors'][0]['actor_id'] = 43
        if fault == 'missing': evidence.pop('native_dynamic_actors')
        if fault == 'episode': evidence['episode_id'] = 'different'
        if fault == 'frame': evidence['frame'] += 1
        if fault == 'nonfinite': evidence['native_dynamic_actors'][0]['state']['y'] = float('nan')
        with pytest.raises(ValueError): advance_measured_flat(view, tick, [.2, 0.])
        assert before == (asdict(view.state), asdict(view.world.dynamic[0]))
    finally: live.close(); view.close()


def test_brake_measured_steering_overshoot_uses_only_exact_production_acknowledgement():
    from dataclasses import replace
    from navstack.vehicle import Command
    live, view, _ = fixture()
    try:
        _, _, _, _, info = live.step(np.array([.2, 0.], dtype=np.float32))
        tick = deepcopy(info['trace'][0])
        view.plant.state = replace(view.state, steering=.5036)
        view.plant.previous_command = Command(0., .49)
        tick['requested_command'] = {'acceleration': -view.config.vehicle.max_decel, 'steering': .5036}
        tick['command'] = {'acceleration': -view.config.vehicle.max_decel, 'steering': .5}
        tick['controller_status'] = 'dynamic_brake'
        original = deepcopy(tick)
        np.testing.assert_array_equal(measured_label(view, tick), [-1., 1.])
        assert tick == original
        tick['command']['steering'] = .49
        with pytest.raises(ValueError, match='acknowledgement'): measured_label(view, tick)
        tick = deepcopy(original); tick['controller_status'] = 'solved'
        with pytest.raises(ValueError, match='physical limits'): measured_label(view, tick)
    finally: live.close(); view.close()
