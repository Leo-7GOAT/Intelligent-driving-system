"""Offline reference queries cannot stand in for physical execution evidence."""
from dataclasses import asdict

import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.native_route import sample_road_grid
from navstack.vehicle import VehiclePlant, VehicleState
from scripts.audit_native_action_availability import state_references


def fixture_scene():
    return {'width': 32., 'height': 32., 'static': [], 'goal': [20., 16., 0.],
            'road_grid': sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'offline'},
                                         lambda x, y: True)}


def test_queries_do_not_step_and_distinguish_references_from_parking(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline diagnostic attempted physical execution')
    monkeypatch.setattr(VehiclePlant, 'step', forbidden)
    config = EnvConfig(planner=PlannerConfig(max_seconds=None, analytic_expansion=True))
    distant = state_references(config, fixture_scene(), asdict(VehicleState(8., 16.)))
    assert distant['mode_counts']['wait'] == 1
    assert distant['reference_actions']
    assert distant['progress_reference_actions']
    assert distant['parking_reference_actions'] == []
    # A target-crossing reference actually contains a slow aligned parking pose.
    nearby = state_references(config, fixture_scene(), asdict(VehicleState(18., 16.)))
    assert nearby['parking_reference_actions']
    assert 'success' not in nearby


def test_refuses_unreconstructed_dynamic_world_and_wall_clock_search():
    config = EnvConfig(planner=PlannerConfig(max_seconds=None))
    scene = fixture_scene()
    scene['dynamic'] = [{'x': 10., 'y': 16.}]
    with pytest.raises(ValueError, match='per-frame reconstruction'):
        state_references(config, scene, asdict(VehicleState(8., 16.)))
    with pytest.raises(ValueError, match='node-budget'):
        state_references(EnvConfig(planner=PlannerConfig(max_seconds=.1)), fixture_scene(),
                         asdict(VehicleState(8., 16.)))
