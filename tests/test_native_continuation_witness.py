from dataclasses import asdict

import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.native_route import sample_road_grid
from navstack.vehicle import VehiclePlant, VehicleState
from scripts.audit_native_continuation_witness import probe_state, summarize


def test_proposed_endpoint_witness_is_not_a_local_reference_or_a_vehicle_step(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline query attempted execution')
    monkeypatch.setattr(VehiclePlant, 'step', forbidden)
    monkeypatch.setattr(NavigationEnv, 'step', forbidden)
    monkeypatch.setattr(NavigationEnv, 'reset', forbidden)
    scene = {'width': 32., 'height': 32., 'static': [], 'goal': [12., 16., 0.],
             'road_grid': sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'connector'},
                                          lambda x, y: True)}
    # Disable terminal truncation so an ordinary straight local target can lie
    # beyond the task goal. The local path is valid, but a forward connector
    # back to the goal is rejected. This must remain unknown, not infeasible.
    config = EnvConfig(terminal_goal_prefix=False, parking=False,
                       planner=PlannerConfig(max_seconds=None, analytic_expansion=True))
    state = asdict(VehicleState(8., 16.))
    result = probe_state(config, scene, state)
    assert result['state_goal_connector']
    assert 5 in result['reference_actions_with_target_goal_connector']  # 3 m straight
    assert 15 in result['reference_actions']  # 6 m straight, beyond goal
    assert 15 not in result['reference_actions_with_target_goal_connector']
    assert state == asdict(VehicleState(8., 16.))
    scene['dynamic'] = [{'x': 10., 'y': 16.}]
    with pytest.raises(ValueError, match='per-frame reconstruction'):
        probe_state(config, scene, state)


def test_first_missing_connector_does_not_claim_loss_of_all_references():
    rows = [
        {'decision': 0, 'state_goal_connector': False, 'reference_actions': [1, 2],
         'reference_actions_with_target_goal_connector': [], 'recorded_mode': 'reference', 'recorded_action': 1},
        {'decision': 1, 'state_goal_connector': True, 'reference_actions': [1, 2],
         'reference_actions_with_target_goal_connector': [2], 'recorded_mode': 'reference', 'recorded_action': 1},
        {'decision': 2, 'state_goal_connector': False, 'reference_actions': [2],
         'reference_actions_with_target_goal_connector': [2], 'recorded_mode': 'plan_failed', 'recorded_action': 30},
        {'decision': 3, 'state_goal_connector': False, 'reference_actions': [],
         'reference_actions_with_target_goal_connector': [], 'recorded_mode': 'wait', 'recorded_action': 0},
    ]
    result = summarize(rows)
    assert result == {'decisions': 4, 'first_no_moving_reference': 3,
                      'first_no_state_goal_connector': 0, 'no_moving_reference_decisions': 1,
                      'no_state_goal_connector_decisions': 3,
                      'selected_reference_without_target_connector_despite_alternative': 1,
                      'failed_selection_despite_reference': 1}
