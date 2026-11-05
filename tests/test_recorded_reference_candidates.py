from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probe_recorded_reference_candidates import (
    clone_option_env, reference_endpoint, selected_rows,
)


def test_case_local_decision_index_rejects_implicit_initial_state():
    rows = [{'case_id': 1, 'n': 0}, {'case_id': 2}, {'case_id': 1, 'n': 1}]
    assert selected_rows(rows, 1, 1) == (rows[0], rows[2])
    for index in (0, -1, True, 2):
        with pytest.raises(ValueError):
            selected_rows(rows, 1, index)


def test_clone_rebinds_closures_and_preserves_internal_reference_aliases():
    original = SimpleNamespace(reference=[], _reference_selection=('target', 1.),
                               _reference_option_version='old')
    original._held_option_reference = original.reference
    original.decode_action = lambda action: original.reference
    original._prepare_reference = lambda action: original.reference

    def configure(current, options):
        assert 'decode_action' not in current.__dict__
        assert '_prepare_reference' not in current.__dict__
        assert '_reference_option_version' not in current.__dict__
        current.decode_action = lambda action: current.reference
        current._prepare_reference = lambda action: current.reference

    first = clone_option_env(original, {}, configure, lambda p: None)
    second = clone_option_env(original, {}, configure, lambda p: None)
    first.reference.append('changed')
    assert first.reference is first._held_option_reference
    assert first.decode_action(1) == ['changed']
    assert second.decode_action(1) == original.reference == []


def test_endpoint_region_is_pose_only_and_wraps_heading():
    env = SimpleNamespace(reference=SimpleNamespace(x=[0., .2], y=[0., 0.],
                          yaw=[0., -np.pi+.01], v=[1., .3], t=np.array([0., 2.]), start_time=3.),
                          goal=(0., 0., np.pi-.01),
                          config=SimpleNamespace(goal_tolerance=.8, goal_yaw_tolerance=.35))
    result = reference_endpoint(env)
    assert result['inside_task_pose_region']
    assert result['yaw_error'] == pytest.approx(.02)
    assert result['end_speed'] == .3  # Pose inclusion alone is not parking success.
    assert result['end_time'] == 5.
    env.reference = None
    assert reference_endpoint(env) is None
