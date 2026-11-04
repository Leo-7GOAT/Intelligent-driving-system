from types import SimpleNamespace

import numpy as np
import pytest

from scripts.native_road_learning import timed_policy_step


@pytest.mark.parametrize('actions', [np.array([12]), np.array([[.25, -.1]], dtype=np.float32)])
def test_complete_native_decision_times_preprocessing_prediction_and_exact_action_once(actions):
    calls = []
    observation = np.array([1., 2.], dtype=np.float32)
    normalized = np.array([[3., 4.]], dtype=np.float32)
    step_result = (np.array([5., 6.]), 1., False, False, {'physical_frame': 42})

    def normalize(value):
        calls.append('normalize')
        np.testing.assert_array_equal(value, observation[None])
        return normalized

    def predict(value, deterministic):
        calls.append('predict')
        assert value is normalized and deterministic is True
        return actions, None

    def step(value):
        calls.append('step')
        np.testing.assert_array_equal(value, actions[0])
        return step_result

    ticks = iter([2., 2.012, 2.037])
    action, result, timing = timed_policy_step(SimpleNamespace(predict=predict),
        SimpleNamespace(normalize_obs=normalize), SimpleNamespace(step=step), observation, clock=lambda: next(ticks))
    assert calls == ['normalize', 'predict', 'step']
    assert action is actions and result is step_result
    assert timing['policy_prediction_seconds'] == pytest.approx(.012)
    assert timing['full_decision_seconds'] == pytest.approx(.037)
