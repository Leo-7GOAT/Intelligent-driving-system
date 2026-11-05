from copy import deepcopy

import pytest

from navstack.contextual_feedback_registration import TRAINING_REGISTRATION, VERSION
from navstack.fixed_lowlevel_training import REGISTRATION_COHORT, registered, outcome_registered


def protocol():
    from test_fixed_lowlevel_training import protocol as original
    value = original()
    value.update(fixed_lowlevel_training=deepcopy(TRAINING_REGISTRATION),
        controlled_budget_per_method=10240, method_budgets={'H':10240},
        reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'),
        reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        mpc_reference_effort=dict(version='reference_centered_mpc_effort_v1'),
        reference_context=dict(version='action_aligned_outcome_context_v1'),
        outcome_context_registration=dict(version='action_aligned_outcome_context_v1',failure_gain_scale=1000.,
            parameter_states=34,new_supervision=0,failure_feedback_ablated=False),
        contextual_feedback_registration=dict(version=VERSION,hidden=32,actor_scale=100.,
            parameter_states=42,new_parameter_states=8,new_supervision=0,ablated=False))
    return value


def test_new_pilot_requires_complete_profile_and_keeps_all_fixed_physical_gates():
    from navstack.mpc_reference_effort import settings as mpc
    from navstack.steering_transition import settings as steering
    value = protocol()
    assert registered(value) and outcome_registered(value) and mpc(value) and steering(value)
    for key, patch in [('controlled_budget_per_method', 30720), ('freeze_observation_statistics', False),
                       ('reference_point_constraints', None), ('contextual_feedback_registration', None)]:
        assert not registered({**value, key:patch})
    old = {**value, 'fixed_lowlevel_training':deepcopy(REGISTRATION_COHORT),
           'controlled_budget_per_method':30720, 'method_budgets':{'H':30720}}
    assert not registered(old)
    old.pop('contextual_feedback_registration')
    assert registered(old)  # Existing cohorts retain their original admission.


@pytest.mark.parametrize('field,value', [('hidden',64),('actor_scale',1000.),('parameter_states',34),
                                       ('new_parameter_states',7),('new_supervision',1),('ablated',1),
                                       ('hidden',32.),('parameter_states',42.),('new_supervision',0.)])
def test_changed_conditional_profile_cannot_enter_native_training(field,value):
    p = protocol()
    p['contextual_feedback_registration'][field] = value
    assert not registered(p)
