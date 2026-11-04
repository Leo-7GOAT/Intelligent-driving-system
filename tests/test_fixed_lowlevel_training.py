from copy import deepcopy
import pytest
from navstack.fixed_lowlevel_training import REGISTRATION,registered
from navstack.reference_road_refinement import settings as road_settings
from navstack.steering_transition import settings as steering_settings
from navstack.reference_option_commitment import settings as option_settings


def protocol():
    return dict(evaluation_only=False,methods=['H'],controlled_budget_per_method=5120,method_budgets={'H':5120},
        freeze_observation_statistics=True,record_initial_parameters=True,fixed_lowlevel_training=deepcopy(REGISTRATION),
        reference_road_refinement=dict(version='sampled_native_reference_refinement_v1',max_attempts=3),
        reference_steering_transition=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000),
        reference_option_commitment=dict(version='selected_local_reference_option_v1'))


def test_one_exact_registered_training_course_is_enabled():
    p=protocol();assert registered(p)
    assert road_settings(p) and steering_settings(p) and option_settings(p)


@pytest.mark.parametrize('key,value',[
    ('controlled_budget_per_method',5120.),('method_budgets',{'H':5120.}),
    ('fixed_lowlevel_training',None),('freeze_observation_statistics',False),
    ('record_initial_parameters',False),('methods',['B1']),
    ('reference_steering_transition',dict(version='measured_steering_slew_primitives_v1'))])
def test_unregistered_or_changed_training_is_rejected(key,value):
    p=protocol();p[key]=value
    assert not registered(p)
    for settings in [road_settings,steering_settings,option_settings]:
        with pytest.raises(ValueError):settings(p)


def test_exact_point_course_requires_its_distinct_registration_and_all_frozen_features():
    from navstack.fixed_lowlevel_training import REGISTRATION_POINTS
    from navstack.steering_retiming import settings as timing_settings
    from navstack.reference_point_constraints import settings as point_settings
    p=protocol();p.update(fixed_lowlevel_training=deepcopy(REGISTRATION_POINTS),
        reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'),
        reference_compute_profile=dict(version='exact_static_reference_compute_v3'))
    assert registered(p) and timing_settings(p) and point_settings(p)
    assert road_settings(p)['point_constraints'] and steering_settings(p)['joint_speed']
    assert not registered({**p,'fixed_lowlevel_training':REGISTRATION})
    for patch in [dict(reference_point_constraints=None),dict(reference_compute_profile=None),
                  dict(reference_joint_speed={'version':'measured_steering_rate_retiming_v2','spatial_speed_ceiling':.5})]:
        assert not registered({**p,**patch})
        with pytest.raises(ValueError):timing_settings({**p,**patch})


def test_outcome_feedback_course_is_distinct_frozen_and_requires_its_complete_budget():
    from navstack.fixed_lowlevel_training import REGISTRATION_OUTCOME,outcome_registered
    from navstack.mpc_reference_effort import settings as mpc_settings
    p=protocol();p.update(fixed_lowlevel_training=deepcopy(REGISTRATION_OUTCOME),controlled_budget_per_method=10240,
        method_budgets={'H':10240},reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'),reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        mpc_reference_effort=dict(version='reference_centered_mpc_effort_v1'),reference_context=dict(version='action_aligned_outcome_context_v1'),
        outcome_context_registration=dict(version='action_aligned_outcome_context_v1',failure_gain_scale=1000.,parameter_states=34,new_supervision=0))
    assert registered(p) and outcome_registered(p) and mpc_settings(p)
    for patch in [dict(controlled_budget_per_method=5120),dict(method_budgets={'H':10240.}),
                  dict(reference_context={'version':'appended_reference_context_v1'}),dict(mpc_reference_effort=None),
                  dict(outcome_context_registration={**p['outcome_context_registration'],'failure_gain_scale':100.})]:
        assert not registered({**p,**patch})
