"""Narrow registration gate for one fixed-lowlevel native adaptation course."""
VERSION='fixed_reference_option_native_adaptation_v1'
REGISTRATION=dict(version=VERSION,new_controls=5120,frozen_lowlevel=True,new_auxiliary_updates=0)
VERSION_POINTS='fixed_exact_point_joint_speed_native_adaptation_v1'
REGISTRATION_POINTS=dict(version=VERSION_POINTS,new_controls=5120,frozen_lowlevel=True,new_auxiliary_updates=0)


def registered(protocol):
    registration=protocol.get('fixed_lowlevel_training')
    if registration==REGISTRATION_POINTS:
        features=(protocol.get('reference_joint_speed')==dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25)
            and protocol.get('reference_point_constraints')==dict(version='native_counterexample_point_tube_v1')
            and protocol.get('reference_compute_profile')==dict(version='exact_static_reference_compute_v3'))
    elif registration==REGISTRATION:
        features=protocol.get('reference_joint_speed') is None and protocol.get('reference_point_constraints') is None
    else:return False
    return (features and protocol.get('evaluation_only') is False
        and type(protocol.get('fixed_lowlevel_training',{}).get('new_controls')) is int
        and protocol.get('methods')==['H']
        and type(protocol.get('controlled_budget_per_method')) is int
        and protocol.get('controlled_budget_per_method')==5120
        and protocol.get('method_budgets')=={'H':5120}
        and type(protocol.get('method_budgets',{}).get('H')) is int
        and protocol.get('freeze_observation_statistics') is True
        and protocol.get('record_initial_parameters') is True
        and protocol.get('reference_road_refinement')==dict(version='sampled_native_reference_refinement_v1',max_attempts=3)
        and protocol.get('reference_steering_transition')==dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000)
        and protocol.get('reference_option_commitment')==dict(version='selected_local_reference_option_v1'))
