"""Narrow registration gate for one fixed-lowlevel native adaptation course."""
VERSION='fixed_reference_option_native_adaptation_v1'
REGISTRATION=dict(version=VERSION,new_controls=5120,frozen_lowlevel=True,new_auxiliary_updates=0)
VERSION_POINTS='fixed_exact_point_joint_speed_native_adaptation_v1'
REGISTRATION_POINTS=dict(version=VERSION_POINTS,new_controls=5120,frozen_lowlevel=True,new_auxiliary_updates=0)
VERSION_OUTCOME='fixed_outcome_context_native_adaptation_v1'
REGISTRATION_OUTCOME=dict(version=VERSION_OUTCOME,new_controls=10240,frozen_lowlevel=True,new_auxiliary_updates=0)
REGISTRATION_COHORT=dict(version='fixed_outcome_TRAIN48_independent_v1',new_controls=30720,
    frozen_lowlevel=True,new_auxiliary_updates=0)


def registered(protocol):
    registration=protocol.get('fixed_lowlevel_training')
    budget=5120
    contextual=(isinstance(registration,dict)
        and registration.get('version')=='fixed_conditional_outcome_TRAIN48_pilot_v1')
    if contextual:
        from .contextual_feedback_registration import TRAINING_REGISTRATION,registered as contextual_registered
        if registration!=TRAINING_REGISTRATION or not contextual_registered(protocol):return False
    elif protocol.get('contextual_feedback_registration') is not None:return False
    if registration in (REGISTRATION_OUTCOME,REGISTRATION_COHORT) or contextual:
        budget=registration['new_controls']
        features=(protocol.get('reference_joint_speed')==dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25)
            and protocol.get('reference_point_constraints')==dict(version='native_counterexample_point_tube_v1')
            and protocol.get('reference_compute_profile')==dict(version='exact_static_reference_compute_v3')
            and protocol.get('mpc_reference_effort')==dict(version='reference_centered_mpc_effort_v1')
            and protocol.get('reference_context')==dict(version='action_aligned_outcome_context_v1')
            and type(protocol.get('outcome_context_registration',{}).get('failure_feedback_ablated')) is bool
            and protocol.get('outcome_context_registration')==dict(version='action_aligned_outcome_context_v1',failure_gain_scale=1000.,parameter_states=34,new_supervision=0,
                failure_feedback_ablated=protocol.get('outcome_context_registration',{}).get('failure_feedback_ablated'))
            and (not contextual or contextual_registered(protocol)))
    elif registration==REGISTRATION_POINTS:
        features=(protocol.get('reference_joint_speed')==dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25)
            and protocol.get('reference_point_constraints')==dict(version='native_counterexample_point_tube_v1')
            and protocol.get('reference_compute_profile')==dict(version='exact_static_reference_compute_v3')
            and protocol.get('reference_context') in (None,dict(version='appended_reference_context_v1'))
            and protocol.get('mpc_reference_effort') is None)
    elif registration==REGISTRATION:
        features=protocol.get('reference_joint_speed') is None and protocol.get('reference_point_constraints') is None
    else:return False
    segment_budget=budget
    recovery=protocol.get('fixed_cohort_recovery')
    if recovery is not None:
        prior=protocol.get('recovery',{}).get('H',{})
        if (registration!=REGISTRATION_COHORT or not isinstance(recovery,dict)
                or set(recovery)!={'version','original_target','saved_controls','parent_protocol_sha256','interruption_audit_sha256'}
                or recovery.get('version')!='saved_outcome_cohort_recovery_v1'
                or type(recovery.get('original_target')) is not int or recovery['original_target']!=budget
                or type(recovery.get('saved_controls')) is not int or not 0<recovery['saved_controls']<budget
                or prior.get('lineage_controls')!=recovery['saved_controls']
                or any(not isinstance(recovery.get(k),str) or len(recovery[k])!=64
                    or any(c not in '0123456789abcdef' for c in recovery[k])
                    for k in ('parent_protocol_sha256','interruption_audit_sha256'))):return False
        segment_budget=budget-recovery['saved_controls']
    return (features and protocol.get('evaluation_only') is False
        and type(protocol.get('fixed_lowlevel_training',{}).get('new_controls')) is int
        and protocol.get('methods')==['H']
        and type(protocol.get('controlled_budget_per_method')) is int
        and protocol.get('controlled_budget_per_method')==segment_budget
        and protocol.get('method_budgets')=={'H':segment_budget}
        and type(protocol.get('method_budgets',{}).get('H')) is int
        and protocol.get('freeze_observation_statistics') is True
        and protocol.get('record_initial_parameters') is True
        and protocol.get('reference_road_refinement')==dict(version='sampled_native_reference_refinement_v1',max_attempts=3)
        and protocol.get('reference_steering_transition')==dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000)
        and protocol.get('reference_option_commitment')==dict(version='selected_local_reference_option_v1'))


def outcome_registered(protocol):
    value=protocol.get('fixed_lowlevel_training')
    return (value in (REGISTRATION_OUTCOME,REGISTRATION_COHORT)
        or isinstance(value,dict) and value.get('version')=='fixed_conditional_outcome_TRAIN48_pilot_v1') and registered(protocol)
