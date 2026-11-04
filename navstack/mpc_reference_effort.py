"""Explicit frozen-transfer gate for reference-centered MPC effort cost."""
VERSION='reference_centered_mpc_effort_v1'


def settings(protocol):
    value=protocol.get('mpc_reference_effort')
    if value is None:return False
    from .reference_point_constraints import settings as point_settings
    from .fixed_lowlevel_training import outcome_registered
    if (value!={'version':VERSION} or not (protocol.get('evaluation_only') is True or outcome_registered(protocol)) or not point_settings(protocol)):
        raise ValueError('MPC effort prototype requires registered fixed-policy point-constrained evaluation')
    return True
