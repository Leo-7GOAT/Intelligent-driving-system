"""Explicit frozen-transfer gate for reference-centered MPC effort cost."""
VERSION='reference_centered_mpc_effort_v1'


def settings(protocol):
    value=protocol.get('mpc_reference_effort')
    if value is None:return False
    from .reference_point_constraints import settings as point_settings
    if (value!={'version':VERSION} or protocol.get('evaluation_only') is not True or not point_settings(protocol)):
        raise ValueError('MPC effort prototype requires registered fixed-policy point-constrained evaluation')
    return True
