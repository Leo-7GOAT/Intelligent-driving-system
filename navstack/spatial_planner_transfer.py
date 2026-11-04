"""Explicit evaluation-only replacement of the selected spatial planner.

The H label retains the frozen actor's action/observation/checkpoint schema.
This registration labels the experiment B2 and selects the actual Grid backend;
it does not enable learning, change MPC, or claim a retrained B2 policy.
"""
VERSION = 'frozen_grid_body_spline_transfer_v1'
PROFILE = dict(version=VERSION, max_nodes=32000, grid_postprocess='body_spline_v1')


def settings(protocol):
    value = protocol.get('reference_spatial_planner')
    if value is None:
        return None
    recovery = protocol.get('recovery', {}).get('H', {})
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (not isinstance(value, dict) or value != PROFILE or type(value.get('max_nodes')) is not int
            or protocol.get('methods') != ['H'] or protocol.get('evaluation_only') is not True
            or type(protocol.get('controlled_budget_per_method')) is not int
            or protocol.get('controlled_budget_per_method') != 0
            or protocol.get('method_budgets') != {'H': 0}
            or type(protocol.get('method_budgets', {}).get('H')) is not int
            or set(protocol.get('recovery', {})) != {'H'}
            or not digest(recovery.get('model_sha256')) or not digest(recovery.get('normalization_sha256'))
            or protocol.get('reference_steering_transition') != dict(version='measured_steering_slew_32000_memo_v2', max_nodes=32000)
            or protocol.get('reference_joint_speed') != dict(version='measured_steering_rate_retiming_v2', spatial_speed_ceiling=.25)
            or protocol.get('reference_compute_profile') != dict(version='exact_static_reference_compute_v3')):
        raise ValueError('Grid spatial transfer requires registered zero-learning frozen H evaluation')
    return dict(value)
