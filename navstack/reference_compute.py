"""Opt-in exact computation shortcuts;no geometric tolerances or action changes."""
VERSION='exact_static_reference_compute_v3'


def enabled(protocol):
    value=protocol.get('reference_compute_profile')
    if value is None:return False
    allowed=protocol.get('methods') in (['H'],['H0'])
    if not allowed:
        from .shared_rule_registration import registered as rule_registered
        allowed=rule_registered(protocol)
    if (value!={'version':VERSION} or not allowed
            or protocol.get('reference_road_refinement')!=dict(version='sampled_native_reference_refinement_v1',max_attempts=3)
            or protocol.get('reference_steering_transition')!=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000)):
        raise ValueError('Exact reference compute profile requires the registered road/slew model')
    return True
