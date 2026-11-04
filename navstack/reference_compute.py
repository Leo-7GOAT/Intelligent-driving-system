"""Opt-in exact computation shortcuts;no geometric tolerances or action changes."""
VERSION='exact_static_reference_compute_v3'


def enabled(protocol):
    value=protocol.get('reference_compute_profile')
    if value is None:return False
    if (value!={'version':VERSION} or protocol.get('methods') not in (['H'],['H0'])
            or protocol.get('reference_road_refinement')!=dict(version='sampled_native_reference_refinement_v1',max_attempts=3)
            or protocol.get('reference_steering_transition')!=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000)):
        raise ValueError('Exact reference compute profile requires the registered road/slew model')
    return True
