"""Pure-data admission profile for the new conditional-feedback pilot."""
VERSION = 'state_conditioned_outcome_residual_v1'
HIDDEN = 32
ACTOR_SCALE = 100.
NEW_PARAMETER_STATES = 8
TOTAL_PARAMETER_STATES = 42
TRAINING_REGISTRATION = dict(version='fixed_conditional_outcome_TRAIN48_pilot_v1',
    new_controls=10240, frozen_lowlevel=True, new_auxiliary_updates=0)


def registered(protocol):
    value = protocol.get('contextual_feedback_registration')
    return (isinstance(value, dict) and type(value.get('ablated')) is bool
            and all(type(value.get(key)) is int for key in
                    ('hidden', 'parameter_states', 'new_parameter_states', 'new_supervision'))
            and value == dict(version=VERSION, hidden=HIDDEN, actor_scale=ACTOR_SCALE,
                              parameter_states=TOTAL_PARAMETER_STATES,
                              new_parameter_states=NEW_PARAMETER_STATES,
                              new_supervision=0, ablated=value['ablated']))
