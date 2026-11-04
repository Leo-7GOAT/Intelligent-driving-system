"""Explicit repeated-option persistence;the policy still chooses every action."""
VERSION='selected_local_reference_option_v1'


def settings(protocol):
    value=protocol.get('reference_option_commitment')
    if value is None:return None
    from .fixed_lowlevel_training import registered
    if (value!={'version':VERSION} or not (protocol.get('evaluation_only') is True or registered(protocol))
            or protocol.get('methods') not in (['H'],['H0'])):
        raise ValueError('Reference option prototype requires registered frozen H/H0 evaluation')
    return value


def configure(env,options):
    if options is None:return env
    if getattr(env,'_reference_option_version',None)==VERSION:return env
    decode,prepare=env.decode_action,env._prepare_reference
    def held(action):
        selection=getattr(env,'_reference_selection',None)
        return (int(action)!=0 and int(action)==getattr(env,'_held_reference_action',None)
            and env.reference is not None and env.reference is getattr(env,'_held_option_reference',None)
            and selection is not None and env.reusable_reference(*selection))
    def decode_option(action):
        return env._reference_selection if held(action) else decode(action)
    def prepare_option(action):
        retained=held(action)
        mode,detail=prepare(action)
        if mode=='reference':
            if env.reference is None:raise ValueError('Reference mode requires a selected timed reference')
            env._held_reference_action=int(action);env._held_option_reference=env.reference
        else:
            env._held_reference_action=None;env._held_option_reference=None
        if detail is not None:
            detail=dict(detail)
            detail['reference_option_commitment']=dict(version=VERSION,policy_action=int(action),retained_before_prepare=bool(retained))
        return mode,detail
    env.decode_action=decode_option;env._prepare_reference=prepare_option
    env._reference_option_version=VERSION
    return env
