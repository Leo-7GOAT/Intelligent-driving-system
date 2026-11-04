from types import SimpleNamespace
import pytest
from navstack.reference_option_commitment import VERSION,settings,configure
from navstack.reference_context import reference_candidate_context


def make_env():
    env=SimpleNamespace(reference=None,_reference_selection=None,offset=0.,valid=True)
    env.decode_action=lambda action:((env.offset+int(action),0.,0.),1.5)
    env.reusable_reference=lambda *selection:env.valid and env.reference is not None and tuple(selection)==env._reference_selection
    def prepare(action):
        if int(action)==0:
            env.reference=None;env._reference_selection=None;return 'wait',None
        selection=env.decode_action(action)
        if selection==env._reference_selection and env.valid:return 'reference',{'status':'reference_reused'}
        env._reference_selection=selection;env.reference=object()
        return 'reference',{'status':'success'}
    env._prepare_reference=prepare
    return configure(env,{'version':VERSION})


def test_policy_repetition_holds_only_the_explicitly_selected_valid_option():
    e=make_env();e._prepare_reference(12);ref=e.reference;selection=e._reference_selection
    e.offset=1.
    assert e.decode_action(12)==selection
    assert e.decode_action(2)!=selection
    context=reference_candidate_context(e)
    assert context[12]==1 and context[2]==0
    mode,detail=e._prepare_reference(12)
    assert mode=='reference' and e.reference is ref and detail['status']=='reference_reused'
    assert detail['reference_option_commitment']['retained_before_prepare']
    e._prepare_reference(2)
    assert e.reference is not ref and e._held_reference_action==2


def test_invalid_reference_and_new_episode_never_retain_an_old_target():
    e=make_env();e._prepare_reference(12);old=e._reference_selection;e.offset=1.;e.valid=False
    assert e.decode_action(12)!=old
    e.valid=True;e.reference=None;e._reference_selection=None
    assert e.decode_action(12)!=old
    e._prepare_reference(12);e._prepare_reference(0)
    assert e._held_reference_action is None and e._held_option_reference is None


def test_default_off_and_versioned_evaluation_only():
    e=SimpleNamespace();assert configure(e,None) is e and settings({}) is None
    p=dict(reference_option_commitment={'version':VERSION},evaluation_only=True,methods=['H'])
    assert settings(p)
    for change in [{'evaluation_only':False},{'methods':['B1']}]:
        with pytest.raises(ValueError):settings({**p,**change})
