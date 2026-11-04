from types import SimpleNamespace
import pytest
from scripts.query_native_persistent_teacher import HeldTeacher


def fixture():
    state=SimpleNamespace(x=0.);command=SimpleNamespace(steering=0.,acceleration=0.)
    expert=SimpleNamespace(config=SimpleNamespace(decision_dt=.5,control_dt=.05,vehicle=None),
        state=state,plant=SimpleNamespace(state=state,previous_command=command),reference=None)
    selections=[];references=[]
    def prepare(action):
        expert.reference=SimpleNamespace(selection=action);return 'reference',{'selection':action}
    expert._prepare_reference=prepare
    def rule(env):selections.append(len(selections));return selections[-1]
    def step(current,reference,previous):
        references.append(reference);return SimpleNamespace(command=command,status='solved')
    controller=SimpleNamespace(step=step)
    return expert,controller,rule,selections,references


def test_controller_and_reference_persist_between_decision_boundaries_without_plant_step():
    expert,controller,rule,selections,references=fixture()
    oracle=HeldTeacher(expert,controller,rule,lambda *_:None)
    for i in range(21):oracle.query(i)
    assert selections==[0,1,2]
    assert len(references)==21
    assert all(r is references[0] for r in references[:10])
    assert all(r is references[10] for r in references[10:20])
    assert references[0] is not references[10]
    assert oracle.controller is controller and not hasattr(expert.plant,'step')


def test_skipped_or_duplicate_measured_control_cannot_reset_the_context():
    expert,controller,rule,_,_=fixture();oracle=HeldTeacher(expert,controller,rule,lambda *_:None)
    oracle.query(0)
    for bad in (0,2):
        with pytest.raises(ValueError,match='exactly once'):oracle.query(bad)


def test_counterfactual_teacher_must_never_change_measured_state():
    expert,controller,rule,_,_=fixture()
    def invalid(state,reference,previous):
        state.x+=1;return SimpleNamespace(command=previous,status='solved')
    controller.step=invalid;oracle=HeldTeacher(expert,controller,rule,lambda *_:None)
    with pytest.raises(ValueError,match='measured plant'):oracle.query(0)


def test_fractional_control_cadence_is_rejected():
    expert,controller,rule,_,_=fixture();expert.config.decision_dt=.53
    with pytest.raises(ValueError,match='whole number'):HeldTeacher(expert,controller,rule,lambda *_:None)


def test_replay_fit_reuse_requires_identical_teacher_inputs():
    import numpy as np
    from scripts.probe_native_corrective_fit import require_replay_inputs
    data={'base_observations':np.zeros((2,4)),'base_actions':np.zeros((2,2)),'base_weights':np.ones(2)}
    require_replay_inputs(data,{k:v.copy() for k,v in data.items()})


@pytest.mark.parametrize('key',['base_observations','base_actions','base_weights'])
def test_changed_replay_training_data_cannot_be_reused_as_the_same_control(key):
    import numpy as np
    from scripts.probe_native_corrective_fit import require_replay_inputs
    data={'base_observations':np.zeros((2,4)),'base_actions':np.zeros((2,2)),'base_weights':np.ones(2)}
    changed={k:v.copy() for k,v in data.items()};changed[key].flat[0]+=1
    with pytest.raises(AssertionError):require_replay_inputs(data,changed)
