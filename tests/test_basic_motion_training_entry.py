from copy import deepcopy
import pytest

from test_native_case_manifest import protocol
from scripts.prepare_native_supervised_ppo import basic_motion_training_gate,representation_gate,copy_cleanup_harness


def example():
    p=protocol();p['methods']=['B1']
    rows=[{'case_id':i,'success':i in (0,1,2,4,6,8),
        'termination_reason':'success' if i in (0,1,2,4,6,8) else 'collision' if i==9 else 'deadline'} for i in range(12)]
    return p,{'episodes':rows}


def test_basic_motion_training_entry_never_claims_full_task_qualification():
    p,e=example();before=deepcopy(e)
    gate=basic_motion_training_gate(p,e)
    assert gate['basic_successes']==gate['required_basic_successes']==3 and gate['basic_cases']==4
    assert gate['full_task_successes']==6 and gate['full_task_cases']==12
    assert not gate['full_task_qualification_claimed'] and e==before
    with pytest.raises(ValueError):representation_gate(p,e) # Original full-inventory gate remains intact.


@pytest.mark.parametrize('fault',['few','unsafe','method','missing','duplicate','untyped'])
def test_basic_motion_entry_rejects_failed_sanity_or_incomplete_evidence(fault):
    p,e=example();rows=e['episodes']
    if fault=='few':rows[2]['success']=False
    elif fault=='unsafe':rows[3]['termination_reason']='out_of_bounds'
    elif fault=='method':p['methods']=['H','H0']
    elif fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]['case_id']=0
    else:rows[0]['success']='true'
    with pytest.raises(ValueError):basic_motion_training_gate(p,e)


@pytest.mark.parametrize('actuation_changed',[False,True])
def test_cleanup_override_cannot_change_actuation_or_policy_code(tmp_path,actuation_changed):
    archived=tmp_path/'old.py';current=tmp_path/'current.py';destination=tmp_path/'snapshot.py'
    archived.write_text('class Bridge:\n    def clear(self): return 1\n    def step(self): return 4\n')
    current.write_text('class Bridge:\n    def clear(self): return 2\n    def step(self): return '+('5' if actuation_changed else '4')+'\n')
    if actuation_changed:
        with pytest.raises(ValueError):copy_cleanup_harness(archived,current,destination)
        assert not destination.exists()
    else:
        copy_cleanup_harness(archived,current,destination)
        assert destination.read_bytes()==current.read_bytes()
