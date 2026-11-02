from collections import Counter
from copy import deepcopy
import pytest
from navstack.native_case_schedule import rehearsal_schedule,scheduled_native_cases,install_schedule_dispatch,BASIC


def cases():
    return [dict(condition=i,**({} if i%3==0 else {'native_hazard':{'kind':'static' if i%3==1 else 'walker'}})) for i in range(48)]


def test_rehearsal_exposes_every_condition_and_cycles_all_four_basic_roads_at_any_phase():
    source=cases();before=deepcopy(source);schedule=rehearsal_schedule(source)
    result=scheduled_native_cases(source,schedule);ids=[c['condition'] for c in result]
    assert source==before and len(ids)==64 and set(ids)==set(range(48))
    assert all(Counter(ids)[i]==4 for i in BASIC)
    for start in range(64):
        window=(ids+ids)[start:start+16]
        assert set(i for i in window if i in BASIC)==set(BASIC)


@pytest.mark.parametrize('fault',['missing','wrong_index','changed_order','hidden_hazard'])
def test_unregistered_rehearsal_or_condition_relabeling_fails_before_native_reset(fault):
    source=cases();schedule=rehearsal_schedule(source)
    if fault=='missing':source.pop()
    elif fault=='wrong_index':source[0]['condition']=1
    elif fault=='hidden_hazard':source[0]['native_hazard']={'kind':'static'}
    else:schedule['indices'][0]=4
    with pytest.raises(ValueError):scheduled_native_cases(source,schedule)


def test_training_overlay_never_expands_frozen_evaluation_cases():
    original='            cases = registered_native_cases(protocol)\n            following=1\n'
    changed=install_schedule_dispatch(original)
    assert "not protocol.get('evaluation_only')" in changed
    assert changed.endswith('            following=1\n')
    with pytest.raises(ValueError):install_schedule_dispatch(changed)


def test_measured_reset_order_audit_uses_the_saved_cursor_and_rejects_valid_but_reordered_cases():
    from scripts.audit_native_training_schedule import verify_order
    order=rehearsal_schedule(cases())['indices'];cursor=179
    observed=[order[(cursor+i)%64] for i in range(20)]
    assert verify_order(order,cursor,observed)==observed
    observed[0],observed[1]=observed[1],observed[0]
    with pytest.raises(ValueError):verify_order(order,cursor,observed)
