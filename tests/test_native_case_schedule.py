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


def test_worker_rehearsal_preserves_unique_evaluation_inventory():
    import ast
    from pathlib import Path
    # Exercise the actual dispatch statements without starting a CARLA client.
    tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/native_road_learning.py').read_text(encoding='utf-8'))
    statements=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('cases','evaluation_cases') for t in node.targets):
            if isinstance(node.value,(ast.Call,ast.IfExp)):statements.append(node)
        elif isinstance(node,ast.If) and "protocol.get('training_case_schedule')" in ast.unparse(node.test):
            statements.append(node)
    statements.sort(key=lambda n:n.lineno)
    # Keep only this worker's initial dispatch,not later case assignments.
    first=next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.Call) and ast.unparse(n.value)=='registered_native_cases(protocol)')
    dispatch=ast.Module(body=statements[first:first+3],type_ignores=[])
    source=cases();p=dict(training_case_schedule=rehearsal_schedule(source),evaluation_only=False)
    namespace=dict(protocol=p,registered_native_cases=lambda protocol:source)
    exec(compile(dispatch,'native_case_dispatch','exec'),namespace)
    assert len(namespace['cases'])==64 and len(namespace['evaluation_cases'])==48
    assert namespace['evaluation_cases']==source
    p['evaluation_only']=True
    exec(compile(dispatch,'native_case_dispatch','exec'),namespace)
    assert len(namespace['cases'])==len(namespace['evaluation_cases'])==48


def test_measured_reset_order_audit_uses_the_saved_cursor_and_rejects_valid_but_reordered_cases():
    from scripts.audit_native_training_schedule import verify_order
    order=rehearsal_schedule(cases())['indices'];cursor=179
    observed=[order[(cursor+i)%64] for i in range(20)]
    assert verify_order(order,cursor,observed)==observed
    observed[0],observed[1]=observed[1],observed[0]
    with pytest.raises(ValueError):verify_order(order,cursor,observed)
