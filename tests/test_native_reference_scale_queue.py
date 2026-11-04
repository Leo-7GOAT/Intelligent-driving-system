import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import queue_native_reference_scale as queue


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def predecessor(tmp_path,monkeypatch):
    baseline=tmp_path/'baseline';execution=tmp_path/'old_execution';results={}
    for method in ('H','H0'):
        folder=baseline/method;folder.mkdir(parents=True);(folder/'model.zip').write_bytes(method.encode())
    for phase,count in (('static',12),('known_validation',24)):
        batch=tmp_path/phase;measured={}
        write(batch/'strict_delivery.json',{'strict_delivery_gate_passed':True})
        for method in ('H','H0'):
            model_sha=queue.sha(baseline/method/'model.zip')
            write(batch/method/'endpoint_raw_audit.json',{'method':method})
            for name in ('measured_start_inference.json','static_action_inference.json'):
                write(batch/method/name,dict(all_recorded_actions_match=True,model_sha256=model_sha))
            measured[method]=dict(cases=count,successes=count,model_sha256=model_sha,
                raw_audit_sha256=queue.sha(batch/method/'endpoint_raw_audit.json'))
        results[phase]=dict(batch=str(batch),results=measured)
    write(execution/'complete.json',dict(results=results))
    monkeypatch.setattr(queue,'require_finished_parent',lambda path:None)
    return execution,baseline,results


def test_requires_both_complete_audited_partitions(predecessor):
    execution,baseline,_=predecessor
    assert set(queue.completed_predecessor(execution,baseline)['results'])=={'static','known_validation'}


@pytest.mark.parametrize('fault',['short_inventory','changed_model','changed_raw','failed_NN','failed_engine'])
def test_refuses_incomplete_or_changed_predecessor(predecessor,fault):
    execution,baseline,results=predecessor;batch=Path(results['static']['batch'])
    if fault=='short_inventory':
        results['static']['results']['H']['cases']=11;write(execution/'complete.json',dict(results=results))
    elif fault=='changed_model':(baseline/'H/model.zip').write_bytes(b'changed')
    elif fault=='changed_raw':write(batch/'H/endpoint_raw_audit.json',{'changed':True})
    elif fault=='failed_NN':write(batch/'H/static_action_inference.json',dict(all_recorded_actions_match=False))
    else:write(batch/'strict_delivery.json',dict(strict_delivery_gate_passed=False))
    with pytest.raises(ValueError):queue.completed_predecessor(execution,baseline)


def test_live_exact_dependency_never_prepares_or_launches_training(tmp_path,monkeypatch):
    initial=tmp_path/'initial';write(initial/'complete.json',dict(results={'H':{}},gain_scale=100.,status='INITIALIZERS_PREPARED_NOT_NATIVE_TRAINED'))
    identity=tmp_path/'identity.json';write(identity,dict(pid=1,creation_epoch=1,executable='python'))
    owner=SimpleNamespace(poll=lambda:None,kernel=SimpleNamespace(CloseHandle=lambda handle:None),handle=1)
    monkeypatch.setattr(queue,'OwnedProcess',lambda **kwargs:owner)
    monkeypatch.setattr(queue,'source_hashes',lambda:{})
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:(_ for _ in ()).throw(RuntimeError('waiting test boundary')))
    monkeypatch.setattr(queue,'prepare',lambda *args:pytest.fail('Prepared before exact dependency ended'))
    monkeypatch.setattr(queue,'run_course',lambda *args:pytest.fail('Launched while dependency lives'))
    output=tmp_path/'new_batch'
    with pytest.raises(RuntimeError,match='waiting test boundary'):
        queue.run(initial,tmp_path/'baseline',tmp_path/'predecessor',identity,output,tmp_path/'execution')
    assert not output.exists()


def test_source_change_during_wait_prevents_dispatch(predecessor,tmp_path,monkeypatch):
    previous,baseline,_=predecessor
    initial=tmp_path/'initial';write(initial/'complete.json',dict(results={'H':{}},gain_scale=100.,status='INITIALIZERS_PREPARED_NOT_NATIVE_TRAINED'))
    identity=tmp_path/'identity.json';write(identity,dict(pid=1,creation_epoch=1,executable='python'))
    owner=SimpleNamespace(poll=lambda:0,kernel=SimpleNamespace(CloseHandle=lambda handle:None),handle=1)
    monkeypatch.setattr(queue,'OwnedProcess',lambda **kwargs:owner)
    snapshots=iter([{'worker.py':'before'},{'worker.py':'after'}])
    monkeypatch.setattr(queue,'source_hashes',lambda:next(snapshots))
    monkeypatch.setattr(queue,'prepare',lambda *args:pytest.fail('Prepared with changed prospective source'))
    output=tmp_path/'new_batch'
    with pytest.raises(ValueError,match='source or initializer changed'):
        queue.run(initial,baseline,previous,identity,output,tmp_path/'new_execution')
    assert not output.exists()
