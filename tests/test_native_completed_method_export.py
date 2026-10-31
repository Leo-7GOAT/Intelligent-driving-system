import hashlib
import json
from pathlib import Path
import pytest
from scripts.native_completed_method_export import completed_method_profile,validate_exported_inventory


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def sample(tmp_path):
    batch=tmp_path/'batch';checkpoint=batch/'H/checkpoint';checkpoint.mkdir(parents=True)
    for name in ('model.zip','normalization.pkl'):(checkpoint/name).write_bytes(name.encode())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata={'physics_steps':10,'optimizer_steps':2,'lineage_control_steps':10,
        'model_sha256':digest(checkpoint/'model.zip'),'normalization_sha256':digest(checkpoint/'normalization.pkl')}
    write(checkpoint/'metadata.json',metadata)
    (batch/'H/model.zip').write_bytes((checkpoint/'model.zip').read_bytes())
    write(batch/'H/latest_checkpoint.json',{'folder':'checkpoint'})
    write(batch/'H/summary.json',{'physics_steps':10,'optimizer_steps':2,'lineage_control_steps':10,'reload_exact':True,'stopped_reason':None})
    write(batch/'H/evaluation.json',{'episodes':[{'case_id':0,'success':True},{'case_id':1,'success':False}]})
    write(batch/'H0/partial.json',{'readable_controls':3,'state':'interrupted'})
    write(batch/'guard_final.json',{'exit_code':1,'reason':'retained failure'})
    write(batch/'launcher_failure.json',{'status':'LAUNCH_OR_SUPERVISION_FAILURE'})
    write(batch/'infrastructure/server_stop.json',{'status':'owned_empty_server_stopped'})
    write(batch/'interruption_audit.json',{'methods':{'H':{'state':'complete'},'H0':{'state':'interrupted'}}})
    protocol={'methods':['H','H0'],'native_case_manifest':{'cases':[{},{}]}}
    return batch,protocol


def profile(batch,protocol):return completed_method_profile(batch,protocol,'H',check_processes=False)


def manifest(batch,p):
    return {'batch':'batch','schema_version':2,'methods':['H'],'completed_method_export':p,
        'files':{'batch/H0/partial.json':{}},'retained_incomplete_method_files':{'H0':['batch/H0/partial.json']}}


def test_interrupted_batch_keeps_failure_all_sibling_bytes_and_failed_target_cases(tmp_path):
    batch,protocol=sample(tmp_path);p=profile(batch,protocol)
    assert p['batch_complete'] is False and p['completed_methods']==['H'] and p['incomplete_methods']==['H0']
    assert 'H0/partial.json' in p['incomplete_file_sha256']['H0']
    assert validate_exported_inventory(batch,protocol,manifest(batch,p)) is True


@pytest.mark.parametrize('change',['normal_guard','complete_sibling','missing_case','failed_cleanup','missing_failure','incomplete_target'])
def test_export_cannot_hide_successful_siblings_partial_target_or_live_cleanup(change,tmp_path):
    batch,protocol=sample(tmp_path)
    if change=='normal_guard':write(batch/'guard_final.json',{'exit_code':0,'reason':None})
    elif change=='complete_sibling':write(batch/'H0/evaluation.json',{'episodes':[]})
    elif change=='missing_case':write(batch/'H/evaluation.json',{'episodes':[{'case_id':0,'success':True}]})
    elif change=='failed_cleanup':write(batch/'infrastructure/server_stop.json',{'status':'failed'})
    elif change=='missing_failure':(batch/'launcher_failure.json').unlink()
    else:write(batch/'H/summary.json',{'physics_steps':10,'optimizer_steps':2,'lineage_control_steps':10,'reload_exact':False})
    with pytest.raises(ValueError):profile(batch,protocol)


def test_changed_sibling_raw_prefix_invalidates_completed_method_capsule(tmp_path):
    batch,protocol=sample(tmp_path);p=profile(batch,protocol)
    write(batch/'H0/partial.json',{'readable_controls':1})
    with pytest.raises(ValueError,match='evidence changed'):
        validate_exported_inventory(batch,protocol,manifest(batch,p))


def test_method_mismatch_cannot_be_silently_accepted_by_normal_archive_reproducer(tmp_path):
    batch,protocol=sample(tmp_path)
    with pytest.raises(ValueError,match='Registered methods'):
        validate_exported_inventory(batch,protocol,{'methods':['H']})


def test_normal_full_method_inventory_stays_compatible(tmp_path):
    batch,protocol=sample(tmp_path)
    assert validate_exported_inventory(batch,protocol,{'methods':['H','H0']}) is False
