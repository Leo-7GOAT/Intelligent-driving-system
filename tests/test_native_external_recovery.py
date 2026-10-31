"""Synthetic OS-receipt validation; no real exit or simulator run is claimed."""
from copy import deepcopy
import pytest
from navstack.experiments import write_json
from scripts.audit_native_method_endpoint import sha
from scripts.prepare_native_evaluation_recovery import verified_interruption, evaluation_lineage


def fixture(parent,monkeypatch):
    (parent/'infrastructure').mkdir()
    write_json(parent/'protocol.json',{'carla_rpc_port':2200})
    write_json(parent/'worker_process.json',{'pid':12341})
    write_json(parent/'guard.json',{'pid':12342})
    write_json(parent/'infrastructure/server_start.json',{'owned_by_task':True,'shipping_pid':12344,'launcher_pid':12345})
    files=('protocol.json','worker_process.json','guard.json','infrastructure/server_start.json')
    receipt={'status':'verified_worker_guard_server_absent','reason':'external_process_loss_cause_unknown',
        'observed_epoch':1000.,'process_ids':{'worker_pid':12341,'guard_pid':12342,'shipping_pid':12344,'launcher_pid':12345},
        'protocol_port_span':[2200,2201,2202],'tcp_endpoints':[],'udp_endpoints':[],'related_bridges':[],
        'source_hashes':{name:sha(parent/name) for name in files}}
    write_json(parent/'process_absence_receipt.json',receipt)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.time.time',lambda:1001.)
    monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:False)
    return receipt


def test_verified_external_absence_keeps_unknown_exit_cause_and_has_no_fake_stop_receipt(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    assert verified_interruption(tmp_path)=='external_process_loss_cause_unknown'
    assert not (tmp_path/'guard_final.json').exists()
    assert not (tmp_path/'infrastructure/server_stop.json').exists()


@pytest.mark.parametrize('fault',['stale','occupied','changed','live','identity','ports'])
def test_external_absence_rejects_stale_live_or_changed_records(tmp_path,monkeypatch,fault):
    receipt=fixture(tmp_path,monkeypatch)
    if fault=='stale':receipt['observed_epoch']=0.
    elif fault=='occupied':receipt['tcp_endpoints']=[{'port':2200}]
    elif fault=='changed':write_json(tmp_path/'worker_process.json',{'pid':12399})
    elif fault=='live':monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:True)
    elif fault=='identity':receipt['process_ids']['worker_pid']=12399
    else:receipt['protocol_port_span']=[2000,2001,2002]
    write_json(tmp_path/'process_absence_receipt.json',receipt)
    with pytest.raises(ValueError):verified_interruption(tmp_path)


@pytest.mark.parametrize('fault',[None,'wrong_error','exit_zero','live','wrong_worker','wrong_server','unclean'])
def test_completed_training_disk_stop_requires_exact_failure_and_owned_absent_processes(tmp_path,monkeypatch,fault):
    fixture(tmp_path,monkeypatch)
    final={'reason':None,'exit_code':1,'worker_pid':12341}
    stop={'status':'owned_empty_server_stopped','shipping_pid':12344,'post_stop_tcp':[],'post_stop_udp':[]}
    failure={'error':"RuntimeError('Disk below12GiB')"}
    if fault=='wrong_error':failure['error']="RuntimeError('arbitrary learner bug')"
    elif fault=='exit_zero':final['exit_code']=0
    elif fault=='live':monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:True)
    elif fault=='wrong_worker':final['worker_pid']=12399
    elif fault=='wrong_server':stop['shipping_pid']=12399
    elif fault=='unclean':stop['post_stop_tcp']=[{'port':2200}]
    write_json(tmp_path/'guard_final.json',final)
    write_json(tmp_path/'failure.json',failure)
    write_json(tmp_path/'infrastructure/server_stop.json',stop)
    if fault is None:
        assert verified_interruption(tmp_path)=='worker_disk_below_12GiB'
    else:
        with pytest.raises(ValueError):verified_interruption(tmp_path)


def test_frozen_replay_cannot_drop_unknown_tail_lost_controls_or_retained_updates():
    prior={'unknown_unflushed_parent_tail':True,'retained_optimizer_steps':320,
           'recovery_chain':['original','first_recovery']}
    before=deepcopy(prior)
    summary={'lineage_control_steps':20710,'charged_control_steps_lower_bound':21640,'optimizer_steps':340}
    replay=evaluation_lineage(prior,summary,'last_training','checkpoint_final','B1')
    assert replay['unknown_unflushed_parent_tail'] is True
    assert replay['consumed_controls_lower_bound']-replay['lineage_controls']==930
    assert replay['retained_optimizer_steps']==660
    assert replay['recovery_chain']==['original','first_recovery','last_training']
    assert prior==before
    repeated=evaluation_lineage(replay,{**summary,'optimizer_steps':0},'last_training','final','B1')
    assert repeated['recovery_chain']==replay['recovery_chain']
    assert repeated['retained_optimizer_steps']==660 and repeated['unknown_unflushed_parent_tail']


@pytest.mark.parametrize('fault',[None,'other_error','live','unclean'])
def test_actor_cleanup_interruption_requires_exact_error_and_absent_owned_lifecycle(tmp_path,monkeypatch,fault):
    fixture(tmp_path,monkeypatch)
    write_json(tmp_path/'guard_final.json',{'reason':None,'exit_code':1,'worker_pid':12341})
    stop={'status':'owned_empty_server_stopped','shipping_pid':12344,'post_stop_tcp':[],'post_stop_udp':[]}
    error="InfrastructureFailure('RuntimeError: Owned actor cleanup incomplete: destroy actor 57: destroy returned false')"
    if fault=='other_error':error="InfrastructureFailure('arbitrary bridge failure')"
    if fault=='live':monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:True)
    if fault=='unclean':stop['post_stop_udp']=[{'port':2201}]
    write_json(tmp_path/'failure.json',{'error':error});write_json(tmp_path/'infrastructure/server_stop.json',stop)
    if fault is None:assert verified_interruption(tmp_path)=='owned_actor_cleanup_failure'
    else:
        with pytest.raises(ValueError):verified_interruption(tmp_path)


def test_interrupted_evaluation_cost_is_not_lost_on_zero_PPO_replay():
    summary={'lineage_control_steps':0,'charged_control_steps_lower_bound':16717,'optimizer_steps':0}
    recovered=evaluation_lineage({},summary,'failed_eval','checkpoint_000000000','B1',5100)
    assert recovered['consumed_controls_lower_bound']==21817
    assert recovered['lineage_controls']==recovered['retained_optimizer_steps']==0


@pytest.mark.parametrize('fault',[None,'occupied','wrong_port','outside','live'])
def test_foreign_guard_recovery_requires_empty_recorded_server_and_ended_owned_lifecycle(tmp_path,monkeypatch,fault):
    fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.recorded_lifecycle_gone',lambda *args:fault!='live')
    write_json(tmp_path/'guard_final.json',{'reason':None,'exit_code':1,'worker_pid':12341,'epoch':1000.})
    write_json(tmp_path/'failure.json',{'error':"InfrastructureFailure('RuntimeError: Foreign actors appeared; refusing to tick')"})
    probe=tmp_path/'infrastructure/stop_probe.json'
    if fault=='outside':probe=tmp_path/'outside.json'
    write_json(probe,{'status':'CONNECTED','actors':[{'id':99}] if fault=='occupied' else [],'client_version':'0.9.15','server_version':'0.9.15',
        'port':2201 if fault=='wrong_port' else 2200,'observed_epoch':1000.1})
    write_json(tmp_path/'infrastructure/server_stop.json',{'status':'owned_empty_server_stopped','shipping_pid':12344,'post_stop_tcp':[],
        'post_stop_udp':[],'epoch':1000.2,'probe':str(probe)})
    if fault is None:assert verified_interruption(tmp_path)=='foreign_actor_guard_fresh_server_only_cause_unknown'
    else:
        with pytest.raises(ValueError):verified_interruption(tmp_path)
