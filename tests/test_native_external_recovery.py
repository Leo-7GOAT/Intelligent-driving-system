"""Synthetic OS-receipt validation; no real exit or simulator run is claimed."""
from copy import deepcopy
import pytest
from navstack.experiments import write_json
from scripts.audit_native_method_endpoint import sha
from scripts.prepare_native_evaluation_recovery import verified_interruption


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
