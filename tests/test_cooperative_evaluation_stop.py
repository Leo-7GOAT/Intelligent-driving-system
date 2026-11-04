import json
import pytest
from scripts.audit_native_partial_evaluation import audit


@pytest.mark.parametrize('change',['different_stop','successful_exit','live_process','foreign_server'])
def test_cooperative_prefix_rejects_unverified_stop(tmp_path,monkeypatch,change):
    import scripts.prepare_native_evaluation_recovery as lifecycle
    monkeypatch.setattr(lifecycle,'recorded_lifecycle_gone',lambda *a:change!='live_process')
    values={
        'STOP':{'epoch':1.,'reason':'development rejection'},
        'development_stop_registration.json':{'epoch':1.,'reason':'development rejection'},
        'guard_final.json':{'exit_code':1,'reason':None,'worker_pid':1,'epoch':2.},
        'worker_process.json':{'pid':1},'guard.json':{'pid':2},
        'infrastructure/server_start.json':{'shipping_pid':3,'launcher_pid':4,'owned_by_task':True},
        'infrastructure/server_stop.json':{'shipping_pid':3,'status':'owned_empty_server_stopped',
            'post_stop_tcp':[],'post_stop_udp':[],'epoch':3.},
        'failure.json':{'error':"RuntimeError('Diagnostic deadline/STOP reached')"}}
    if change=='different_stop':values['STOP']['epoch']=1.5
    if change=='successful_exit':values['guard_final.json']['exit_code']=0
    if change=='foreign_server':values['infrastructure/server_start.json']['owned_by_task']=False
    for name,value in values.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='Require closed cooperative'):
        audit(tmp_path,'H',True)
