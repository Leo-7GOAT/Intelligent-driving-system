import gzip
import hashlib
import json

import pytest
from navstack.experiments import write_json
from scripts.audit_interrupted_native import frozen_evaluation_prefix


def fixture(tmp_path):
    folder=tmp_path/'B1';checkpoint=folder/'checkpoint_000000000';checkpoint.mkdir(parents=True)
    metadata={'physics_steps':0,'actions':0,'optimizer_steps':0}
    for filename,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        payload=filename.encode();(checkpoint/filename).write_bytes(payload)
        metadata[key]=hashlib.sha256(payload).hexdigest()
        if filename!='resume_state.pkl':(folder/filename).write_bytes(payload)
    for where in (folder,checkpoint):write_json(where/'config.json',{'env':{'method':'B1'}})
    write_json(checkpoint/'metadata.json',metadata)
    write_json(folder/'latest_checkpoint.json',{'folder':checkpoint.name})
    write_json(folder/'summary.json',{'physics_steps':0,'actions':0,'optimizer_steps':0,'reload_exact':True,'stopped_reason':None})
    (folder/'optimization.jsonl').touch()
    with gzip.open(folder/'transitions_0.jsonl.gz','wt',encoding='utf-8'):pass
    with gzip.open(folder/'evaluation_trace.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps({'info':{'trace':[{},{}]}})+'\n')
    protocol={'evaluation_only':True,'controlled_budget_per_method':0,'method_budgets':{'B1':0}}
    return folder,checkpoint,protocol


def test_frozen_evaluation_can_be_audited_without_fabricating_training_heartbeat(tmp_path):
    folder,checkpoint,protocol=fixture(tmp_path)
    result=frozen_evaluation_prefix(folder,protocol)
    assert result['readable_trace_controls']==0 and result['evaluation_controls_readable_lower_bound']==2
    assert not result['unknown_unflushed_tail'] and not (folder/'heartbeat.json').exists()


@pytest.mark.parametrize('fault',['budget','learning','checkpoint','optimizer','training_trace'])
def test_frozen_prefix_rejects_learning_or_changed_inputs(tmp_path,fault):
    folder,checkpoint,protocol=fixture(tmp_path)
    if fault=='budget':protocol['controlled_budget_per_method']=100
    elif fault=='learning':write_json(folder/'summary.json',{'physics_steps':1,'actions':0,'optimizer_steps':0,'reload_exact':True})
    elif fault=='checkpoint':(checkpoint/'model.zip').write_bytes(b'changed')
    elif fault=='optimizer':(folder/'optimization.jsonl').write_text('{}\n')
    else:
        with gzip.open(folder/'transitions_0.jsonl.gz','wt') as stream:stream.write('{}\n')
    with pytest.raises(ValueError):frozen_evaluation_prefix(folder,protocol)
