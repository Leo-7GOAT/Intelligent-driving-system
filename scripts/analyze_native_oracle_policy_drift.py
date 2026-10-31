"""Compare initial/final B1 on provenance-bound TRAIN-only oracle observations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.audit_native_method_endpoint import read,sha


def errors(predicted,labels):
    residual=np.asarray(predicted,dtype=np.float64)-labels
    return {'rows':len(labels),'MSE':float(np.square(residual).mean()),
        'acceleration_MSE':float(np.square(residual[:,0]).mean()),
        'steering_MSE':float(np.square(residual[:,1]).mean())}


def analyze(batch,oracle,output):
    batch,oracle,output=map(Path,(batch,oracle,output))
    if output.exists():raise FileExistsError('Preserve prior diagnostics')
    folder=batch/'B1';manifest=read(oracle/'manifest.json');raw=read(folder/'endpoint_raw_audit.json')
    frozen=(batch/'source').resolve();protocol=read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(frozen/name)!=digest:
            raise ValueError('Frozen policy/observation source changed')
    sys.path.insert(0,str(frozen))
    from navstack.independent_clip_ppo import load_ppo
    for name,module in list(sys.modules.items()):
        if name.startswith('navstack.') and getattr(module,'__file__',None):
            relative='navstack/'+Path(module.__file__).name
            if relative in protocol['source_hashes'] and sha(Path(module.__file__))!=protocol['source_hashes'][relative]:
                raise ValueError('Run in a fresh process with the recorded policy implementation')
    if (manifest['batch']!=batch.name or manifest['raw_audit_sha256']!=sha(folder/'endpoint_raw_audit.json')
        or manifest['coverage_profile']!='first_episode_full_trajectory_stride80_v1'
        or any(manifest[k] for k in ('new_controls','PPO_updates','supervised_updates'))):
        raise ValueError('Require the explicit training-only full-trajectory oracle profile')
    for name,key in [('corrective_training_queries.npz','dataset_sha256'),('queries.jsonl','query_log_sha256'),
                     ('executed_query_script.py','script_sha256'),('executed_history_script.py','history_source_sha256')]:
        if sha(oracle/name)!=manifest[key]:raise ValueError('Oracle provenance differs:'+name)
    if raw['protocol_sha256']!=sha(batch/'protocol.json'):raise ValueError('Raw source binding differs')
    records=[json.loads(line) for line in (oracle/'queries.jsonl').read_text().splitlines()]
    available=[r for r in records if r['label_available']]
    with np.load(oracle/'corrective_training_queries.npz',allow_pickle=False) as data:
        observations=data['observations'];labels=data['actions'].astype(np.float64)
    if len(available)!=manifest['available_labels'] or len(labels)!=len(available):raise ValueError('Query row count differs')
    np.testing.assert_array_equal(labels,np.asarray([r['label'] for r in available],dtype=np.float32).astype(np.float64))
    models=[];normalizers=[]
    for path in (batch/'resume/B1',folder):
        models.append(load_ppo(path/'model.zip',device='cpu'))
        with (path/'normalization.pkl').open('rb') as stream:normalizers.append(pickle.load(stream))
        normalizers[-1].training=False
    for field in ('mean','var','count'):
        np.testing.assert_array_equal(getattr(normalizers[0].obs_rms,field),getattr(normalizers[1].obs_rms,field))
    normalized=[norm.normalize_obs(observations) for norm in normalizers]
    np.testing.assert_array_equal(*normalized)
    if models[0].num_timesteps or models[0]._n_updates or models[0].policy.optimizer.state:
        raise ValueError('Initial actor is not zero-PPO')
    torch.set_num_threads(1)
    predictions=[model.predict(obs,deterministic=True)[0] for model,obs in zip(models,normalized)]
    scopes={'all':np.ones(len(labels),dtype=bool)}
    for mode in sorted({r['oracle_mode'] for r in available}):scopes['mode_'+mode]=np.asarray([r['oracle_mode']==mode for r in available])
    for condition in sorted({r['condition'] for r in available}):scopes['condition_'+str(condition)]=np.asarray([r['condition']==condition for r in available])
    metrics={name:{kind:errors(prediction[mask],labels[mask]) for kind,prediction in zip(('initial','trained'),predictions)} for name,mask in scopes.items()}
    result={'scope':'Offline initial/final B1 disagreement with TRAIN-only counterfactual controls;not driving,policy fitting or unseen-performance proof',
        'batch':batch.name,'queries':len(records),'available_labels':len(labels),
        'audit_script_sha256':sha(Path(__file__)),'policy_implementation':'Exact hash-verified batch/source',
        'rejections':dict(Counter(r.get('rejection') for r in records if not r['label_available'])),
        'selection':manifest['selection'],'oracle_dataset_sha256':manifest['dataset_sha256'],
        'oracle_manifest_sha256':sha(oracle/'manifest.json'),'raw_audit_sha256':manifest['raw_audit_sha256'],
        'initial_model_sha256':sha(batch/'resume/B1/model.zip'),'final_model_sha256':sha(folder/'model.zip'),
        'normalization_statistics_and_inputs_exactly_equal':True,'metrics':metrics,
        'new_controls':0,'PPO_updates':0,'supervised_updates':0,
        'limits':'First completed training episodes can precede later drift. Oracle labels are counterfactual,including waits/brakes;label disagreement does not prove driving quality. No validation observations are used.','research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}));print(json.dumps(metrics['all']));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('batch','oracle','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();analyze(a.batch,a.oracle,a.output)
