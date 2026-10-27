import json
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_native_measured_bc as module
from navstack.experiments import signature
from test_measured_demo import fixture


def data_fixture(tmp_path, monkeypatch):
    live, view, _ = fixture()
    config = view.config
    live.close(); view.close()
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps({'env':config.to_dict(),'signature':signature(config)}))
    donor = tmp_path/'runs'/'teacher'; donor.mkdir(parents=True)
    (donor/'protocol.json').write_text('{}')
    audit = {'trace_sha256':'trace','evaluation_controls_verified':6}
    (donor/'raw_audit.json').write_text(json.dumps(audit))
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'require_finished_parent',lambda path: None)
    monkeypatch.setattr(module,'audit',lambda path: audit)
    for name in ('navstack/measured_demo.py','scripts/build_native_measured_views.py'):
        path=tmp_path/name; path.parent.mkdir(exist_ok=True); path.write_text('source')
    arrays={'observations':np.zeros((6,4701),np.float32),'actions':np.zeros((6,2),np.float32),
            'returns':np.arange(6,dtype=np.float32),'train_mask':np.arange(6)%5!=0,'groups':np.zeros(6,np.int64)}
    views=tmp_path/'views'; views.mkdir()
    dataset=views/'measured_demonstrations.npz'; np.savez_compressed(dataset,**arrays)
    sources=[{'batch':'teacher','controls':6,'trace_sha256':'trace',
              'raw_audit_sha256':module.sha(donor/'raw_audit.json'),'protocol_sha256':module.sha(donor/'protocol.json')}]
    manifest={'reference_config_sha256':module.sha(reference),'sources':sources,
              'view_source_sha256':module.sha(tmp_path/'navstack/measured_demo.py'),
              'script_sha256':module.sha(tmp_path/'scripts/build_native_measured_views.py'),
              'new_controls':0,'supervised_optimizer_steps':0,'ppo_updates':0,
              'dataset_sha256':module.sha(dataset),'teacher_unique_controls':6,
              'cases':[{'group':0,'controls':6}]}
    (views/'registration.json').write_text(json.dumps(manifest))
    (views/'manifest.json').write_text(json.dumps(manifest))
    return views,reference,config,donor,arrays,manifest


def test_registered_views_load_exact_arrays_without_execution(tmp_path,monkeypatch):
    views,reference,config,donor,arrays,_=data_fixture(tmp_path,monkeypatch)
    loaded,_=module.load_registered_views(views,reference,config,donor)
    for name in arrays: np.testing.assert_array_equal(loaded[name],arrays[name])


@pytest.mark.parametrize('fault',['reference','dataset','teacher','action','holdout'])
def test_registered_views_reject_changed_evidence_before_fitting(tmp_path,monkeypatch,fault):
    views,reference,config,donor,arrays,manifest=data_fixture(tmp_path,monkeypatch)
    if fault=='reference': reference.write_text(reference.read_text()+' ')
    elif fault=='dataset': (views/'measured_demonstrations.npz').write_bytes(b'changed')
    elif fault=='teacher': (donor/'protocol.json').write_text('{"changed":true}')
    else:
        if fault=='action': arrays['actions'][0,0]=1.1
        else: arrays['train_mask'][0]=True
        np.savez_compressed(views/'measured_demonstrations.npz',**arrays)
        manifest['dataset_sha256']=module.sha(views/'measured_demonstrations.npz')
        (views/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): module.load_registered_views(views,reference,config,donor)
