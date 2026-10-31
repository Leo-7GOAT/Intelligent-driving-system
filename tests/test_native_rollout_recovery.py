"""Synthetic recovery staging; no fixture is claimed as CARLA/PPO evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

pytestmark = pytest.mark.usefixtures('ample_training_disk')

from navstack.experiments import write_json
from scripts.native_road_learning import registered_jobs
import scripts.prepare_native_recovery as recovery


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint(folder, lineage, steps, adam):
    folder.mkdir(parents=True)
    metadata={'physics_steps':steps,'lineage_control_steps':lineage,
        'charged_control_steps_lower_bound':lineage+9545,'optimizer_steps':adam}
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),
                     ('resume_state.pkl','resume_state_sha256')]:
        payload=(name+str(lineage)).encode()
        (folder/name).write_bytes(payload)
        metadata[key]=hashlib.sha256(payload).hexdigest()
    write_json(folder/'metadata.json',metadata)
    write_json(folder/'config.json',{'scope':'synthetic staging fixture'})
    return metadata


@pytest.fixture
def interrupted(tmp_path,monkeypatch):
    original_root=recovery.ROOT
    parent=tmp_path/'runs/parent'
    for folder in ('source/navstack','source/scripts','calibration','infrastructure'):
        (parent/folder).mkdir(parents=True)
    for directory in ('scripts','navstack'):
        (tmp_path/directory).mkdir()
        for pattern in ('*.py','*.ps1'):
            for source in (original_root/directory).glob(pattern):
                shutil.copy2(source,tmp_path/directory/source.name)
    frozen=parent/'source/navstack/env.py'
    frozen.write_text('# immutable fixture environment\n')
    jobs=[{'name':method+'_rollout'+str(size),'method':method,'gradient_clip_mode':'global',
           'rollout_actions':size} for method,size in [('H0',32),('H0',256),('H',32)]]
    names=[job['name'] for job in jobs]
    prior={}
    for name in names:
        checkpoint(parent/'resume'/name,120320,20160,126)
        prior[name]={'checkpoint':'resume/'+name,'lineage_controls':120320,
            'consumed_controls_lower_bound':129865,'retained_optimizer_steps':752,
            'unknown_unflushed_parent_tail':True,'recovery_chain':['original']}
    checkpoint(parent/'H0_rollout256/checkpoint_000010000',130560,10240,64)
    write_json(parent/'H0_rollout256/config.json',{'scope':'synthetic staging fixture'})
    protocol={'methods':names,'jobs':jobs,'rollout_length_experiment':True,
        'rollout_actions':32,'ppo_epochs':2,'resource_limits':{},'parent_batch':'original',
        'source_hashes':{'navstack/env.py':hashlib.sha256(frozen.read_bytes()).hexdigest()},
        'controlled_budget_per_method':222720,'lineage_targets':dict.fromkeys(names,222720),
        'recovery':prior}
    write_json(parent/'protocol.json',protocol)
    write_json(parent/'interruption_audit.json',{'methods':{
        'H0_rollout32':{'state':'complete'},
        'H0_rollout256':{'state':'interrupted','consumed_controls_lower_bound':11000,
            'unknown_unflushed_tail':True,'checkpoints':[{'folder':'checkpoint_000010000','physics_steps':10240}]},
        'H_rollout32':{'state':'not_started','consumed_controls_lower_bound':0}}})
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    monkeypatch.setattr(recovery,'resources_ok',lambda path:(True,None))
    return parent


def test_recovery_retains_per_job_rollout_and_all_lost_costs(interrupted):
    recovery.prepare(interrupted,7200)
    child=Path(read(interrupted.parent/'native_recovery.json')['batch'])
    protocol=read(child/'protocol.json')
    assert [job['rollout_actions'] for job in registered_jobs(protocol)]==[256,32]
    assert protocol['methods']==['H0_rollout256','H_rollout32']
    assert not (child/'resume/H0_rollout32').exists()
    resumed=protocol['recovery']['H0_rollout256']
    assert resumed['lineage_controls']==130560
    assert resumed['consumed_controls_lower_bound']==140865
    assert resumed['retained_optimizer_steps']==816
    assert resumed['unknown_unflushed_parent_tail'] is True
    assert protocol['method_budgets']=={'H0_rollout256':92160,'H_rollout32':102400}
    for name in protocol['methods']:
        assert protocol['recovery'][name]['recovery_chain']==['original','parent']
    assert (child/'source/navstack/env.py').read_bytes()==(interrupted/'source/navstack/env.py').read_bytes()
    assert read(interrupted/'recovery_registration.json')['child_batch']==child.name


@pytest.mark.parametrize('filename',['model.zip','resume_state.pkl'])
def test_changed_checkpoint_after_prefix_audit_cannot_claim_recovery(interrupted,filename):
    (interrupted/'H0_rollout256/checkpoint_000010000'/filename).write_bytes(b'corrupt after audit')
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):
        recovery.prepare(interrupted,7200)
    assert not (interrupted/'recovery_registration.json').exists()
    assert not (interrupted.parent/'native_recovery.json').exists()


def test_corrupt_staging_copy_cannot_claim_recovery(interrupted,monkeypatch):
    original=recovery.shutil.copytree
    def corrupt(source,destination,*args,**kwargs):
        result=original(source,destination,*args,**kwargs)
        if Path(destination).name=='H0_rollout256':
            (Path(destination)/'model.zip').write_bytes(b'copy corrupted')
        return result
    monkeypatch.setattr(recovery.shutil,'copytree',corrupt)
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):
        recovery.prepare(interrupted,7200)
    assert not (interrupted/'recovery_registration.json').exists()
    assert not (interrupted.parent/'native_recovery.json').exists()
