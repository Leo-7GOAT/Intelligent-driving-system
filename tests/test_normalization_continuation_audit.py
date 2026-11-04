import gzip
import json
from pathlib import Path
import shutil
import time

import pytest

from navstack.config import EnvConfig,PlannerConfig
from navstack.experiments import write_json
from navstack.research import cumulative_train
from scripts.hierarchy_development import evaluate
from scripts.train_recovery_demonstrations import sha
from scripts.verify_normalization_continuation import analyze


def test_real_resumed_pair_audit_and_reject_wrong_executed_command(tmp_path):
    config=EnvConfig(method='B1',scenario='empty',deadline=1,
                     planner=PlannerConfig(max_seconds=None))
    options={'use_sde':True,'log_std_init':-4,'sde_sample_freq':1}
    original=cumulative_train(tmp_path/'parent',config,11,64,time.time()+60,
                              rollout_physics=64,epochs=2,ppo_options=options)
    checkpoint=Path(original['checkpoint'])
    batch=tmp_path/'pair';batch.mkdir()
    cases=[{'scenario':'empty','seed':7301},{'scenario':'empty','seed':7302}]
    branches=[];hashes={};progress=[]
    for name,freeze in [('updating',False),('frozen',True)]:
        folder=batch/name
        shutil.copytree(checkpoint,folder/'initial_checkpoint')
        branch_options={**options,**({'freeze_observation_stats':True} if freeze else {})}
        write_json(folder/'ppo_options.json',branch_options)
        write_json(folder/'latest.json',{**original,'checkpoint':str(folder/'initial_checkpoint')})
        hashes[name]={p.name:sha(p) for p in (folder/'initial_checkpoint').iterdir()}
        branches.append({'name':name,'ppo_options':branch_options})
        saved=cumulative_train(folder,config,11,128,time.time()+60,
                               rollout_physics=64,epochs=2,ppo_options=branch_options)
        evaluate(folder,saved['checkpoint'],EnvConfig.from_dict(config.to_dict()),cases,128)
        evaluation=json.loads((folder/'validation_128.json').read_text(encoding='utf-8'))
        progress.append({'branch':name,'target':128,'actual_physics_steps':128,
                         'additional_physics_steps':64,
                         'full_task_success':sum(e['success'] for e in evaluation['episodes']),
                         'retention':{str(d):0 for d in (6,12,18,23)}})
    write_json(batch/'protocol.json',{'scope':'Real physical integration test fixture',
        'source_hashes':{},'inherited_controls':64,'branches':branches,'milestones':[128],
        'initial_checkpoint_hashes':hashes,'cases':cases,'env':config.to_dict()})
    write_json(batch/'progress.json',progress)
    write_json(batch/'completed.json',{'results':progress})
    write_json(batch/'guard_final.json',{'exit_code':0,'reason':None})
    analyze(batch)
    result=json.loads((batch/'continuation_verification.json').read_text(encoding='utf-8'))
    assert result['complete']
    for row in result['branches']:
        assert row['new_training_controls']==64 and row['new_optimizer_steps']>0
        assert row['normalization_checkpoints_checked']
        assert row['endpoints'][0]['evaluation_controls']>0
    path=next((batch/'updating').glob('transitions_*.jsonl.gz'))
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        records=[json.loads(line) for line in stream]
    records[0]['trace'][0]['requested_command']['acceleration']+=.5
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        stream.write('\n'.join(json.dumps(row) for row in records)+'\n')
    with pytest.raises(AssertionError):
        analyze(batch)
