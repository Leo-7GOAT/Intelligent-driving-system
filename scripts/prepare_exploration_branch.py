"""One fixed exploration-scale contrast after audited supervised-PPO collapse."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_recovery_demonstrations import read, sha
from navstack.experiments import write_json
from navstack.research import resources_ok


def prepare(parent):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    parent = Path(parent).resolve()
    pointer = ROOT/'runs/reduced_exploration.json'
    if pointer.exists():
        raise RuntimeError('Inspect existing exploration branch; no duplicate')
    old, complete = read(parent/'protocol.json'), read(parent/'completed.json')
    audit, guard = read(parent/'analysis/verified_summary.json'), read(parent/'guard_final.json')
    factorial = read(parent/'normalization_25k_factorial/verification.json')
    assert audit['complete'] and guard['exit_code']==0 and guard['reason'] is None
    assert old['ppo_milestones']==[0,25000,100000] and not old['rounds']
    assert factorial['optimizer_updates']==0 and factorial['teacher_queries']==0
    assert len(factorial['groups'])==4
    zero = complete['results'][0]
    assert zero['target']==zero['actual_ppo_controls']==zero['ppo_optimizer_steps']==0
    start = parent/'B1'/Path(zero['checkpoint']).name
    assert sha(start/'model.zip') == read(parent/'B1/validation_0.json')['model_sha256']
    assert old['ppo_options']['log_std_init']==-2
    for name,digest in old['source_hashes'].items():
        assert sha(parent/'source'/name)==digest
    # Dynamics and policy definitions must remain exactly comparable.
    for name,digest in old['source_hashes'].items():
        if name.startswith('navstack/'):
            assert sha(ROOT/name)==digest
    healthy,reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:
        raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_reduced_exploration')
    batch.mkdir()
    name='reduced_exploration'
    folder=batch/name
    folder.mkdir()
    checkpoint=folder/'initial_checkpoint'
    shutil.copytree(start,checkpoint)
    for filename in ('ppo_options.json','resolved_config.json','version.json'):
        shutil.copy2(parent/'B1'/filename,folder/filename)
    original=PPO.load(start/'model.zip',device='cpu')
    changed=PPO.load(start/'model.zip',device='cpu')
    assert original.use_sde and original._n_updates==0 and not original.policy.optimizer.state
    assert torch.all(original.policy.log_std == -2)
    with torch.no_grad():
        changed.policy.log_std.fill_(-4)
    changed.policy.log_std_init=-4
    changed.policy_kwargs['log_std_init']=-4
    for key,value in original.policy.state_dict().items():
        if key != 'log_std':
            assert torch.equal(value,changed.policy.state_dict()[key])
    assert not changed.policy.optimizer.state
    # Exact deterministic agreement on raw network inputs; only sampling changes.
    inputs=torch.linspace(-1,1,605)[None].repeat(8,1)
    with torch.no_grad():
        assert torch.equal(original.policy.get_distribution(inputs).distribution.mean,
                           changed.policy.get_distribution(inputs).distribution.mean)
    changed.save(checkpoint/'model.zip')
    source_hashes={}
    for source_folder in ('navstack','scripts','tests'):
        for path in (ROOT/source_folder).rglob('*.py'):
            if '__pycache__' in path.parts: continue
            relative=path.relative_to(ROOT);target=batch/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            source_hashes[relative.as_posix()]=sha(path)
    baseline_hashes={p.name:sha(p) for p in start.iterdir() if p.is_file()}
    branch_hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
    assert all(branch_hashes[k]==v for k,v in baseline_hashes.items() if k!='model.zip')
    options={**old['ppo_options'],'log_std_init':-4}
    write_json(folder/'ppo_options.json',options)
    resolved=read(folder/'resolved_config.json')
    resolved['intervention']='Only initial gSDE log_std -2 -> -4; optimizer/RNG/vehicle state and normalization restored unchanged'
    write_json(folder/'resolved_config.json',resolved)
    saved=read(checkpoint/'training_summary.json')
    write_json(folder/'latest.json',{**saved,'checkpoint':str(checkpoint)})
    now=time.time()
    protocol={'scope':'One correlated development exploration-scale branch; inherited supervision and completed comparator costs retained, not pure RL or independent seeds',
        'created_epoch':now,'deadline_epoch':now+3600,'parent_batch':parent.name,
        'parent_audit_sha256':sha(parent/'analysis/verified_summary.json'),
        'factorial_audit_sha256':sha(parent/'normalization_25k_factorial/verification.json'),
        'inherited_controls':0,'branches':[{'name':name,'learning_rate':.0001}],
        'milestones':[0,25000,100000],'env':old['env'],'seed':old['seed'],'cases':old['cases'],'ppo_options':options,
        'hypothesis':'Reduce initial gSDE actuator-scale noise after measured standard deviation near1.02 despite log_std=-2; test parking retention and full-task learning at fixed budgets',
        'intervention':'Only log_std tensor starts at-4 instead of-2, reducing conditional standard deviation by exp(-2). It remains trainable. All other policy tensors, optimizer/RNG/environment/normalization checkpoint files match. Same PPO objective, LR, reward, physics and cases.',
        'selection':'Original final supervised endpoint; retain0/25k/100k, never select a better intermediate model',
        'expected_zero_result':{k:zero[k] for k in ('full_task_success','retention')},
        'inherited_costs':{**old['inherited_costs'],'completed_comparator_ppo_controls':complete['final_ppo_controls'],
                           'normalization_factorial_controls':factorial['actual_controls']},
        'checkpoint_hashes':baseline_hashes,'initial_checkpoint_hashes':{name:branch_hashes},'source_hashes':source_hashes,
        'resource_limits':{'wall_seconds':3600,'disk_min_GiB':12,'ram_min_GiB':4,'cpu_workers':1,'gpu_stop_temperature_C':85,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json',protocol)
    write_json(batch/'initial_pairing_audit.json',{'only_log_std_tensor_changed':True,
        'optimizer_empty_and_equal':True,'all_other_checkpoint_files_equal':True,
        'deterministic_mean_exact':True,'old_log_std':-2,'new_log_std':-4})
    write_json(pointer,{'batch':str(batch)})
    print(batch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True)
    prepare(parser.parse_args().parent)
