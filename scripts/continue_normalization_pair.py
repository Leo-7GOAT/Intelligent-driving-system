"""Continue both completed normalization branches at matched physical budgets."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok
from scripts.train_recovery_demonstrations import read,sha


def prepare(updating, frozen):
    parents=[Path(updating).resolve(),Path(frozen).resolve()]
    pointer=ROOT/'runs/normalization_retention_pair.json'
    if pointer.exists():
        raise RuntimeError('Inspect existing continuation; never duplicate')
    protocols=[read(p/'protocol.json') for p in parents]
    audits=[read(parents[0]/'exploration_verification.json'),
            read(parents[1]/'normalization_verification.json')]
    for p,protocol,audit in zip(parents,protocols,audits,strict=True):
        guard=read(p/'guard_final.json')
        assert audit['complete'] and guard['exit_code']==0 and guard['reason'] is None
        assert not (p/'failure.json').exists() and (p/'completed.json').exists()
        assert audit['endpoints'][-1]['actual_physics_steps']==102400
        for name,digest in protocol['source_hashes'].items():
            assert sha(p/'source'/name)==digest
            if name.startswith('navstack/') and name!='navstack/research.py':
                assert sha(ROOT/name)==digest
    old,new=protocols
    assert all(old[k]==new[k] for k in ('env','seed','cases','milestones'))
    assert new['ppo_options']=={**old['ppo_options'],'freeze_observation_stats':True}
    assert old['initial_checkpoint_hashes'][old['branches'][0]['name']]==new['initial_checkpoint_hashes'][new['branches'][0]['name']]
    assert audits[1]['normalization_audit']['mean_variance_count_exact_at_every_endpoint']
    assert sha(ROOT/'navstack/research.py')==new['source_hashes']['navstack/research.py']
    # Reproduction must cover the complete frozen endpoint before extension.
    receipt_path=ROOT/'runs/prerelease_20261022/native_road_checkout/runs/frozen_normalization_complete_reproduction.json'
    receipt=read(receipt_path)
    expected=sha(parents[1]/'normalization_verification.json')
    assert any(row['sha256']==expected and row['byte_identical'] for row in receipt['reproduced'])
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:
        raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_normalization_retention_pair')
    batch.mkdir()
    branches=[];initial_hashes={};parent_records={}
    for name,parent,protocol,audit in zip(('updating_stats','frozen_stats'),parents,protocols,audits,strict=True):
        origin=parent/protocol['branches'][0]['name']
        evaluation=read(origin/'validation_100000.json')
        checkpoint=origin/Path(evaluation['checkpoint']).name
        summary=read(checkpoint/'training_summary.json')
        assert sha(checkpoint/'model.zip')==evaluation['model_sha256']==audit['endpoints'][-1]['model_sha256']
        assert summary['physics_steps']==102400 and summary['reason']=='milestone_reached' and summary['reload_exact']
        folder=batch/name
        shutil.copytree(checkpoint,folder/'initial_checkpoint')
        for filename in ('ppo_options.json','version.json','resolved_config.json'):
            shutil.copy2(origin/filename,folder/filename)
        assert read(folder/'ppo_options.json')==protocol['ppo_options']
        write_json(folder/'latest.json',{**summary,'checkpoint':str(folder/'initial_checkpoint')})
        hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
        assert hashes=={p.name:sha(p) for p in (folder/'initial_checkpoint').iterdir() if p.is_file()}
        initial_hashes[name]=hashes
        branches.append({'name':name,'learning_rate':.0001,'ppo_options':protocol['ppo_options']})
        parent_records[name]={'batch':parent.name,'branch':origin.name,'checkpoint':checkpoint.name,
                              'final_endpoint':audit['endpoints'][-1],
                              'inherited_cost_ledger':protocol['inherited_costs']}
    source_hashes={}
    for source_folder in ('navstack','scripts','tests'):
        for path in (ROOT/source_folder).rglob('*.py'):
            if '__pycache__' in path.parts:continue
            relative=path.relative_to(ROOT);target=batch/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            source_hashes[relative.as_posix()]=sha(path)
    now=time.time()
    protocol={'scope':'Paired continuation of two correlated supervised-initialized PPO branches; no new independent seed, supervision, teacher query or low-level change',
        'created_epoch':now,'deadline_epoch':now+5400,'parent_records':parent_records,
        'inherited_controls':102400,'branches':branches,'milestones':[200000,300000],
        'env':old['env'],'seed':old['seed'],'cases':old['cases'],'ppo_options':old['ppo_options'],
        'initial_checkpoint_hashes':initial_hashes,'checkpoint_hashes':{},'source_hashes':source_hashes,
        'reproduction_receipt_sha256':sha(receipt_path),
        'hypothesis':'Test whether restored parking and4/10full-task competence persist at200k/300k while the updating-statistics comparator receives the same cumulative interaction budget',
        'selection':'Both fixed100k endpoints; interleave updating/frozen at200k then300k, retain failures and every milestone; never choose best points',
        'cost_accounting':'102400inherited PPO controls per policy plus new actual controls; parent ledgers overlap and must not be naively summed; no extra supervised updates in this continuation',
        'resource_limits':{'wall_seconds':5400,'disk_min_GiB':12,'ram_min_GiB':4,'cpu_workers':1,'gpu_stop_temperature_C':85,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json',protocol)
    write_json(pointer,{'batch':str(batch)})
    print(batch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--updating',required=True);parser.add_argument('--frozen',required=True)
    args=parser.parse_args();prepare(args.updating,args.frozen)
