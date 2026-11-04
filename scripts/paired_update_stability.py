"""Same-checkpoint development branches isolate the constant PPO learning rate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def prepare(parent, checkpoint):
    parent,checkpoint=Path(parent).resolve(),Path(checkpoint).resolve()
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    if not checkpoint.is_relative_to(parent):raise ValueError('Checkpoint must belong to the frozen parent')
    old=read(parent/'protocol.json');job=old['jobs'][0];summary=read(checkpoint/'training_summary.json')
    if job['env']['method']!='B1' or summary['seed']!=11:raise ValueError('This probe is the registered seed11 direct baseline')
    origin=checkpoint.parent
    if not (origin/'completed.json').exists():raise ValueError('Parent must be completed before branching')
    for relative,digest in old['source_hashes'].items():
        if hashlib.sha256((parent/'source'/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen parent source changed')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_update_stability')
    batch.mkdir();shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ['navstack/research.py','scripts/paired_update_stability.py']:
        shutil.copy2(ROOT/relative,batch/'source'/relative)
    start=summary['physics_steps'];branches=[]
    cases=list(old['development_cases'])
    cases += [{'scenario':'empty','seed':seed,'label':f'retention_{distance}m','options':{'diagnostic_route_length':float(distance)}}
              for distance in (6,12,18,23) for seed in (7301,7302)]
    for name,rate in [('original_rate',.0003),('lower_rate',.0001)]:
        run=batch/name;run.mkdir();shutil.copytree(checkpoint,run/'initial_checkpoint')
        for filename in ['ppo_options.json','version.json','resolved_config.json']:
            shutil.copy2(origin/filename,run/filename)
        latest=dict(summary,checkpoint=str(run/'initial_checkpoint'))
        write_json(run/'latest.json',latest)
        branches.append({'name':name,'learning_rate':rate})
    now=time.time()
    protocol={'scope':'Two correlated branches of one existing seed; development stability diagnosis, not independent seeds or retrospective final model selection',
        'created_epoch':now,'deadline_epoch':now+1800,'parent_batch':parent.name,'parent_checkpoint':checkpoint.name,
        'inherited_controls':start,'branches':branches,'milestones':[start+n for n in (50000,100000,150000,200000)],
        'env':job['env'],'seed':job['seed'],'ppo_options':job['ppo_options'],'cases':cases,
        'hypothesis':'Lower constant learning rate retains long-route parking after mixed-task updates; all full-task collision/deadline outcomes remain reported',
        'primary_diagnostic':'23m deterministic parking retention at each of four predeclared milestones; full-task safety/completion also reported at every milestone',
        'resource_limits':{'wall_seconds':1800,'disk_min_GiB':12,'ram_min_GiB':4,'cpu_threads':1,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()},
        'source_hashes':{p.relative_to(batch/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch/'source').rglob('*.py')}}
    write_json(batch/'protocol.json',protocol);write_json(ROOT/'runs/update_stability.json',{'batch':str(batch)})
    print(batch,flush=True)


def run(batch):
    import torch
    from navstack.config import EnvConfig
    from navstack.research import cumulative_train
    from navstack.process_lease import ProcessLease
    from scripts.hierarchy_development import evaluate
    batch=Path(batch).resolve();protocol=read(batch/'protocol.json');torch.set_num_threads(1)
    for relative,digest in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Execute frozen source only')
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():raise RuntimeError('No automatic repeated stability experiment')
        write_json(batch/'started.json',{'epoch':time.time()})
        results=[]
        try:
            # Interleave milestones; each branch restores its own RNG/optimizer
            # and actual vehicle state, including inherited training costs.
            for target in protocol['milestones']:
                for branch in protocol['branches']:
                    healthy,reason=resources_ok(batch)
                    if not healthy or shutil.disk_usage(batch).free<12*1024**3 or time.time()>protocol['deadline_epoch']-120:
                        raise RuntimeError('Resource/deadline guard: '+str(reason))
                    folder=batch/branch['name'];cfg=EnvConfig.from_dict(protocol['env'])
                    saved=cumulative_train(folder,cfg,protocol['seed'],target,protocol['deadline_epoch']-120,
                        ppo_options=branch.get('ppo_options',protocol['ppo_options']),resume_learning_rate=branch['learning_rate'])
                    if saved['reason']!='milestone_reached':raise RuntimeError('Incomplete stability milestone: '+saved['reason'])
                    evaluate(folder,saved['checkpoint'],cfg,protocol['cases'],target)
                    outcome=read(folder/f'validation_{target}.json')
                    row={'branch':branch['name'],'target':target,'actual_physics_steps':saved['physics_steps'],
                        'additional_physics_steps':saved['physics_steps']-protocol['inherited_controls'],
                        'full_task_success':sum(e['success'] for e in outcome['episodes'] if 'label' not in e),
                        'retention':{str(d):sum(e['success'] for e in outcome['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)}}
                    if target == 0 and 'expected_zero_result' in protocol:
                        assert all(row[key] == value for key,value in protocol['expected_zero_result'].items()), 'Exploration-only change altered deterministic zero-update results'
                    results.append(row);write_json(batch/'progress.json',results);print(json.dumps(row),flush=True)
            write_json(batch/'completed.json',{'results':results,'scope':protocol['scope']})
        except BaseException as exc:
            write_json(batch/'failure.json',{'error':repr(exc),'results':results,'epoch':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--checkpoint');p.add_argument('--batch')
    a=p.parse_args();prepare(a.parent,a.checkpoint) if a.parent else run(a.batch)
