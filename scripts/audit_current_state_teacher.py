"""Actual four-case competence check of memoryless teacher labels, not PPO."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.experiments import load_run,write_json,jsonable
from navstack.imitation import collect_recovery_episode
from navstack.research import resources_ok


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();parent=Path(a.parent).resolve();output=Path(a.output).resolve()
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False)
    source={x.relative_to(ROOT).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'navstack').glob('*.py')}
    started=time.time()
    write_json(output/'protocol.json',{'scope':__doc__,'source_hashes':source,'seed':8301,
        'cases':['empty','static','narrow','dynamic'],'teacher_probability':1.,'teacher_mode':'current_state',
        'wall_seconds':900,'created_epoch':started,'no_policy_learning':True,
        'fairness':'Teacher replans each control and initializes MPC anew; extra computation is recorded, not hidden in a learned policy.'})
    import torch
    torch.set_num_threads(1)
    model,norm=load_run(parent/'initial')
    rows=[]
    try:
        for scenario in ('empty','static','narrow','dynamic'):
            cfg=EnvConfig.from_dict(protocol['env']);cfg.scenario=scenario;cfg.split='train'
            with gzip.open(output/(scenario+'_trace.jsonl.gz'),'wt',encoding='utf-8') as stream:
                def record(row):
                    if row['teacher_query']['control']%20==0:
                        ok,reason=resources_ok(output)
                        if not ok or time.time()-started>900:raise RuntimeError('Audit guard: '+str(reason))
                        stream.flush()
                    stream.write(json.dumps(row,default=jsonable)+'\n')
                sample=collect_recovery_episode(cfg,model,norm,8301,1.,8301,callback=record,teacher_mode='current_state')
            np.savez_compressed(output/(scenario+'_examples.npz'),observations=sample['observations'],actions=sample['actions'],returns=sample['returns'])
            row={'scenario':scenario,'seed':8301,'teacher_queries':sample['teacher_queries'],
                'teacher_decisions':sample['teacher_decisions'],'teacher_query_seconds':sample['teacher_query_seconds'],
                'teacher_control_failures':sample['teacher_control_failures'],'metrics':sample['metrics'],'initial':sample['initial']}
            rows.append(row);write_json(output/'progress.json',rows)
            print(json.dumps({'scenario':scenario,'success':row['metrics']['success'],
                'reason':row['metrics']['termination_reason'],'controls':row['teacher_queries']}),flush=True)
        for name,digest in source.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        write_json(output/'results.json',{'scope':__doc__,'episodes':rows,'actual_teacher_controls':sum(r['teacher_queries'] for r in rows),
            'source_unchanged':True,'ppo_updates':0})
    except BaseException as exc:
        write_json(output/'failure.json',{'error':repr(exc),'epoch':time.time()});raise
    finally:norm.close()


if __name__=='__main__':main()
