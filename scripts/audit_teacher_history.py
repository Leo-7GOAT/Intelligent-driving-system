"""Physically replay fixed collected actions; audit fresh vs held-reference labels.

Fresh teacher commands are queries only and never execute. This diagnoses
supervision, not a better teacher, a learned result or independent confirmation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();batch=Path(args.batch).resolve();output=Path(args.output).resolve()
    sys.path.insert(0,str(batch/'source'))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.imitation import RecoveryTeacher
    from navstack.experiments import write_json,jsonable
    from navstack.research import resources_ok
    protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((batch/'source'/name).read_bytes()).hexdigest()==digest
    collection=json.loads((batch/'round_02/collection.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False)
    started=time.time();rows=[]
    write_json(output/'protocol.json',{'scope':__doc__,'case_ids':list(range(8)),
        'wall_seconds':900,'started_epoch':started,'source_batch':batch.name,'round':2,
        'intervention':'Replan and initialize MPC anew at each measured state; source held-reference label is never changed in its original dataset'})
    with gzip.open(output/'trace.jsonl.gz','wt',encoding='utf-8') as stream:
        for case_id in range(8):
            case=protocol['collection_cases'][case_id]
            cfg=EnvConfig.from_dict(protocol['env']);cfg.scenario=case['scenario'];cfg.split='train'
            env=NavigationEnv(cfg);env.reset(seed=case['seed'])
            delta=[];phases=[];maximum=0.;query_seconds=0.
            try:
                with gzip.open(batch/'round_02'/f'collection_{case_id:03d}.jsonl.gz','rt',encoding='utf-8') as source:
                    for control,line in enumerate(source):
                        if control%20==0:
                            ok,reason=resources_ok(output)
                            if not ok or time.time()-started>900:raise RuntimeError('Audit guard: '+str(reason))
                        old=json.loads(line)
                        fresh,audit=RecoveryTeacher(env).query()
                        difference=fresh-np.asarray(old['teacher_action'])
                        delta.append(difference);phases.append(control%10)
                        query_seconds+=audit['query_seconds']
                        _,_,_,_,info=env.step(np.asarray(old['executed_action'],dtype=np.float32))
                        expected=old['info']['trace'][0]['state'];actual=info['trace'][0]['state']
                        error=max(abs(actual[k]-expected[k]) for k in expected)
                        maximum=max(maximum,error)
                        assert error<1e-5, 'Actual replay diverged'
                        stream.write(json.dumps({'case_id':case_id,'control':control,'phase':control%10,
                            'held_reference_label':old['teacher_action'],'fresh_reference_label':fresh,
                            'fresh_query':audit,'executed_action':old['executed_action'],'actual_state':actual,
                            'replay_state_error':error},default=jsonable)+'\n')
                assert env.done
                assert info['episode_metrics']['termination_reason']==collection['episodes'][case_id]['mixture_metrics']['termination_reason']
                differences=np.asarray(delta);phases=np.asarray(phases)
                rows.append({'case_id':case_id,**case,'actual_replay_controls':len(delta),'fresh_teacher_queries':len(delta),
                    'max_state_error':maximum,'query_seconds':query_seconds,
                    'mean_absolute_label_difference':np.abs(differences).mean(0),
                    'boundary_mean_absolute_difference':np.abs(differences[phases==0]).mean(0),
                    'within_block_mean_absolute_difference':np.abs(differences[phases!=0]).mean(0),
                    'max_absolute_difference':np.abs(differences).max(0),
                    'fraction_abs_difference_over_01':(np.abs(differences)>.1).mean(0)})
                write_json(output/'progress.json',rows)
                print(json.dumps(rows[-1],default=jsonable),flush=True)
            finally:env.close()
    write_json(output/'results.json',{'scope':__doc__,'cases':rows,
        'additional_replay_controls':sum(r['actual_replay_controls'] for r in rows),
        'additional_teacher_queries':sum(r['fresh_teacher_queries'] for r in rows)})


if __name__=='__main__':main()
