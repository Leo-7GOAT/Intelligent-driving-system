"""Replay a measured begin into the frozen evaluation constructor; never step a plant."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import pickle
import sys

ROOT=Path(os.environ['NAVSTACK_AUDIT_SOURCE_ROOT']).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from navstack.native_cases import registered_native_cases,verify_registered_native_begin
from navstack.reference_context_policy import ReferenceContextObservationMixin
from navstack.research import atomic_json
from scripts import native_road_learning as frozen_worker
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def probe(batch,parent,begin_stream):
    batch,parent,begin_stream=map(lambda p:Path(p).resolve(),(batch,parent,begin_stream))
    p=read(batch/'protocol.json')
    if not p.get('evaluation_only') or p['method_budgets']!={'H':0,'H0':0}:
        raise ValueError('Require a zero-learning prepared pair')
    for name,digest in p['source_hashes'].items():
        if sha(batch/'source'/name)!=digest:raise ValueError('Frozen source changed')
    rows,error=read_prefix(begin_stream)
    if error:raise ValueError(error)
    begins=[row['response'] for row in rows if 'scene' in row['response']]
    if len(begins)!=1:raise ValueError('Use the single retained measured begin from the failed pre-control attempt')
    begin=begins[0];verify_registered_native_begin(begin,p,0)
    worker_path=ROOT/'scripts/native_road_learning.py';source=worker_path.read_text(encoding='utf-8')
    assignments=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='evaluation' for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Ambiguous evaluation constructor')
    expression=ast.Expression(assignments[0].value);ast.fix_missing_locations(expression)
    constructor=compile(expression,str(worker_path),'eval')
    class ReplayOnlyClient:
        def __init__(self):self.operations=[]
        def request(self,operation,**payload):
            if operation!='begin':raise AssertionError('Preflight must never step,finalize or tick a simulator')
            self.operations.append(operation)
            return deepcopy(begin)
    class ContextRecordedRoadEnv(ReferenceContextObservationMixin,frozen_worker.RecordedRoadEnv):pass
    results={}
    for method in ('H','H0'):
        checkpoint=batch/p['recovery'][method]['checkpoint'];verified_checkpoint_metadata(checkpoint)
        model=load_ppo(checkpoint/'model.zip',device='cpu')
        saved=load_ppo(parent/method/'model.zip',device='cpu')
        if model.policy.state_dict().keys()!=saved.policy.state_dict().keys():raise ValueError('Policy tensor inventory differs')
        for name,value in model.policy.state_dict().items():
            if not torch.equal(value,saved.policy.state_dict()[name]):raise ValueError('Checkpoint differs from saved learning endpoint:'+name)
        old_params=dict(saved.policy.named_parameters())
        for name,parameter in model.policy.named_parameters():
            before=saved.policy.optimizer.state.get(old_params[name],{});after=model.policy.optimizer.state.get(parameter,{})
            if before.keys()!=after.keys():raise ValueError('Optimizer inventory differs')
            for key,value in before.items():
                if (not torch.equal(value,after[key]) if torch.is_tensor(value) else value!=after[key]):
                    raise ValueError('Optimizer state differs:'+name+':'+key)
        config=EnvConfig.from_dict(read(checkpoint/'config.json')['env'])
        client=ReplayOnlyClient();output=batch/('offline_schema_probe_'+method);output.mkdir()
        # Execute the exact RHS from the staged frozen worker, with a begin-only
        # client. The old constructor produces4705 and fails normalization here.
        env=eval(constructor,dict(environment_class=ContextRecordedRoadEnv,
            RecordedRoadEnv=frozen_worker.RecordedRoadEnv,CarlaNavigationEnv=frozen_worker.CarlaNavigationEnv,
            config=config,client=client,evaluation_cases=registered_native_cases(p),output=output,evaluation_only=True))
        try:
            obs,_=env.reset(seed=7301,options={'case':registered_native_cases(p)[0]})
            with (checkpoint/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            norm.training=False
            normalized=norm.normalize_obs(obs[None]);action=int(model.predict(normalized,deterministic=True)[0][0])
            if obs.shape!=(4736,) or normalized.shape!=(1,4736) or not 0<=action<31:
                raise ValueError('Declared reference policy observation/action shape differs')
            if client.operations!=['begin']:raise ValueError('Unexpected preflight plant operation')
            results[method]=dict(observation_shape=list(obs.shape),normalized_shape=list(normalized.shape),
                action=action,checkpoint_model_sha256=sha(checkpoint/'model.zip'),saved_learning_model_sha256=sha(parent/method/'model.zip'),
                all_policy_and_optimizer_tensors_identical=True,client_operations=client.operations)
        finally:env.close()
    return dict(scope=__doc__,script_sha256=sha(Path(__file__)),worker_sha256=sha(worker_path),
        protocol_sha256=sha(batch/'protocol.json'),measured_begin_stream_sha256=sha(begin_stream),results=results,
        new_controls=0,new_world_ticks=0,new_updates=0,
        limits='Recorded begin replay and actual model inference only. No CARLA connection,plant.step or performance score.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    parser.add_argument('--parent',required=True);parser.add_argument('--begin-stream',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous schema preflight')
    result=probe(args.batch,args.parent,args.begin_stream);atomic_json(output,result)
    print({m:r['observation_shape'] for m,r in result['results'].items()})
