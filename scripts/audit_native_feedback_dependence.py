"""Frozen-input mechanism checks; no driving or causal performance claim."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch

VARIANTS=('zero_continuation','zero_outcome','zero_all_feedback')


def ablate(raw,variant):
    result=np.asarray(raw,dtype=np.float32).copy()
    if result.ndim!=1 or result.size not in (4701,4705) or variant not in VARIANTS:raise ValueError('Unsupported candidate observation or intervention')
    candidates=result[351:599].reshape(31,8)
    if variant=='zero_continuation':candidates[:,7]=0
    elif variant=='zero_outcome':result[599:605]=0
    else:candidates[:,5:]=0;result[599:605]=0
    return result


def unmasked_logits(policy,normalized):
    features=policy.extract_features(torch.as_tensor(normalized,dtype=torch.float32))
    if not hasattr(policy.mlp_extractor,'backbone') or not hasattr(policy.action_net,'head'):raise ValueError('Require the registered endpoint-clear policy')
    extractor=policy.mlp_extractor
    latent=(extractor.forward_actor_unmasked(features) if hasattr(extractor,'forward_actor_unmasked')
            else extractor.backbone.forward_actor(features))
    return policy.action_net.head(latent)


def probabilities(policy,normalized,fixed_mask):
    logits=unmasked_logits(policy,normalized)
    return torch.distributions.Categorical(logits=logits.masked_fill(~fixed_mask,-1e8)).probs


def analyze(batch,output,wall_seconds=600):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve prior mechanism evidence')
    if type(wall_seconds) is not int or not 60<=wall_seconds<=900:raise ValueError('Invalid bounded offline budget')
    started=time.monotonic();frozen=batch/'source'
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=read(batch/'protocol.json')
    if protocol['methods']!=['H','H0']:raise ValueError('Require the complete trained H/H0pair')
    for name,digest in protocol['source_hashes'].items():
        if sha(frozen/name)!=digest:raise ValueError('Frozen source changed')
    sys.path.insert(0,str(frozen))
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.native_cases import registered_native_cases
    from navstack.research import resources_ok
    from scripts.audit_native_policy_sequence import observation_env,advance_observation
    expected_cases=set(range(len(registered_native_cases(protocol))))
    torch.set_num_threads(1);methods=[]
    for method in protocol['methods']:
        folder=batch/method;audit=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json')
        for path,key in [(batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
                         (folder/'evaluation.json','evaluation_sha256'),(folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256')]:
            if sha(path)!=audit[key]:raise ValueError('Input differs from completed raw audit')
        for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
            if sha(folder/name)!=evaluation[key]:raise ValueError('Frozen inference artifact changed')
        bridge=batch/'infrastructure'/audit['bridge_response_stream']
        if sha(bridge)!=audit['bridge_response_sha256']:raise ValueError('Measured bridge stream changed')
        begins={}
        with gzip.open(bridge,'rt',encoding='utf-8') as stream:
            for line in stream:
                response=json.loads(line)['response']
                if 'scene' in response:begins[response['episode_id']]=response
        config=EnvConfig.from_dict(read(folder/'config.json')['env'])
        if not config.endpoint_candidate_filter:raise ValueError('Require explicit endpoint-only filter')
        model=load_ppo(folder/'model.zip',device='cpu');model.policy.set_training_mode(False)
        saved={k:v.clone() for k,v in model.policy.state_dict().items()};counters=(model.num_timesteps,model._n_updates)
        with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
        norm.training=False;statistics=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
        envs={};results=[];counts=Counter();expected={r['case_id']:r['actions'] for r in evaluation['episodes']}
        with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic()-started>=wall_seconds:raise TimeoutError('Declared offline wall budget exhausted')
                healthy,reason=resources_ok(batch)
                if not healthy:raise RuntimeError(str(reason))
                row=json.loads(line);case=row['case_id']
                if case not in envs:
                    begin=begins[row['info']['trace'][0]['backend_evidence']['episode_id']]
                    envs[case]=observation_env(config,begin['scene'],row['info']['initial_scenario'],begin=begin)
                    envs[case].world.road_collision_backend=config.road_collision_backend
                env=envs[case];raw=env._observation();normalized=norm.normalize_obs(raw[None])
                with torch.no_grad():
                    actual=model.policy.get_distribution(torch.as_tensor(normalized)).distribution.probs
                    latent=model.policy.mlp_extractor.forward_actor(model.policy.extract_features(torch.as_tensor(normalized)))
                    mask=latent[:,-31:]>.5
                    torch.testing.assert_close(probabilities(model.policy,normalized,mask),actual,rtol=0,atol=0)
                    action=int(actual.argmax(1)[0]);assert action==int(row['policy_action']),'Factual measured decision did not reproduce'
                    changes={}
                    for variant in VARIANTS:
                        altered=norm.normalize_obs(ablate(raw,variant)[None])
                        alternative=probabilities(model.policy,altered,mask)
                        changes[variant]={'action':int(alternative.argmax(1)[0]),'total_variation':float(.5*(actual-alternative).abs().sum())}
                    unmasked=torch.distributions.Categorical(logits=unmasked_logits(model.policy,normalized)).probs
                results.append({'case_id':case,'decision':counts[case],'factual_action':action,'allowed_actions':int(mask.sum()),
                    'unmasked_action':int(unmasked.argmax(1)[0]),'variants_fixed_original_mask':changes})
                counts[case]+=1;advance_observation(env,row)
                if len(results)%40==0:print(json.dumps({'method':method,'decisions_reproduced':len(results),'case_id':case}),flush=True)
        if set(counts)!=expected_cases or dict(counts)!=expected:raise ValueError('Incomplete registered decision inventory')
        assert counters==(model.num_timesteps,model._n_updates)
        assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in saved.items())
        np.testing.assert_array_equal(statistics[0],norm.obs_rms.mean);np.testing.assert_array_equal(statistics[1],norm.obs_rms.var)
        assert statistics[2]==norm.obs_rms.count
        summary={v:{'changed_actions':sum(r['variants_fixed_original_mask'][v]['action']!=r['factual_action'] for r in results),
            'mean_total_variation':float(np.mean([r['variants_fixed_original_mask'][v]['total_variation'] for r in results]))} for v in VARIANTS}
        methods.append({'method':method,'decisions_reproduced':len(results),'cases':len(counts),'variants':summary,
            'mask_removal_action_changes':sum(r['unmasked_action']!=r['factual_action'] for r in results),
            'factual_action_histogram':dict(Counter(r['factual_action'] for r in results)),
            'endpoint_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'model_sha256':evaluation['model_sha256'],
            'normalization_unchanged':True,'model_and_learning_counters_unchanged':True,'decisions':results})
    report={'scope':'Within each frozen policy,zero selected feedback inputs while HOLDING ITS ORIGINAL endpoint mask. Separate removal of the mask. All measured decisions reproduce;no plant integration,driving or causal task-performance claim.',
        'batch':batch.name,'wall_budget_seconds':wall_seconds,'actual_wall_seconds':time.monotonic()-started,
        'source_hashes_verified':True,'new_controls':0,'optimizer_updates':0,'analysis_script_sha256':sha(Path(__file__)),
        'limitations':'Feature dependence is not independent training replication or evidence of improved closed-loop completion. Mask-only and learned-input effects are separated;zeroing feedback is a frozen deployment intervention.',
        'methods':methods}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps([{k:v for k,v in m.items() if k!='decisions'} for m in methods]),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True);parser.add_argument('--wall-seconds',type=int,default=600)
    a=parser.parse_args();analyze(a.batch,a.output,a.wall_seconds)
