"""Reconstruct all decisions,then hide learned feedback with weights/mask fixed.

This is an observational-state intervention,not new driving or performance of
an ablated policy. H0 supplies a negative control because it already hides it.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.candidate_filter import SharedEndpointFeedbackAblationExtractor
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_policy_sequence import observation_env,advance_observation
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch,method,output):
    batch,output=Path(batch),Path(output);folder=batch/method
    if output.exists():raise FileExistsError('Preserve previous reliance evidence')
    require_finished_parent(batch)
    protocol,raw=read(batch/'protocol.json'),read(folder/'endpoint_raw_audit.json')
    if protocol.get('shared_endpoint_mask_v1') is not True or method not in ('H','H0'):
        raise ValueError('Require the registered shared physical-mask feedback pair')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:
            raise ValueError('Use the exact frozen policy/observation implementation')
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
            (folder/'evaluation.json','evaluation_sha256'),(folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Audited native input binding differs')
    evaluation=read(folder/'evaluation.json')
    for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
        if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated model bytes differ')
    bridge=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(bridge)!=raw['bridge_response_sha256']:raise ValueError('Native response stream differs')
    begins={}
    with gzip.open(bridge,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'scene' in response:begins[response['episode_id']]=response
    config=EnvConfig.from_dict(read(folder/'config.json')['env']);torch.set_num_threads(1)
    model=load_ppo(folder/'model.zip',device='cpu');model.policy.set_training_mode(False)
    weights={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    expected={r['case_id']:r['actions'] for r in evaluation['episodes']}
    results=[];counts=Counter();controls=0;env=None;current=None;started=time.monotonic()
    try:
        with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic()-started>300:raise TimeoutError('Bounded offline reconstruction expired')
                row=json.loads(line);case=row['case_id'];info=row['info']
                if case!=current:
                    if env is not None:env.close()
                    begin=begins[info['trace'][0]['backend_evidence']['episode_id']]
                    env=observation_env(config,begin['scene'],info['initial_scenario'],begin=begin)
                    env.world.road_collision_backend=config.road_collision_backend
                    current=case
                features=torch.as_tensor(norm.normalize_obs(env._observation()[None]))
                with torch.no_grad():
                    actual=model.policy.get_distribution(features).distribution
                    action=int(actual.probs.argmax(1)[0])
                    if action!=int(row['policy_action']):raise ValueError('Full recorded decision failed exact reconstruction')
                    extractor=model.policy.mlp_extractor
                    mask=extractor.forward_actor(features)[:,-31:]
                    hidden=SharedEndpointFeedbackAblationExtractor.learned_inputs(features)
                    # The original physical endpoint mask is appended unchanged;
                    # only the learned actor's feedback inputs are hidden.
                    latent=extractor.backbone.forward_actor(hidden)
                    logits=model.policy.action_net(torch.cat((latent,mask),dim=1))
                    counterfactual=torch.distributions.Categorical(logits=logits)
                    valid=mask>.5
                    if not torch.equal(actual.probs>0,valid) or not torch.equal(counterfactual.probs>0,valid):
                        raise ValueError('Counterfactual changes the common physical action mask')
                    altered=int(counterfactual.probs.argmax(1)[0])
                    tv=float(.5*torch.abs(actual.probs-counterfactual.probs).sum())
                results.append({'case_id':case,'decision':counts[case],'measured_before_frame':env._measured_frame,
                    'actual_action':action,'hidden_feedback_action':altered,'action_changed':action!=altered,
                    'probability_total_variation':tv,'recorded_mode':info['mode']})
                counts[case]+=1;controls+=len(info['trace']);advance_observation(env,row)
    finally:
        if env is not None:env.close()
    if dict(counts)!=expected or controls!=raw['evaluation_control_frames_verified']:
        raise ValueError('Incomplete full-decision/frame inventory')
    if not all(torch.equal(v,model.policy.state_dict()[k]) for k,v in weights.items()):
        raise ValueError('Policy weights or buffers changed')
    np.testing.assert_array_equal(stats[0],norm.obs_rms.mean);np.testing.assert_array_equal(stats[1],norm.obs_rms.var)
    if stats[2]!=norm.obs_rms.count:raise ValueError('Observation statistics changed')
    if method=='H0' and any(r['action_changed'] or r['probability_total_variation']>1e-7 for r in results):
        raise ValueError('Already-ablated H0 negative control changed')
    report={'scope':__doc__,'batch':batch.name,'method':method,'script_sha256':sha(Path(__file__)),
        'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'model_sha256':evaluation['model_sha256'],
        'decisions_reproduced':len(results),'native_control_history_verified':controls,
        'changed_actions':sum(r['action_changed'] for r in results),
        'mean_probability_total_variation':float(np.mean([r['probability_total_variation'] for r in results])),
        'weights_statistics_and_physical_mask_unchanged':True,'results':results,
        'wall_seconds':time.monotonic()-started,'new_controls':0,'optimizer_updates':0,
        'limits':'Same visited states only;no counterfactual trajectories or success rates. Nonzero sensitivity is not evidence of performance benefit or feedback sufficiency.',
        'research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('batch','method','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();analyze(args.batch,args.method,args.output)
