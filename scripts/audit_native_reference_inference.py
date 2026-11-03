"""Audit reference-context inference from recorded native observations only."""
import argparse
import os
from pathlib import Path
import pickle
import sys
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.reference_context import reference_candidate_context
from navstack.reference_context_policy import VERSION
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env,advance_observation


def audit(batch,method,*,checkpoint=None,migration_proof=None,all_static=False):
    import numpy as np
    from navstack.independent_clip_ppo import load_ppo
    batch=Path(batch).resolve();folder=batch/method
    p=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json')
    if raw['protocol_sha256']!=sha(batch/'protocol.json') or raw['evaluation_sha256']!=sha(folder/'evaluation.json'):
        raise ValueError('Require a full raw-bound evaluation')
    for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
        if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated model/normalizer changed')
    for name,h in p['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=h:raise ValueError('Frozen observation/planning source differs')
    if migration_proof is not None:
        proof=read(migration_proof)['results'][method]
        checkpoint=Path(checkpoint)
        if (proof['old_model_sha256']!=evaluation['model_sha256']
                or proof['new_model_sha256']!=sha(checkpoint/'model.zip')
                or proof['normalization_sha256']!=sha(checkpoint/'normalization.pkl')):
            raise ValueError('Migration comparison must bind both old and transformed models')
        scope='MIGRATION_EQUIVALENCE_ON_RECORDED_LEGACY_STATES'
    else:
        checkpoint=folder
        if (p.get('reference_context')!={'version':VERSION}
                or sha(checkpoint/'model.zip')!=evaluation['model_sha256']
                or sha(checkpoint/'normalization.pkl')!=evaluation['normalization_sha256']):
            raise ValueError('Require the registered reference-context evaluated model')
        scope='MEASURED_REFERENCE_CONTEXT_NATIVE_INFERENCE'
    model=load_ppo(checkpoint/'model.zip',device='cpu')
    if getattr(model.policy,'reference_context_version',None)!=VERSION or model.observation_space.shape!=(4736,):
        raise ValueError('Unsupported reference-context policy schema')
    if migration_proof is not None:
        import torch
        if (torch.count_nonzero(model.policy.action_net.reference_bias.gain)
                or torch.count_nonzero(model.policy.value_net.context_weight)):
            raise ValueError('A migration equivalence check requires zero new parameters')
    with (checkpoint/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
    norm.training=False
    if migration_proof is not None:
        with (folder/'normalization.pkl').open('rb') as f:original_norm=pickle.load(f)
        np.testing.assert_array_equal(norm.obs_rms.mean[:-31],original_norm.obs_rms.mean)
        np.testing.assert_array_equal(norm.obs_rms.var[:-31],original_norm.obs_rms.var)
        if norm.obs_rms.count!=original_norm.obs_rms.count:raise ValueError('Migrated RMS count differs')
    stream=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(stream)!=raw['bridge_response_sha256']:raise ValueError('Measured bridge stream changed')
    bridge,error=read_prefix(stream)
    if error:raise ValueError(error)
    begins={w['response']['episode_id']:w['response'] for w in bridge if 'scene' in w.get('response',{})}
    trace=folder/'evaluation_trace.jsonl.gz'
    if sha(trace)!=raw['evaluation_trace_sha256']:raise ValueError('Recorded policy trace changed')
    rows,error=read_prefix(trace)
    if error:raise ValueError(error)
    cfg=EnvConfig.from_dict(read(folder/'config.json')['env']);results=[]
    for case in sorted({r['case_id'] for r in rows}):
        selected=[r for r in rows if r['case_id']==case]
        episode=selected[0]['info']['trace'][0]['backend_evidence']['episode_id'];begin=begins[episode]
        static=not begin['dynamic']
        if all_static and not static:continue
        env=observation_env(cfg,begin['scene'],selected[0]['info']['initial_scenario'],begin=begin)
        env.world.road_collision_backend=cfg.road_collision_backend
        env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
        env.episode=dict(plan_times=[],plan_failures=0)
        counts=0
        try:
            for row in selected if all_static else selected[:1]:
                obs=np.r_[env._observation(),reference_candidate_context(env)].astype(np.float32)
                action=int(model.predict(norm.normalize_obs(obs[None]),deterministic=True)[0][0])
                if action!=int(row['policy_action']):raise ValueError(f'NN action differs at case{case},decision{counts}')
                counts+=1
                if all_static:
                    recorded=row['info']['plan']
                    if recorded and recorded['status'] in ('success','reference_reused'):
                        _,detail=env._prepare_reference(action)
                        if detail['status']!=recorded['status'] or detail.get('expanded_nodes')!=recorded.get('expanded_nodes'):
                            raise ValueError('Reconstructed timed reference differs from recorded planner')
                    else:
                        # The recorded failure/wait has no active reference.
                        # Do not rerun a rejected search or invent its success.
                        env.reference=None;env._reference_selection=None
                    advance_observation(env,row)
        finally:env.close()
        results.append(dict(case_id=case,decisions=counts))
    if not results:raise ValueError('No measured inference cases selected')
    if not all_static and len(results)!=len(evaluation['episodes']):raise ValueError('Missing measured start')
    return dict(scope=scope,method=method,batch=batch.name,all_static_actions=all_static,
        model_sha256=sha(checkpoint/'model.zip'),normalization_sha256=sha(checkpoint/'normalization.pkl'),
        raw_audit_sha256=sha(folder/'endpoint_raw_audit.json'),script_sha256=sha(Path(__file__)),
        results=results,all_recorded_actions_match=True,new_controls=0,new_updates=0,
        limits='Recorded-state inference only,no vehicle integration or new performance score. Static action replay reconstructs successful references and consumes recorded failure/wait outcomes;it does not prove failed searches infeasible.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--method',required=True)
    p.add_argument('--checkpoint');p.add_argument('--migration-proof');p.add_argument('--all-static-actions',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous inference audit')
    result=audit(a.batch,a.method,checkpoint=a.checkpoint,migration_proof=a.migration_proof,all_static=a.all_static_actions)
    atomic_json(output,result);print({'cases':len(result['results']),'decisions':sum(r['decisions'] for r in result['results']),'scope':result['scope']})
