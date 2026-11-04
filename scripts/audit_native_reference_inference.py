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

OUTCOME_VERSION='action_aligned_outcome_context_v1'


def outcome_codec(version,*,migration=False):
    """Old frozen sources predate the new module and must remain replayable."""
    if version!=OUTCOME_VERSION and not migration:
        return None
    from navstack.outcome_context_policy import VERSION as actual_version,outcome_context
    if actual_version!=OUTCOME_VERSION:
        raise ValueError('Outcome codec version differs from the registered auditor')
    return outcome_context


def audit(batch,method,*,checkpoint=None,migration_proof=None,all_static=False,prefix_audit=None,outcome_checkpoint=None):
    import numpy as np
    from navstack.independent_clip_ppo import load_ppo
    batch=Path(batch).resolve();folder=batch/method
    p=read(batch/'protocol.json');raw_path=folder/'endpoint_raw_audit.json';completed=None
    if prefix_audit is None:
        raw=read(raw_path);evaluation=read(folder/'evaluation.json')
        if raw['protocol_sha256']!=sha(batch/'protocol.json') or raw['evaluation_sha256']!=sha(folder/'evaluation.json'):
            raise ValueError('Require a full raw-bound evaluation')
    else:
        if migration_proof is not None:raise ValueError('Prefix inference does not certify policy migration')
        raw_path=Path(prefix_audit);prefix=read(raw_path)
        if (prefix['protocol_sha256']!=sha(batch/'protocol.json') or prefix['full_evaluation_complete'] is not False
                or prefix['new_PPO_updates']!=0 or len(prefix['bridge_streams'])!=1):
            raise ValueError('Require a closed raw-verified zero-update prefix')
        from scripts.prepare_native_recovery import verified_checkpoint_metadata
        meta=verified_checkpoint_metadata(batch/p['recovery'][method]['checkpoint'])
        evaluation={key:meta[key] for key in ['model_sha256','normalization_sha256']}
        completed=set(prefix['completed_case_ids']);evaluation['episodes']=prefix['completed_cases']
        stream=prefix['bridge_streams'][0]
        raw=dict(protocol_sha256=prefix['protocol_sha256'],evaluation_trace_sha256=prefix['evaluation_trace_sha256'],
            bridge_response_stream=stream['name'],bridge_response_sha256=stream['sha256'])
    for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
        if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated model/normalizer changed')
    for name,h in p['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=h:
            from scripts.prepare_fixed_lowlevel_adaptation import without_settings
            if (outcome_checkpoint is None or name not in ('navstack/fixed_lowlevel_training.py','navstack/mpc_reference_effort.py')
                    or without_settings(ROOT/name)!=without_settings(batch/'source'/name)):
                raise ValueError('Frozen observation/planning source differs')
    expected_version=p.get('reference_context',{}).get('version',VERSION)
    outcome_context=outcome_codec(expected_version,migration=outcome_checkpoint is not None)
    if outcome_checkpoint is not None:
        if migration_proof is not None or prefix_audit is not None:raise ValueError('Outcome migration requires a closed full endpoint')
        checkpoint=Path(outcome_checkpoint);receipt=read(checkpoint/'outcome_migration.json')
        if (receipt['old_model_sha256']!=evaluation['model_sha256'] or receipt['new_model_sha256']!=sha(checkpoint/'model.zip')
                or receipt['normalization_sha256']!=sha(checkpoint/'normalization.pkl')
                or sha(folder/'normalization.pkl')!=sha(checkpoint/'normalization.pkl')
                or receipt['policy_source_sha256']!=sha(ROOT/'navstack/outcome_context_policy.py')):
            raise ValueError('Outcome migration artifacts differ')
        expected_version=OUTCOME_VERSION;scope='OUTCOME_ZERO_EFFECT_ON_RECORDED_NATIVE_STATES'
    elif migration_proof is not None:
        proof=read(migration_proof)['results'][method]
        checkpoint=Path(checkpoint)
        if (proof['old_model_sha256']!=evaluation['model_sha256']
                or proof['new_model_sha256']!=sha(checkpoint/'model.zip')
                or proof['normalization_sha256']!=sha(checkpoint/'normalization.pkl')):
            raise ValueError('Migration comparison must bind both old and transformed models')
        scope='MIGRATION_EQUIVALENCE_ON_RECORDED_LEGACY_STATES'
    else:
        checkpoint=folder
        if (p.get('reference_context') not in ({'version':VERSION},{'version':OUTCOME_VERSION})
                or sha(checkpoint/'model.zip')!=evaluation['model_sha256']
                or sha(checkpoint/'normalization.pkl')!=evaluation['normalization_sha256']):
            raise ValueError('Require the registered reference-context evaluated model')
        scope='MEASURED_REFERENCE_CONTEXT_NATIVE_INFERENCE'
        if prefix_audit is not None:scope+='_CLOSED_COMPLETED_PREFIX_ONLY'
    model=load_ppo(checkpoint/'model.zip',device='cpu')
    if getattr(model.policy,'reference_context_version',None)!=expected_version or model.observation_space.shape!=(4736,):
        raise ValueError('Unsupported reference-context policy schema')
    if migration_proof is not None:
        import torch
        if (torch.count_nonzero(model.policy.action_net.reference_bias.gain)
                or torch.count_nonzero(model.policy.value_net.context_weight)):
            raise ValueError('A migration equivalence check requires zero new parameters')
    with (checkpoint/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
    norm.training=False
    original_model=None
    if outcome_checkpoint is not None:
        import torch
        original_model=load_ppo(folder/'model.zip',device='cpu')
        if (torch.count_nonzero(model.policy.action_net.reference_bias.failure_gain)
                or torch.count_nonzero(model.policy.value_net.failure_weight)):
            raise ValueError('Outcome migration must have zero new actor and critic terms')
    rms=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    if migration_proof is not None:
        with (folder/'normalization.pkl').open('rb') as f:original_norm=pickle.load(f)
        np.testing.assert_array_equal(norm.obs_rms.mean[:-31],original_norm.obs_rms.mean)
        np.testing.assert_array_equal(norm.obs_rms.var[:-31],original_norm.obs_rms.var)
        if norm.obs_rms.count!=original_norm.obs_rms.count:raise ValueError('Migrated RMS count differs')
    stream=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(stream)!=raw['bridge_response_sha256']:raise ValueError('Measured bridge stream changed')
    bridge,error=read_prefix(stream)
    if error:raise ValueError(error)
    refinement=p.get('reference_road_refinement')
    probe_responses={(x['response'].get('episode_id'),x['response'].get('request_id')):x['response'] for x in bridge}
    probe_requests={}
    if refinement is not None:
        for request_path in (batch/'infrastructure').glob('*_requests.jsonl.gz'):
            requests,request_error=read_prefix(request_path)
            if request_error:raise ValueError(request_error)
            for wrapper in requests:
                request=wrapper['request'];probe_requests[(request['episode_id'],request['request_id'])]=request
    begins={w['response']['episode_id']:w['response'] for w in bridge if 'scene' in w.get('response',{})}
    trace=folder/'evaluation_trace.jsonl.gz'
    if sha(trace)!=raw['evaluation_trace_sha256']:raise ValueError('Recorded policy trace changed')
    rows,error=read_prefix(trace)
    if error:raise ValueError(error)
    if completed is not None:
        actual_complete={r['case_id'] for r in rows if 'episode_metrics' in r['info']}
        if actual_complete!=completed:raise ValueError('Completed prefix selection differs from actual episode metrics')
        rows=[r for r in rows if r['case_id'] in completed]
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
        if p.get('reference_option_commitment'):
            from navstack.reference_option_commitment import configure,settings as option_settings
            configure(env,option_settings(p))
        refinement_cells=set();refinement_points=set()
        counts=0;last_action=None;failed_actions=set()
        try:
            for row in selected if all_static else selected[:1]:
                legacy_obs=np.r_[env._observation(),reference_candidate_context(env)].astype(np.float32)
                obs=(np.r_[legacy_obs[:-31],outcome_context(env,last_action,failed_actions)] if expected_version==OUTCOME_VERSION else legacy_obs).astype(np.float32)
                if original_model is not None:
                    import torch
                    a=torch.as_tensor(norm.normalize_obs(legacy_obs[None]));z=torch.as_tensor(norm.normalize_obs(obs[None]))
                    with torch.no_grad():
                        if (not torch.equal(original_model.policy.get_distribution(a).distribution.probs,model.policy.get_distribution(z).distribution.probs)
                                or not torch.equal(original_model.policy.predict_values(a),model.policy.predict_values(z))):
                            raise ValueError('Outcome migration changes a recorded distribution or value')
                action=int(model.predict(norm.normalize_obs(obs[None]),deterministic=True)[0][0])
                if action!=int(row['policy_action']):raise ValueError(f'NN action differs at case{case},decision{counts}')
                counts+=1
                if all_static:
                    recorded=row['info']['plan']
                    if refinement is not None:
                        from navstack.reference_road_refinement import planning_view,rejected_cells,pose_digest,VERSION as REFINEMENT_VERSION
                        evidence=(recorded or {}).get('road_refinement')
                        if evidence is None or evidence['version']!=REFINEMENT_VERSION:raise ValueError('Missing planning refinement evidence')
                        if p.get('reference_compute_profile'):
                            from navstack.reference_compute import VERSION as COMPUTE_VERSION,enabled
                            if not enabled(p) or evidence.get('compute_profile')!=COMPUTE_VERSION:
                                raise ValueError('Missing registered exact-compute evidence')
                        for reply in evidence['probes']:
                            key=(episode,reply['request_id']);request=probe_requests.get(key)
                            if reply!=probe_responses.get(key) or not request or request['op']!='reference_road_probe':
                                raise ValueError('Road query not bound to actual bridge stream')
                            response=reply['reference_road_probe']
                            if (reply['frame']!=env._measured_frame or request['expected_frame']!=env._measured_frame
                                    or response['pose_sha256']!=pose_digest(request['poses'])
                                    or response['sample_count']!=len(request['poses'])*108):
                                raise ValueError('Road query frame/pose/sample binding differs')
                            refinement_cells.update(rejected_cells(response['rejected_world'],begin['scene']['native_layout'],env.world.road))
                            if p.get('reference_point_constraints'):
                                from navstack.reference_point_constraints import local_points,MAX_POINTS
                                proposed=refinement_points|set(local_points(response['rejected_world'],begin['scene']['native_layout']))
                                if len(proposed)>MAX_POINTS:
                                    if not evidence['point_constraints']['budget_exceeded']:
                                        raise ValueError('Unreported point-constraint budget exhaustion')
                                else:refinement_points=proposed
                        if sorted([list(c) for c in refinement_cells])!=evidence['cells']:
                            raise ValueError('Planning constraints do not match recorded counterexamples')
                        if p.get('reference_point_constraints'):
                            from navstack.reference_point_constraints import settings as point_settings
                            point_settings(p)
                            points=evidence.get('point_constraints')
                            if (not points or points['version']!=p['reference_point_constraints']['version']
                                    or points['points']!=sorted([list(q) for q in refinement_points])):
                                raise ValueError('Exact point constraints differ from actual native counterexamples')
                    if recorded and recorded['status'] in ('success','reference_reused'):
                        owner=env.world
                        if refinement is not None and recorded['status']=='success':
                            env.world=planning_view(owner,refinement_cells,refinement_points if p.get('reference_point_constraints') else None)
                        from navstack.steering_transition import prepare_reference as steering_prepare,settings as steering_settings
                        try:_,detail=steering_prepare(env,action,env._prepare_reference,steering_settings(p))
                        finally:env.world=owner
                        if detail.get('steering_transition')!=recorded.get('steering_transition'):
                            raise ValueError('Measured steering transition differs')
                        if detail.get('reference_option_commitment')!=recorded.get('reference_option_commitment'):
                            raise ValueError('Selected option persistence differs')
                        def effective_nodes(value):
                            if value.get('cache_hit'):
                                if value.get('expanded_nodes')!=0 or type(value.get('cached_expanded_nodes')) is not int or value['cached_expanded_nodes']<=0:
                                    raise ValueError('Invalid exact spatial-plan cache counters')
                                return value['cached_expanded_nodes']
                            return value.get('expanded_nodes')
                        if recorded.get('cache_scope')=='exact_static_steering_plan_v3':
                            from navstack.reference_compute import enabled
                            if not enabled(p):raise ValueError('Unregistered successful spatial-plan memo')
                        if detail['status']!=recorded['status'] or effective_nodes(detail)!=effective_nodes(recorded):
                            raise ValueError('Reconstructed timed reference differs from recorded planner')
                        if refinement is not None and recorded['status']=='success':
                            if not evidence['probes']:raise ValueError('New reference lacks native geometry query')
                            accepted=evidence['probes'][-1]['reference_road_probe']
                            poses=[[float(x),float(y),float(yaw)] for x,y,yaw in zip(env.reference.x,env.reference.y,env.reference.yaw)]
                            if (accepted['pose_sha256']!=pose_digest(poses) or accepted['rejected_world'] or accepted['truncated']):
                                raise ValueError('Accepted query differs from reconstructed timed reference')
                    else:
                        # The recorded failure/wait has no active reference.
                        # Do not rerun a rejected search or invent its success.
                        env.reference=None;env._reference_selection=None
                    advance_observation(env,row)
                    last_action=int(row['policy_action'])
                    if row['info']['mode']=='reference':failed_actions.clear()
                    elif row['info']['mode']=='plan_failed' and last_action:failed_actions.add(last_action)
        finally:env.close()
        results.append(dict(case_id=case,decisions=counts))
    if not results:raise ValueError('No measured inference cases selected')
    if not all_static and len(results)!=len(evaluation['episodes']):raise ValueError('Missing measured start')
    np.testing.assert_array_equal(rms[0],norm.obs_rms.mean);np.testing.assert_array_equal(rms[1],norm.obs_rms.var)
    if rms[2]!=norm.obs_rms.count:raise ValueError('Frozen inference RMS count changed')
    return dict(scope=scope,method=method,batch=batch.name,all_static_actions=all_static,
        model_sha256=sha(checkpoint/'model.zip'),normalization_sha256=sha(checkpoint/'normalization.pkl'),
        raw_audit_sha256=sha(raw_path),script_sha256=sha(Path(__file__)),
        results=results,all_recorded_actions_match=True,inference_normalization_frozen=not norm.training,
        outcome_migration_probabilities_and_values_exact=original_model is not None,
        RMS_unchanged=True,new_controls=0,new_updates=0,
        limits='Recorded-state inference only,no vehicle integration or new performance score. Static action replay reconstructs successful references and consumes recorded failure/wait outcomes;it does not prove failed searches infeasible.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--method',required=True)
    p.add_argument('--checkpoint');p.add_argument('--migration-proof');p.add_argument('--all-static-actions',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--prefix-audit')
    p.add_argument('--outcome-checkpoint')
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous inference audit')
    result=audit(a.batch,a.method,checkpoint=a.checkpoint,migration_proof=a.migration_proof,all_static=a.all_static_actions,prefix_audit=a.prefix_audit,outcome_checkpoint=a.outcome_checkpoint)
    atomic_json(output,result);print({'cases':len(result['results']),'decisions':sum(r['decisions'] for r in result['results']),'scope':result['scope']})
