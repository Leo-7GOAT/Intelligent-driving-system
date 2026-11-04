"""Compare requested pedals with CARLA's returned last-tick control,offline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha,read_prefix


def control_errors(desired,applied,tolerance=1e-5):
    import math
    keys=('throttle','brake','steer')
    if any(not math.isfinite(desired[k]) or not math.isfinite(applied[k]) for k in keys):
        raise ValueError('Nonfinite engine control acknowledgement')
    return {k:abs(desired[k]-applied[k]) for k in keys if abs(desired[k]-applied[k])>tolerance}


def evaluation_episodes(batch,protocol):
    batch=Path(batch)
    paths=[batch/method/'evaluation_trace.jsonl.gz' for method in protocol['methods']]
    summary=batch/'summary.json'
    rule=(protocol.get('policy_kind')=='rule_only_B3' or
          summary.is_file() and read(summary).get('policy_kind')=='rule_only_B3')
    if rule:paths.append(batch/'evaluation_trace.jsonl.gz')
    result=set()
    for path in paths:
        if path.exists():
            rows,error=read_prefix(path)
            if error:raise ValueError(error)
            result.update(t['backend_evidence']['episode_id'] for r in rows for t in r['info']['trace'])
    return result


def analyze(batch):
    batch=Path(batch);p=read(batch/'protocol.json');cal=read(batch/'calibration/calibration.json')
    mode=cal['recommended_adapter'].get('control_delivery','actor_async_legacy')
    if not (batch/'guard_final.json').exists() or not (batch/'infrastructure/server_stop.json').exists():
        raise ValueError('Inspect only a terminal,cleaned-up native lifecycle')
    if read(batch/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
        raise ValueError('Owned server cleanup not verified')
    evaluations=evaluation_episodes(batch,p)
    counts=Counter();channels=Counter();examples=[];streams=[];last={};seen=set();by_phase={}
    for path in sorted((batch/'infrastructure').glob('*_responses.jsonl.gz')):
        rows,error=read_prefix(path)
        if error:raise ValueError(error)
        streams.append({'name':path.name,'sha256':sha(path)})
        for wrapper in rows:
            r=wrapper.get('response',{})
            if 'vehicle_control' not in r:continue
            key=(r['episode_id'],r['frame'])
            if key in seen:raise ValueError('Duplicate actual control frame')
            seen.add(key)
            phase='evaluation' if r['episode_id'] in evaluations else 'training_or_untraced_failure'
            phase_counts=by_phase.setdefault(phase,Counter());phase_counts['controls']+=1
            desired=r['vehicle_control'];applied=r['applied_vehicle_control'];errors=control_errors(desired,applied)
            prior=last.get(r['episode_id']);counts['controls']+=1
            if errors:
                counts['mismatches']+=1;phase_counts['mismatches']+=1
                prior_match=prior is not None and not control_errors(prior['vehicle_control'],applied)
                counts['mismatches_matching_previous_request']+=int(prior_match)
                channels.update(errors.keys())
                if len(examples)<12:examples.append({'episode_id':r['episode_id'],'frame':r['frame'],
                    'requested':desired,'applied':{k:applied[k] for k in ('throttle','brake','steer')},
                    'matches_previous_request':prior_match})
            manual=cal['recommended_adapter'].get('drivetrain_mode','automatic')=='manual_first_v1'
            gear_ok=(applied['manual_gear_shift']==manual and applied['hand_brake'] is False
                and applied['reverse'] is False and (not manual or applied['gear']==1))
            counts['gear_or_handbrake_mismatches']+=int(not gear_ok)
            if mode=='batch_sync_v1':
                counts['missing_or_negative_sync_ack']+=int(r.get('control_delivery')!=mode or r.get('applied_control_matches_request') is not True)
            last[r['episode_id']]=r
    if not counts['controls']:raise ValueError('No actual native controls')
    raw_total=0
    for method in p['methods']:
        raw=batch/method/'endpoint_raw_audit.json'
        if raw.exists():
            a=read(raw);raw_total+=a['total_applied_controls_verified']
            if a['protocol_sha256']!=sha(batch/'protocol.json'):raise ValueError('Parent raw audit protocol differs')
            match=next((s for s in streams if s['name']==a['bridge_response_stream']),None)
            if match is None or match['sha256']!=a['bridge_response_sha256']:raise ValueError('Parent raw stream differs')
    if raw_total and raw_total!=counts['controls']:raise ValueError('Applied-frame inventory differs from complete raw audit')
    return {'scope':__doc__,'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),
        'control_delivery':mode,'bridge_streams':streams,'tolerance':1e-5,
        'counts':dict(counts),'mismatch_channels':dict(channels),'by_phase':{k:dict(v) for k,v in by_phase.items()},
        'mismatch_examples':examples,'all_engine_controls_match_requested':counts['mismatches']==0 and counts['gear_or_handbrake_mismatches']==0,
        'strict_delivery_gate_passed':mode=='batch_sync_v1' and counts['mismatches']==0 and counts['gear_or_handbrake_mismatches']==0 and counts['missing_or_negative_sync_ack']==0,
        'new_controls':0,'new_optimizer_updates':0,
        'limits':'Checks CARLAget_control against requested pedals,not independent wheel torque. Previous-request matches establish an acknowledgement discrepancy,not its causal contribution to policy failure.'}


def compare_transfer(old,new):
    old,new=Path(old),Path(new);a,b=analyze(old),analyze(new)
    op,np=read(old/'protocol.json'),read(new/'protocol.json')
    if (op['routes']!=np['routes'] or op['native_case_manifest']!=np['native_case_manifest']
            or read(old/'B1/config.json')['env']!=read(new/'B1/config.json')['env']):
        raise ValueError('Transfer changes task or policy observation/termination semantics')
    oc,nc=read(old/'calibration/calibration.json'),read(new/'calibration/calibration.json')
    if nc['recommended_adapter'].pop('control_delivery',None)!='batch_sync_v1' or nc!=oc:
        raise ValueError('Transfer changes calibration gains beyond command delivery')
    for name in ('model.zip','normalization.pkl'):
        if sha(old/'B1'/name)!=sha(new/'B1'/name):raise ValueError('Transfer changes learned checkpoint')
    allowed={'navstack/carla_actuation.py','navstack/carla_bridge.py','scripts/carla_bridge37.py'}
    for name,digest in op['source_hashes'].items():
        if name not in allowed and np['source_hashes'].get(name)!=digest:raise ValueError('Unrelated frozen source change:'+name)
    if not b['strict_delivery_gate_passed']:raise ValueError('Synchronous frame acknowledgement gate failed')
    arms=[]
    for folder,proof in ((old,a),(new,b)):
        episodes=read(folder/'B1/evaluation.json')['episodes']
        if sorted(e['case_id'] for e in episodes)!=[0,1,2,3]:raise ValueError('Require every original TRAINroad')
        arms.append({'batch':folder.name,'delivery':proof['control_delivery'],'actual_controls':proof['counts']['controls'],
            'control_mismatches':proof['counts'].get('mismatches',0),
            'previous_request_matches':proof['counts'].get('mismatches_matching_previous_request',0),
            'successes':sum(e['success'] for e in episodes),
            'unsafe_cases':sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes),
            'case_outcomes':[{'case_id':e['case_id'],'termination':e['termination_reason'],'controls':e['physics_steps']} for e in episodes],
            'raw_audit_sha256':sha(folder/'B1/endpoint_raw_audit.json'),'protocol_sha256':sha(folder/'protocol.json')})
    return {'scope':'Versioned low-level command delivery transfer;fixed replay policy and all4known TRAINroads. No new learning or research superiority claim.',
        'parent_model_sha256':sha(old/'B1/model.zip'),'normalization_sha256':sha(old/'B1/normalization.pkl'),
        'model_normalizer_and_env_exact':True,'calibration_gains_exact_except_delivery_version':True,
        'physical_source_overlays':sorted(allowed),'other_parent_source_hashes_exact':True,'arms':arms,
        'new_controls':b['counts']['controls'],'new_PPO_updates':0,'new_supervised_updates':0,
        'all_new_engine_acknowledgements_match':True,'new_raw_frames_verified':b['counts']['controls'],
        'new_measured_start_actions_verified':len(read(new/'B1/measured_start_inference.json')['results']),
        'original_basic_driving_gate_passed':arms[-1]['successes']>=3 and arms[-1]['unsafe_cases']==0,
        'legacy_PPO_diagnosis':'reports/native_PPO_legacy_control_delivery_20261101.json',
        'disclosure':'Earlier frame audits verified CARLAstates,requests and bridge echoes;they did not establish equality to returned engine controls. Historical physical outcomes remain preserved. Legacy mismatches are not erased or claimed to cause failure.',
        'limits':'Single known seed,frozen-policy low-level transfer. Native3/4and sync3/4both retain one bounds failure;no health qualification or PPOcontribution. Dynamic actor delivery was not tested.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    p.add_argument('--compare',help='Completed legacy parent for a checkpoint/task-matched frozen transfer comparison');a=p.parse_args()
    result=compare_transfer(a.compare,a.batch) if a.compare else analyze(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('arms','new_controls')} if a.compare else
        {k:result[k] for k in ('batch','control_delivery','counts','strict_delivery_gate_passed')}),flush=True)
