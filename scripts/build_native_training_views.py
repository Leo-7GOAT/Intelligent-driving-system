"""Convert complete measured H TRAINING controls into B1 views;no new driving."""
import argparse
from collections import defaultdict,Counter
from dataclasses import asdict
import gzip,json,os
from pathlib import Path
import shutil,sys,time
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.experiments import signature
from navstack.measured_demo import measured_flat_env,measured_label,advance_measured_flat
from navstack.native_cases import verify_registered_native_begin,registered_native_cases
from navstack.research import resources_ok
from scripts import audit_native_method_endpoint as verifier
read,sha,read_prefix=verifier.read,verifier.sha,verifier.read_prefix


def sampled_indices(length,stride):
    if type(length) is not int or length<1 or type(stride) is not int or not 1<=stride<=20:
        raise ValueError('Require nonempty training episode and bounded integer stride')
    return set(range(0,length,stride))|{length-1}


def compatible_configs(teacher,target):
    differences={k for k in teacher.keys()|target.keys() if teacher.get(k)!=target.get(k)}
    if (teacher.get('method')!='H' or target.get('method')!='B1' or target.get('flat_hold')!=1
            or differences-{'method','flat_hold','endpoint_candidate_filter'}):
        raise ValueError('Teacher and target physical/task configurations differ')


def build(batch,reference_config,output,stride=5,wall_seconds=600):
    batch,reference_config,output=map(lambda p:Path(p).resolve(),(batch,reference_config,output))
    if output.exists():raise FileExistsError('Preserve measured datasets')
    sampled_indices(1,stride)
    if type(wall_seconds) is not int or not 60<=wall_seconds<=1800:raise ValueError('Bounded reconstruction budget required')
    started=time.monotonic();folder=batch/'H';protocol=read(batch/'protocol.json');summary=read(folder/'summary.json')
    if (read(batch/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped'
            or summary['stopped_reason'] is not None or not summary['physics_steps'] or not summary['optimizer_steps']):
        raise ValueError('Require completed training and verified owned cleanup;validation may have failed separately')
    if (batch/'interruption_audit.json').exists():
        prefix=read(batch/'interruption_audit.json')['methods']['H']
        if prefix['trace_errors'] or prefix['optimization_error'] or prefix['unknown_unflushed_tail'] or prefix['readable_trace_controls']!=summary['physics_steps']:
            raise ValueError('Training prefix is incomplete or uncertain')
    for name,digest in protocol['source_hashes'].items():
        if sha(batch/'source'/name)!=digest:raise ValueError('Teacher frozen source changed')
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:raise ValueError('Use the exact teacher observation/runtime')
    if sha(batch/'calibration/calibration.json')!=protocol['calibration_sha256']:raise ValueError('Teacher calibration changed')
    configuration=read(reference_config);cfg=EnvConfig.from_dict(configuration['env'])
    if signature(cfg)!=configuration['signature']:raise ValueError('Target configuration signature differs')
    compatible_configs(read(folder/'config.json')['env'],cfg.to_dict())
    target_source=reference_config.parent.parent/'source'
    for name in ('env','world','carla_actuation','measured_demo','road_grid','config','native_route'):
        if sha(target_source/'navstack'/f'{name}.py')!=sha(ROOT/'navstack'/f'{name}.py'):
            raise ValueError('Teacher/target observation or physics implementation differs:'+name)
    optimization,error=read_prefix(folder/'optimization.jsonl')
    if error or not optimization or any(optimization[-1][k]!=summary[k] for k in ('physics_steps','optimizer_steps','optimization_phases')):
        raise ValueError('Training optimizer counters differ')
    checkpoint=folder/read(folder/'latest_checkpoint.json')['folder'];metadata=read(checkpoint/'metadata.json')
    if any(metadata[k]!=summary[k] for k in ('physics_steps','optimizer_steps','lineage_control_steps')):raise ValueError('Training checkpoint counters differ')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        if sha(checkpoint/name)!=metadata[key]:raise ValueError('Training checkpoint changed')
    actual,acks,owners,begins,bridge_hashes={},{},{},{},{}
    for path in sorted((batch/'infrastructure').glob('*_responses.jsonl.gz')):
        responses,error=read_prefix(path)
        if error:raise ValueError('Complete bridge stream required:'+str(error))
        bridge_hashes[path.name]=sha(path)
        for wrapper in responses:
            response=wrapper.get('response',{});episode=response.get('episode_id');request=response.get('request_id')
            if episode is not None and request is not None:acks[(episode,request)]=response
            if 'scene' in response:begins[episode]=response
            if 'vehicle_control' in response:
                key=(episode,response['frame'])
                if key in actual:raise ValueError('Duplicate measured control')
                actual[key]=response;owners[key]=path.name
    episodes=defaultdict(list);trace_hashes={};used=set();previous={};identities=set();verified=0
    # Only TRAINING transition streams are opened;no evaluation trace path is used.
    for path in sorted(folder.glob('transitions*.jsonl.gz')):
        rows,error=read_prefix(path)
        if error:raise ValueError('Complete training trace required:'+str(error))
        trace_hashes[path.name]=sha(path)
        for row in rows:
            ticks=row['trace'];verified+=verifier._verify_ticks(ticks,actual,acks,owners,used,previous,identities,cfg.actuator_observation)
            episodes[row['episode_id']].extend(ticks)
    if verified!=summary['physics_steps'] or not episodes:raise ValueError('Complete training control inventory required')
    output.mkdir(parents=True);script_hash=sha(Path(__file__))
    registration={'scope':__doc__,'batch':batch.name,'method':'H','protocol_sha256':sha(batch/'protocol.json'),
        'teacher_summary_sha256':sha(folder/'summary.json'),'teacher_training_controls':verified,
        'teacher_learning_optimizer_steps':summary['optimizer_steps'],'teacher_charged_controls_lower_bound':summary['charged_control_steps_lower_bound'],
        'reference_config_sha256':sha(reference_config),'script_sha256':script_hash,'tick_verifier_sha256':sha(Path(verifier.__file__)),
        'trace_hashes':trace_hashes,'bridge_hashes':bridge_hashes,'stride':stride,'wall_seconds':wall_seconds,
        'selection':'Every training episode,regardless of success/failure,including the final unfinished episode;stride5physical controls plus last pre-control state when stride=5. No validation states or outcome filtering.',
        'new_controls':0,'PPO_updates':0,'supervised_updates':0}
    (output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),output/'executed_view_builder.py');shutil.copy2(Path(verifier.__file__),output/'executed_tick_verifier.py')
    observations=[];actions=[];groups=[];conditions=[];cases=[]
    try:
        for group,(episode,ticks) in enumerate(sorted(episodes.items())):
            if time.monotonic()-started>wall_seconds:raise TimeoutError('Training-view budget expired')
            healthy,reason=resources_ok(ROOT)
            if not healthy:raise RuntimeError(str(reason))
            begin=begins[ticks[0]['backend_evidence']['episode_id']]
            verify_registered_native_begin(begin,protocol,partition='training')
            scene=begin['scene'];condition=scene['condition']
            initial={'name':scene['name'],'initial':begin['state'],'goal':scene['goal'],'static':scene['static'],
                'dynamic':begin['dynamic'],'road_sha256':scene['road_grid']['sha256'],
                'actuator_evidence':{k:begin[k] for k in ('actuator_observation','speed_setpoint','actuation_state')}}
            env=measured_flat_env(cfg,scene,initial,begin=begin);selected=sampled_indices(len(ticks),stride);first=len(actions)
            try:
                for index,tick in enumerate(ticks):
                    if index%128==0 and time.monotonic()-started>wall_seconds:raise TimeoutError('Training-view budget expired')
                    label=measured_label(env,tick)
                    if index in selected:
                        observations.append(env._observation());actions.append(label);groups.append(group);conditions.append(condition)
                    advance_measured_flat(env,tick,label)
            finally:env.close()
            cases.append({'training_episode':episode,'condition':condition,'verified_controls':len(ticks),'sampled_rows':len(actions)-first})
            print(json.dumps(cases[-1]),flush=True)
        inventory=len(registered_native_cases(protocol))
        if set(conditions)!=set(range(inventory)):raise ValueError('Every registered training condition must be represented')
        dataset=output/'measured_training_views.npz'
        np.savez_compressed(dataset,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32),
            groups=np.asarray(groups,dtype=np.int64),conditions=np.asarray(conditions,dtype=np.int64))
        if sha(Path(__file__))!=script_hash:raise ValueError('Construction code changed during execution')
        result={**registration,'cases':cases,'training_conditions':inventory,'sampled_rows':len(actions),
            'dataset_sha256':sha(dataset),'elapsed_seconds':time.monotonic()-started,
            'limits':'Measured executed Hcontrols are extra supervision with their original PPO/teacher costs disclosed;not independently optimal controls or evidence of B1driving. All failed/unfinished training episodes are retained.','research_acceptance':'UNMET'}
        (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'completed_episodes':cases,'new_controls':0},indent=2),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('batch','reference-config','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--stride',type=int,default=5);p.add_argument('--wall-seconds',type=int,default=600)
    a=p.parse_args();result=build(a.batch,a.reference_config,a.output,a.stride,a.wall_seconds)
    print(json.dumps({'verified_training_controls':result['teacher_training_controls'],'sampled_rows':result['sampled_rows'],'new_controls':0}))
