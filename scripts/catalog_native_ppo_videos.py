"""Bind every queued native PPO recording to an unambiguous experiment identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.render_carla_recordings import verified_frames


def bind_video(batch_name,method,seed,original_ids,video,episode,model_hash,norm_hash):
    """Retain the old manifest identity while adding batch-qualified public IDs."""
    case=video['case_id'];filename=video['file']
    if (not re.fullmatch(r'[A-Za-z0-9_-]+',batch_name) or method not in ('H','H0','B1')
            or type(seed) is not int or type(case) is not int or not 0<=case<len(original_ids)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+\.mp4',filename)):
        raise ValueError('Invalid recording identity or unsafe asset filename')
    original=original_ids[case]
    if (episode['case_id']!=case or episode.get('original_parent_case_id')!=original
            or episode.get('status')!='completed' or video['model_sha256']!=model_hash
            or video['seed']!=seed or video['case']!=episode['case']
            or video['success']!=episode['success'] or video['termination_reason']!=episode['termination_reason']
            or bool(video['success'])!=(video['termination_reason']=='success')):
        raise ValueError('Video and executed episode/model identities disagree')
    return {'video_id':f'{batch_name}:{method}:seed{seed}:parent_case{original}',
        'run_id':batch_name+'/'+method,'source_batch':batch_name,'method':method,'seed':seed,
        'source_video_manifest_run_id':video.get('run_id'),'case_id_in_recording':case,
        'original_parent_case_id':original,'model_sha256':model_hash,'normalization_sha256':norm_hash,
        'source_file':filename,'release_asset_name':f'{batch_name}__{method}__seed{seed}__parentcase{original}__{filename}',
        'video_sha256':video['sha256'],'case_sha256':hashlib.sha256(json.dumps(episode['case'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'success':video['success'],'termination_reason':video['termination_reason']}


def catalog(execution,output):
    execution,output=Path(execution).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve the previous native recording catalog')
    root=execution.parent.parent;plan=read(execution/'plan.json');state=read(execution/'status.json')
    if ((execution/'failure.json').exists()
            or state.get('phase')!='ALL_REGISTERED_NATIVE_PPO_RECORDINGS_AND_FRAME_VERIFICATIONS_COMPLETE'
            or [(r['batch'],r['method']) for r in state.get('completed',[])]!=[(r['batch'],r['method']) for r in plan['recordings']]):
        raise ValueError('The entire registered recording queue and owned cleanup must complete')
    records=[];proofs=[]
    for item,complete in zip(plan['recordings'],state['completed']):
        batch=root/'runs'/item['batch'];method=item['method'];folder=batch/method;videos=batch/'videos'
        require_finished_parent(batch)
        original=batch/'queued_protocol.json' if (batch/'queued_protocol.json').exists() else batch/'protocol.json'
        if sha(original)!=item['protocol_sha256']:raise ValueError('Registered camera inputs changed')
        protocol=read(batch/'protocol.json');camera=protocol['native_camera_evaluation'];raw=read(folder/'endpoint_raw_audit.json')
        parent=root/'runs'/camera['parent_batch'];require_finished_parent(parent)
        if (protocol['methods']!=[method] or protocol.get('evaluation_only') is not True
                or protocol['method_budgets']!={method:0} or protocol['recovery'][method]['retained_optimizer_steps']<=0
                or sha(parent/method/'endpoint_raw_audit.json')!=camera['parent_raw_audit_sha256']):
            raise ValueError('Require a frozen actual PPO-trained source,not an untrained or rule recording')
        for path,key in [(batch/'protocol.json','protocol_sha256')]+[(folder/n,k) for n,k in (
                ('config.json','config_sha256'),('summary.json','summary_sha256'),('evaluation.json','evaluation_sha256'),
                ('evaluation_trace.jsonl.gz','evaluation_trace_sha256'))]:
            if sha(path)!=raw[key]:raise ValueError('Camera raw-audit binding changed')
        summary=read(folder/'summary.json');config=read(folder/'config.json');evaluation=read(folder/'evaluation.json')
        if (summary['physics_steps'] or summary['optimizer_steps'] or summary['lineage_control_steps']<=0
                or config.get('policy')!='learned_PPO' or not config.get('record_camera')):
            raise ValueError('Recording unexpectedly learned or lacks an actual camera/PPO lineage')
        for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
            if sha(folder/name)!=camera[key] or evaluation[key]!=camera[key]:raise ValueError('Recorded policy artifact differs')
        if (sha(videos/'manifest.json')!=complete['video_manifest_sha256']
                or sha(videos/'decode_verification.json')!=complete['decode_verification_sha256']):
            raise ValueError('Completed encoder/decode receipt changed')
        episodes=[json.loads(line) for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        encoded=read(videos/'manifest.json')['videos'];decoded=read(videos/'decode_verification.json')['videos']
        ids=camera['original_case_ids'];count=len(ids)
        if (len(set(ids))!=count or len(episodes)!=count or len(encoded)!=count or len(decoded)!=count
                or len(evaluation['episodes'])!=count
                or sorted(r['case_id'] for r in evaluation['episodes'])!=list(range(count))
                or sorted(r['case_id'] for r in episodes)!=list(range(count))
                or sorted(r['case_id'] for r in encoded)!=list(range(count))
                or len({r['file'] for r in encoded})!=count or len({r['file'] for r in decoded})!=count):
            raise ValueError('Every prespecified scene must have exactly one completed video')
        by_case={r['case_id']:r for r in episodes};by_file={r['file']:r for r in decoded}
        audited_cases={r['case_id']:r for r in evaluation['episodes']}
        for video in encoded:
            episode=by_case[video['case_id']]
            if any(episode[k]!=audited_cases[video['case_id']][k] for k in ('success','termination_reason','physics_steps')):
                raise ValueError('Video episode differs from raw-audited outcome')
            row=bind_video(batch.name,method,protocol['seed'],ids,video,episode,camera['model_sha256'],camera['normalization_sha256'])
            path=videos/video['file']
            if not path.resolve().is_relative_to(videos.resolve()) or sha(path)!=video['sha256']:
                raise ValueError('Video path or bytes changed')
            decode=by_file.get(video['file'])
            if (decode is None or decode['sha256']!=video['sha256'] or decode['decoded_frames']!=video['frame_count']
                    or abs(decode['fps']-video['fps'])>.001
                    or abs(video['fps']*episode['recording']['sensor_tick']-1)>1e-6):
                raise ValueError('Decoded video is not the same complete recording')
            frame_manifest,frames=verified_frames(folder,episode['recording'])
            if (sha(frame_manifest)!=video['frame_manifest_sha256'] or len(frames)!=video['frame_count']
                    or frames[0]['frame']!=video['first_world_frame'] or frames[-1]['frame']!=video['last_world_frame']
                    or len(frames)!=episode['physics_steps']+1):
                raise ValueError('RGB frames and actual executed episode disagree')
            row.update(source_file_root_relative=path.relative_to(root).as_posix(),
                source_parent_training_batch=parent.name,source_parent_case_id=ids[video['case_id']],
                retained_policy_training_controls=summary['lineage_control_steps'],
                retained_PPO_Adam_steps=protocol['recovery'][method]['retained_optimizer_steps'],
                frame_manifest_sha256=video['frame_manifest_sha256'],frame_count=len(frames),fps=video['fps'],
                first_world_frame=frames[0]['frame'],last_world_frame=frames[-1]['frame'],
                raw_audit_sha256=sha(folder/'endpoint_raw_audit.json'))
            records.append(row)
        proofs.append({'batch':batch.name,'method':method,'source_video_manifest_sha256':sha(videos/'manifest.json'),
            'decode_verification_sha256':sha(videos/'decode_verification.json'),
            'contact_sheet_sha256':sha(videos/'contact_sheet.png'),'parent_raw_audit_sha256':camera['parent_raw_audit_sha256']})
    if len({r['video_id'] for r in records})!=len(records) or len({r['release_asset_name'] for r in records})!=len(records):
        raise ValueError('Duplicate qualified recording identity')
    result={'scope':'Supplementary identity catalog for every registered real native PPO RGBrecording;original manifests preserved. Known illustrative conditions,not final confirmation.',
        'execution':execution.name,'script_sha256':sha(Path(__file__)),'plan_sha256':sha(execution/'plan.json'),
        'queue_complete_status_sha256':sha(execution/'status.json'),'videos':records,'proofs':proofs,
        'visual_inspection':'Separate human/agent image inspection is required;this catalog attests identities,raw frames and decode receipts only.',
        'new_controls':0,'optimizer_updates':0,'original_assets_changed':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=catalog(args.execution,args.output)
    print(json.dumps({'videos':len(result['videos']),'new_controls':0,'original_assets_changed':False}))
