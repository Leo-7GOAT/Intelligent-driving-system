"""Replay exact recorded static inputs with opt-in search/refinement shortcuts."""
import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.independent_clip_ppo import load_ppo
from navstack.reference_compute import VERSION
from navstack.reference_context import reference_candidate_context
from navstack.reference_option_commitment import configure,settings as option_settings
from navstack.reference_road_refinement import prepare_reference as road_prepare,settings as road_settings,pose_digest
from navstack.steering_transition import prepare_reference as steer_prepare,settings as steer_settings
from navstack.vehicle import brake,VehicleState
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha,_verify_ticks,verify_initial_actuator
from scripts.audit_native_policy_sequence import observation_env,advance_observation


def probe(batch,case,decisions,output):
    import torch
    torch.set_num_threads(1);batch=Path(batch).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Preserve each compute equivalence attempt')
    p=read(batch/'protocol.json');rows,error=read_prefix(batch/'H/evaluation_trace.jsonl.gz')
    rows=[r for r in rows if r['case_id']==case]
    if not rows or 'episode_metrics' not in rows[-1]['info']:raise ValueError('Require a complete retained case')
    rows=rows[:decisions];episode=rows[0]['info']['trace'][0]['backend_evidence']['episode_id']
    actual={};acks={};begin=None;streams=[]
    for path in (batch/'infrastructure').glob('*_responses.jsonl.gz'):
        records,stream_error=read_prefix(path);streams.append(dict(name=path.name,sha256=sha(path),prefix_error=stream_error))
        for record in records:
            r=record['response']
            if r.get('episode_id')!=episode:continue
            acks[(episode,r['request_id'])]=r
            if 'scene' in r:begin=r
            if 'vehicle_control' in r:actual[(episode,r['frame'])]=r
    if begin is None or begin['dynamic']:raise ValueError('Require the actual static reset')
    verify_initial_actuator(rows[0]['info']['initial_scenario'],episode,acks)
    cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    used=set();previous={};episodes=set()
    for row in rows:_verify_ticks(row['info']['trace'],actual,acks,{},used,previous,episodes,cfg.actuator_observation)
    env=observation_env(cfg,begin['scene'],rows[0]['info']['initial_scenario'],begin=begin)
    env.scene_evidence=begin['scene'];env.world.road_collision_backend=cfg.road_collision_backend
    env.reference=None;env._reference_selection=None;env._rule_plan_cache=None;env.episode=dict(plan_times=[],plan_failures=0)
    new=deepcopy(p);new['reference_compute_profile']={'version':VERSION}
    base=env._prepare_reference;road_options=road_settings(new);steer_options=steer_settings(new)
    current={};calls=[]
    def validate(poses):
        digest=pose_digest(poses)
        reply=next((r for r in current['row']['info']['plan']['road_refinement']['probes']
                    if r['reference_road_probe']['pose_sha256']==digest),None)
        if reply is None or reply['frame']!=env._measured_frame or reply!=acks.get((episode,reply['request_id'])):
            raise ValueError('New reference query differs from actual recorded geometry/frame')
        calls.append(digest);return reply
    def prepare(action):
        return road_prepare(env,action,lambda a:steer_prepare(env,a,base,steer_options),validate,road_options)
    env._prepare_reference=prepare;configure(env,option_settings(new))
    model=load_ppo(batch/'H/model.zip',device='cpu')
    with (batch/'H/normalization.pkl').open('rb') as f:norm=pickle.load(f)
    norm.training=False;results=[]
    try:
        for index,row in enumerate(rows):
            current['row']=row
            obs=np.r_[env._observation(),reference_candidate_context(env)].astype(np.float32)
            action=int(model.predict(norm.normalize_obs(obs[None]),deterministic=True)[0][0])
            if action!=int(row['policy_action']):raise ValueError('NN action differs')
            start=time.perf_counter();mode,detail=env._prepare_reference(action);seconds=time.perf_counter()-start
            old=row['info']['plan'];original=old['road_refinement'];fresh=detail['road_refinement']
            if mode!=row['info']['mode'] or fresh['cells']!=original['cells']:raise ValueError('Physical execution mode or planning constraints differ')
            if mode!='plan_failed' or env.reference is not None:raise ValueError('This bounded proof covers the selected repeated-rejection failure only')
            state=env.state
            for tick in row['info']['trace']:
                command=brake(state,cfg.vehicle)
                if tick['requested_command']!={'acceleration':command.acceleration,'steering':command.steering}:
                    raise ValueError('Requested brake differs')
                state=VehicleState(**tick['state'])
            results.append(dict(decision=index,policy_action=action,mode=mode,requested_command_exact=True,
                constraint_cells_exact=True,all_new_reference_poses_match_actual_queries=True,
                old_search_nodes=original['total_expanded_nodes'],new_search_nodes=fresh['total_expanded_nodes'],
                old_queries=len(original['probes']),new_queries=len(fresh['probes']),
                replay_prepare_seconds=seconds,shortcut=fresh.get('shortcut')))
            advance_observation(env,row)
    finally:env.close()
    return dict(scope=__doc__,batch=batch.name,case_id=case,model_sha256=sha(batch/'H/model.zip'),
        protocol_sha256=sha(batch/'protocol.json'),script_sha256=sha(Path(__file__)),
        compute_source_hashes={name:sha(ROOT/name) for name in ['navstack/planner_memo.py','navstack/reference_compute.py',
            'navstack/reference_road_refinement.py','navstack/steering_transition.py','navstack/env.py',
            'navstack/reference_option_commitment.py','navstack/planning.py']},
        actual_control_frames_bound=len(used),bridge_prefix=streams,live_trace_prefix_error=error,
        results=results,new_controls=0,new_ticks=0,new_PPO_updates=0,
        limits='Read-only exact-prefix equivalence for this recorded static case,not a new native latency/performance score. Native query CPUtime is excluded because actual replies are replayed. Active trace/streams may contain an unfinished later prefix;only bound complete selected-case frames are used.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--decisions',type=int,default=2);p.add_argument('--output',required=True);a=p.parse_args()
    r=probe(a.batch,a.case,a.decisions,a.output);atomic_json(a.output,r)
    print(dict(decisions=len(r['results']),bound_controls=r['actual_control_frames_bound'],
        old_nodes=sum(x['old_search_nodes'] for x in r['results']),new_nodes=sum(x['new_search_nodes'] for x in r['results']),
        old_queries=sum(x['old_queries'] for x in r['results']),new_queries=sum(x['new_queries'] for x in r['results'])))
