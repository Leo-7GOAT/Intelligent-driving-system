"""Freeze native20Hz trained policies for10/4Hz command-update stress probes."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.control_cadence import VERSION
from navstack.experiments import signature,write_json
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.prepare_native_shared_brake import require_closed_method_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import copy_walker_spawn_harness
from scripts.prepare_native_training_view_pair import evaluation_input_protocol


def cadence_record(original,method,hold):
    config=EnvConfig.from_dict(original['env'])
    if (method not in ('H','B1') or type(hold) is not int or hold not in (2,5)
            or config.shared_shield or config.control_dt!=.05 or config.decision_dt!=.5
            or method=='B1' and config.flat_hold!=1):
        raise ValueError('Require the native20Hz control profile and declared10/4Hz intervention')
    if method=='B1':config.flat_hold=hold;config.__post_init__()
    return {**deepcopy(original),'env':config.to_dict(),'signature':signature(config),
            'deployment_control_cadence':{'version':VERSION,'hold_steps':hold,
                'physics_seconds':.05,'command_update_seconds':hold*.05,
                'H_policy_decision_seconds':.5,'MPC_prediction_grid_seconds':.05,
                'MPC_horizon_steps':config.mpc_horizon,'MPC_horizon_seconds':.05*config.mpc_horizon,
                'MPC_design_applicable':method=='H','MPCdesign_retuned':False,'training_updates':0}}


def environment_overlay(original):
    block=('        self.controller = MPC(\n'
           '            self.config.vehicle, self.config.control_dt, self.config.mpc_horizon\n'
           '        )\n')
    if original.count(block)!=1 or 'apply_controller_cadence' in original:
        raise ValueError('Native controller initialization structure differs')
    return original.replace(block,block+
        '        from .control_cadence import apply_controller_cadence\n'
        '        self.controller = apply_controller_cadence(self.controller,self.config)\n')


def worker_overlay(original):
    if 'deployment_control_cadence' in original:raise ValueError('Cadence override already present')
    lines=original.splitlines(keepends=True)
    functions=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='training_config']
    if len(functions)!=1:raise ValueError('Require one native configuration function')
    function=functions[0];body=''.join(lines[function.lineno-1:function.end_lineno])
    if body.count('    return config')!=1:raise ValueError('Unknown configuration return')
    body=body.replace('    return config',
        '    from navstack.control_cadence import configure_cadence\n'
        '    configure_cadence(config,protocol,method)\n    return config')
    changed=''.join(lines[:function.lineno-1])+body+''.join(lines[function.end_lineno:])
    before='"seed": protocol["seed"], "backend": "CARLA_0.9.15", "scope": protocol["scope"]})'
    after='"seed": protocol["seed"], "backend": "CARLA_0.9.15", "scope": protocol["scope"], "deployment_control_cadence": protocol["deployment_control_cadence"]})'
    if changed.count(before)!=1:raise ValueError('Unknown native resolved-config record')
    changed=changed.replace(before,after)
    def without_profile(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='training_config']
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                pairs=[(k,v) for k,v in zip(node.keys,node.values) if not isinstance(k,ast.Constant) or k.value!='deployment_control_cadence']
                node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
        return ast.dump(tree,include_attributes=False)
    if without_profile(original)!=without_profile(changed):raise ValueError('Cadence overlay changes another worker behavior')
    return changed


def prepare(parent,method,wall_seconds=3600):
    parent=Path(parent).resolve()
    if method not in ('H','B1') or type(wall_seconds) is not int or not 900<=wall_seconds<=3600:
        raise ValueError('Bounded H/B1deployment probe required')
    require_closed_method_parent(parent)
    old=read(parent/'protocol.json')
    if method not in old['methods'] or old.get('evaluation_only'):raise ValueError('Require an actually PPOtrained native parent')
    raw=audit(parent,method)
    if raw!=read(parent/method/'endpoint_raw_audit.json'):raise ValueError('Complete parent raw audit changed')
    inventory=old.get('native_validation',{}).get('manifest',old.get('native_case_manifest'))
    expected=len(inventory['cases']) if inventory is not None else len(old['routes'])
    episodes=read(parent/method/'evaluation.json')['episodes']
    if len(episodes)!=expected or {r['case_id'] for r in episodes}!=set(range(expected)):
        raise ValueError('Require all native parent conditions,including failures')
    folder=parent/method;summary=read(folder/'summary.json')
    origin=folder/read(folder/'latest_checkpoint.json')['folder'];metadata=verified_checkpoint_metadata(origin)
    if (summary['stopped_reason'] is not None or not summary['reload_exact']
            or summary['physics_steps']<=0 or summary['optimizer_steps']<=0
            or any(metadata[k]!=summary[k] for k in ('physics_steps','optimizer_steps','lineage_control_steps'))
            or sha(folder/'model.zip')!=metadata['model_sha256']):
        raise ValueError('Require the complete trained endpoint,not a selected intermediate model')
    records={hold:cadence_record(read(origin/'config.json'),method,hold) for hold in (2,5)}
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed:'+name)
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:raise ValueError('Parent calibration changed')
    changes={'navstack/env.py':environment_overlay((parent/'source/navstack/env.py').read_text(encoding='utf-8-sig')),
             'scripts/native_road_learning.py':worker_overlay((parent/'source/scripts/native_road_learning.py').read_text(encoding='utf-8-sig'))}
    with ProcessLease(parent/(method+'_cadence_prepare.lock')):
        registration=parent/(method+'_control_cadence_registration.json')
        if registration.exists():raise FileExistsError('Do not repeat a registered preparation')
        entries=[];atomic_json(registration,{'status':'PREPARING','batches':entries})
        try:
            for hold in (2,5):
                healthy,reason=resources_ok(ROOT)
                if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
                batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_control_hold'+str(hold))
                batch.mkdir();(batch/'infrastructure').mkdir();entries.append({'hold_steps':hold,'batch':batch.name})
                atomic_json(registration,{'status':'PREPARING','batches':entries})
                shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copytree(parent/'calibration',batch/'calibration');destination=batch/'resume'/method
                destination.parent.mkdir();shutil.copytree(origin,destination)
                for name,text in changes.items():(batch/'source'/name).write_text(text,encoding='utf-8')
                for name in ('navstack/control_cadence.py','scripts/prepare_native_control_cadence.py'):
                    shutil.copy2(ROOT/name,batch/'source'/name)
                write_json(destination/'config.json',records[hold])
                for name in ('model.zip','normalization.pkl','resume_state.pkl'):
                    if sha(destination/name)!=sha(origin/name):raise ValueError('Frozen learning bytes changed')
                verified_checkpoint_metadata(destination)
                if old.get('extra_supervision'):
                    dataset=parent/'measured_demonstrations.npz'
                    if sha(dataset)!=old['extra_supervision']['dataset_sha256']:raise ValueError('Teacher input changed')
                    shutil.copy2(dataset,batch/dataset.name)
                bridge='scripts/carla_bridge37.py'
                cleanup=copy_walker_spawn_harness(parent/'source'/bridge,ROOT/bridge,batch/'source'/bridge)
                for name in ('guard_final.json','failure.json','launcher_failure.json'):
                    if (parent/name).exists():shutil.copy2(parent/name,batch/('parent_'+name))
                write_json(batch/'parent_protocol.json',old);write_json(batch/'parent_training_summary.json',summary)
                protocol=evaluation_input_protocol(old)
                now=time.time();prior=old.get('recovery',{}).get(method,{})
                profile=records[hold]['deployment_control_cadence']
                protocol.update(methods=[method],evaluation_only=True,controlled_budget_per_method=0,
                    method_budgets={method:0},checkpoint_thresholds=[],created_epoch=now,deadline_epoch=now+wall_seconds,
                    parent_batch=parent.name,walker_spawn_initialization=cleanup,
                    recovery={method:{**deepcopy(prior),'checkpoint':'resume/'+method,'parent_batch':parent.name,
                        'parent_checkpoint':origin.name,'lineage_controls':summary['lineage_control_steps'],
                        'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
                        'retained_optimizer_steps':prior.get('retained_optimizer_steps',0)+summary['optimizer_steps']}},
                    deployment_control_cadence={**profile,'baseline_parent':parent.name,'baseline_raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
                        'model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256'],
                        'baseline_reused_without_new_driving':True,'native_safety_gates_unchanged':True,
                        'limits':'Frozen-policy deployment sensitivity. Hkeeps2Hz PPOand20Hz MPCprediction grid;only commanded updates become10/4Hz. B1direct policy updates become10/4Hz. No controller retuning,retraining,safety certification or independent new training seeds.'},
                    scope='Native control-cadence DEVELOPMENT sensitivity;all registered conditions,zero learning;fixed20Hz CARLAphysics')
                if method=='B1':protocol['flat_hold']=hold
                protocol['resource_limits']['wall_seconds']=wall_seconds
                protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
                write_json(batch/'protocol.json',protocol);entries[-1]['protocol_sha256']=sha(batch/'protocol.json')
                write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':entries[-1]['protocol_sha256']})
            atomic_json(registration,{'status':'BOTH_CADENCES_PREPARED_NOT_STARTED','batches':entries})
        except BaseException as error:
            atomic_json(registration,{'status':'FAILED_PARTIALS_RETAINED','error':repr(error),'batches':entries})
            raise
    return entries


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--method',required=True,choices=['H','B1'])
    parser.add_argument('--wall-seconds',type=int,default=3600)
    args=parser.parse_args();prepare(args.parent,args.method,args.wall_seconds)
