"""Run the strong rule with the learned system's exact physical lowlevel."""
import argparse
import ast
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def without_functions(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
    return ast.dump(tree,include_attributes=False)


def prepare(parent,template,output,phase):
    from scripts.run_outcome_development_validation import prepare as frozen_prepare
    from scripts.audit_native_method_endpoint import read,sha
    from navstack.research import atomic_json
    from navstack.shared_rule_registration import VERSION,registered
    parent,output=Path(parent).resolve(),Path(output).resolve()
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    if gpu<6144:raise ValueError('Disjoint B3 requires at least6GiB currently free GPU memory')
    output=frozen_prepare(parent,template,output,phase)
    p=read(output/'protocol.json');task=p['fixed_outcome_development_validation'];config=read(parent/'H/config.json')['env']
    config={**config,'method':'B3'}
    for key in ('recovery','outcome_initialization','outcome_context_registration','reference_learning_registration',
                'fixed_lowlevel_adaptation_registration','fixed_outcome_development_validation',
                'record_initial_parameters','reference_entropy_followup_registration'):
        p.pop(key,None)
    p.update(methods=['B3'],method_budgets={'B3':0},controlled_budget_per_method=0,
        evaluation_only=True,training_physics_steps=0,optimizer_updates=0,policy_kind='rule_only_B3',
        rule='strong_catalog_with_failed_action_preference_v1',rule_speed_policy='native_score',config=config,
        source_method='H',
        parent_batch=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
        scope='Strong rule on complete known development conditions;shared fixed lowlevel,no neural policy or PPO')
    p.update(carla_rpc_port=2110,carla_rpc_port_span=[2110,2111,2112],allow_disjoint_carla=True)
    p['fixed_shared_rule']=dict(version=VERSION,failed_action_penalty_m=5.,
        goal_wait_requires_heading=True,
        phase=phase,expected_cases=30 if phase=='placements30' else 24,
        template=task['template'],template_protocol_sha256=task['template_protocol_sha256'],
        information='Same current candidate geometry,measured outcomes and failed-label history;no future actor truth',
        scoring='Existing goal/path/speed/continuation scoring plus finite5m failed-label preference;no failure mask',
        compute='All candidate connectors and up to3nominal searches charged;selected action always uses the same measured-steering production pipeline',
        PPO_environment_config_exact_except_method=True,selection='All prespecified known cases;no score-based case selection')
    p['fixed_shared_rule']['resource_admission']=dict(free_GPU_MiB=gpu,minimum_free_GPU_MiB=6144,
        maximum_simultaneous_servers=2,unowned_peer_never_stopped=True)
    p['fixed_shared_rule']['source_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(output/'source_config.json',dict(env=config,scope='Exact parent environment except B3 method label'))
    p['source_config_sha256']=sha(output/'source_config.json')
    atomic_json(output/'case_manifest.json',p['native_case_manifest'])
    p['case_manifest_sha256']=sha(output/'case_manifest.json')
    allowed={'navstack/env.py':{'rule_action'},'navstack/reference_road_refinement.py':{'settings'},
        'navstack/steering_transition.py':{'settings'},'navstack/reference_option_commitment.py':{'settings'},
        'navstack/reference_compute.py':{'enabled'}}
    for name,functions in allowed.items():
        if without_functions(ROOT/name,functions)!=without_functions(parent/'source'/name,functions):
            raise ValueError('Shared physical code differs outside rule/registration:'+name)
        shutil.copy2(ROOT/name,output/'source'/name)
    p['fixed_shared_rule']['unchanged_lowlevel_AST_except']= {name:sorted(functions) for name,functions in allowed.items()}
    for name in ('navstack/shared_rule_registration.py','scripts/fixed_shared_native_rule.py','scripts/evaluate_native_rule.py',
                 'scripts/run_carla_owned_foreground.ps1','scripts/run_registered_native.ps1'):
        shutil.copy2(ROOT/name,output/'source'/name)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*')
        if f.is_file() and '__pycache__' not in f.parts}
    if not registered(p):raise ValueError('Incomplete zero-learning shared rule registration')
    from navstack.reference_road_refinement import settings
    settings(p)
    atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    return output


def run(batch):
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import evaluate_cases,require_finished_parent
    from scripts.carla_steering_audit import guard
    from navstack.carla_bridge import BridgeClient
    from navstack.config import EnvConfig
    from navstack.env import rule_action
    from navstack.native_cases import registered_native_cases
    from navstack.planner_memo import configure_planner_memo
    from navstack.research import atomic_json
    from navstack.shared_rule_registration import registered
    from scripts.native_road_learning import native_environment_class
    batch=Path(batch).resolve();p=read(batch/'protocol.json')
    if not registered(p):raise ValueError('Only registered zero-learning B3 is admitted')
    parent=(batch.parent/p['parent_batch']).resolve()
    if parent.parent!=batch.parent:raise ValueError('Parent must stay in the same explicit archive')
    require_finished_parent(parent)
    if sha(parent/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    for name,digest in p['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Execute only the frozen source:'+name)
    if (sha(batch/'source_config.json')!=p['source_config_sha256']
            or read(batch/'source_config.json')['env']!=p['config']
            or sha(batch/'calibration/calibration.json')!=p['calibration_sha256']):
        raise ValueError('Shared physical configuration changed')
    guard(batch,p)
    with (batch/'started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(epoch=time.time(),policy_kind='rule_only_B3'),stream)
    client=env=None;started=time.time()
    try:
        client=BridgeClient(batch)
        cls=native_environment_class(p['reference_context']['version'],recorded=False)
        env=cls(EnvConfig.from_dict(p['config']),client)
        configure_planner_memo(env,p)
        cases=registered_native_cases(p,'evaluation') if p.get('native_validation') else registered_native_cases(p)
        episodes,controls=evaluate_cases(env,cases,batch,lambda:guard(batch,p),
            action_selector=partial(rule_action,failed_action_penalty=p['fixed_shared_rule']['failed_action_penalty_m'],
                require_goal_heading=True))
        if len(episodes)!=p['fixed_shared_rule']['expected_cases']:raise ValueError('Incomplete known rule inventory')
        atomic_json(batch/'summary.json',dict(status='complete',policy_kind='rule_only_B3',scope=p['scope'],
            episodes=episodes,evaluation_controls=controls,world_ticks=client.total_ticks,
            reset_ticks=client.total_ticks-controls,training_physics_steps=0,optimizer_updates=0,
            wall_seconds=time.time()-started))
    except BaseException as error:
        atomic_json(batch/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:
        if env is not None:env.close()
        if client is not None:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','template','output','phase'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();prepare(args.parent,args.template,args.output,args.phase)
