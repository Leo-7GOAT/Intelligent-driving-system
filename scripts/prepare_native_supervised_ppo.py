"""Start real PPO from a disclosed, evaluated teacher-supervised initialization."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pickle
import zipfile
from navstack.experiments import write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_hazard_learning import apply_guarded_progress
from scripts.native_road_learning import training_config
from navstack.config import EnvConfig
from navstack.experiments import signature


def initial_exploration(model, log_std):
    if (not -4. <= log_std <= -1. or model.num_timesteps or model._n_updates
            or model.policy.optimizer.state or not hasattr(model.policy, 'log_std')
            or tuple(model.policy.log_std.shape) != (2,)):
        raise ValueError('Only untouched PPO optimizer/interaction state from continuous supervised initialization is supported')
    before = {name: value.detach().clone() for name, value in model.policy.named_parameters()}
    with torch.no_grad():
        model.policy.log_std.fill_(log_std)
    changed = [name for name, value in model.policy.named_parameters() if not torch.equal(before[name], value)]
    if changed != ['log_std']:
        raise ValueError('Exploration initialization changed parameters beyond log_std')
    return {'parameter_changes': changed, 'previous_log_std': before['log_std'].tolist(),
            'initial_log_std': model.policy.log_std.detach().tolist(), 'mean_and_critic_unchanged': True}


def categorical_exploration(model,normalizer,observations,target_probability=.9):
    """Soften learned logits, preserving their deterministic ordering.

    Calibration uses teacher TRAINING views only, not evaluation outcomes.
    """
    if (not .5<=target_probability<1. or model.num_timesteps or model._n_updates or model.policy.optimizer.state
            or hasattr(model.policy,'log_std')):
        raise ValueError('Require untouched categorical supervised initialization')
    head=getattr(model.policy.action_net,'head',model.policy.action_net)
    if not isinstance(head,torch.nn.Linear):raise ValueError('Unsupported categorical head')
    before={name:value.detach().clone() for name,value in model.policy.named_parameters()}
    model.policy.set_training_mode(False)
    inputs=torch.as_tensor(normalizer.normalize_obs(observations),dtype=torch.float32,device=model.device)
    with torch.no_grad():
        logits=model.policy.get_distribution(inputs).distribution.logits.detach()
        original_actions=logits.argmax(1)
        initial=float(logits.softmax(1).amax(1).mean())
        lo,hi=1.,64.
        if initial>target_probability:
            for _ in range(32):
                middle=(lo+hi)/2
                if float((logits/middle).softmax(1).amax(1).mean())>target_probability:lo=middle
                else:hi=middle
            temperature=hi
        else:temperature=1.
        head.weight.div_(temperature)
        if head.bias is not None:head.bias.div_(temperature)
        distribution=model.policy.get_distribution(inputs).distribution
        if not torch.equal(original_actions,distribution.probs.argmax(1)):
            raise ValueError('Positive logit scaling changed deterministic calibration actions')
        actual=float(distribution.probs.amax(1).mean())
        if initial>target_probability and abs(actual-target_probability)>1e-4:
            raise ValueError('Requested categorical exploration probability is outside the bounded calibration range')
    allowed={id(p) for p in head.parameters()}
    changed=[name for name,value in model.policy.named_parameters() if not torch.equal(before[name],value)]
    if any(id(value) not in allowed and not torch.equal(before[name],value) for name,value in model.policy.named_parameters()):
        raise ValueError('Categorical calibration changed encoder/critic parameters')
    return {'parameter_changes':changed,'temperature':temperature,'target_probability':target_probability,
            'initial_mean_max_probability':initial,'initialized_mean_max_probability':actual,
            'calibration_examples':len(observations),'calibration_split':'teacher_training_only',
            'deterministic_calibration_actions_unchanged':True,'encoder_and_critic_unchanged':True}


def fresh_learning_rate(model,rate):
    if (not np.isfinite(rate) or not 1e-6<=rate<=1e-3 or model.num_timesteps
            or model._n_updates or model.policy.optimizer.state):
        raise ValueError('Learning-rate tuning requires an untouched PPO optimizer')
    previous=[group['lr'] for group in model.policy.optimizer.param_groups]
    model.learning_rate=float(rate);model._setup_lr_schedule()
    model.policy.optimizer.defaults['lr']=float(rate)
    for group in model.policy.optimizer.param_groups:group['lr']=float(rate)
    return {'previous_optimizer_learning_rates':previous,'initialized_learning_rate':float(rate),
            'scope':'Constant PPO schedule and optimizer initialization only; no model parameter or update changes'}


def fresh_rollout(path, actions=None):
    """Rebuild an empty SB3 buffer before the first PPO update, preserving weights."""
    original=load_ppo(path,device='cpu')
    if actions is None:return original,None
    if (type(actions) is not int or not 2<=actions<=4096
            or actions%original.batch_size or actions==original.n_steps
            or original.num_timesteps or original._n_updates or original.policy.optimizer.state):
        raise ValueError('Require a changed bounded rollout, saved minibatch multiple, and untouched PPO state')
    changed=load_ppo(path,device='cpu',rollout_actions=actions,allow_rollout_intervention=True)
    if (changed.rollout_buffer.buffer_size!=actions or changed.rollout_buffer.full
            or changed.rollout_buffer.pos or changed.policy.optimizer.state
            or any(not torch.equal(value,changed.policy.state_dict()[name])
                   for name,value in original.policy.state_dict().items())):
        raise ValueError('Rollout initialization changed policy/optimizer or failed to rebuild an empty buffer')
    return changed,{'previous_rollout_actions':original.n_steps,'initialized_rollout_actions':actions,
                    'batch_size':changed.batch_size,'ppo_epochs':changed.n_epochs,
                    'policy_parameters_exactly_unchanged':True,'ppo_optimizer_untouched':True,
                    'scope':'Changed collection/update interval only; same minibatch size and epochs. '
                            'A common full physical budget divisible by both lengths has equal Adam step count.'}


def guarded_initialization(protocol,metadata_config,method='B1'):
    """Explicit new reward version before ANY PPO update; never rewrite parent."""
    original=EnvConfig.from_dict(metadata_config['env'])
    if metadata_config['signature']!=signature(original):
        raise ValueError('Supervised initialization signature is inconsistent')
    apply_guarded_progress(protocol,original.to_dict())
    changed=training_config(protocol,method,original.vehicle)
    before,after=original.to_dict(),changed.to_dict()
    expected=dict(before['reward']);expected.update(progress_weight=.5,collision=-90.,out_of_bounds=-90.)
    if {k:v for k,v in before.items() if k!='reward'}!={k:v for k,v in after.items() if k!='reward'} or after['reward']!=expected:
        raise ValueError('Reward initialization changes unrelated task/control fields')
    return {**metadata_config,'env':after,'signature':signature(changed),
            'scope':'Same supervised actor and untouched PPO optimizer, explicitly initialized for guarded_progress_v1 before training'}


def representation_inventory(protocol,evaluated):
    from navstack.native_cases import registered_native_cases
    cases=registered_native_cases(protocol) if protocol.get('native_case_manifest') else None
    count=len(cases) if cases is not None else 4
    episodes=evaluated['episodes']
    if (len(episodes)!=count or any(type(r.get('case_id')) is not int or type(r.get('success')) is not bool for r in episodes)
            or {r['case_id'] for r in episodes}!=set(range(count))):
        raise ValueError('Representation gate requires every registered condition exactly once')
    return cases,episodes,count


def representation_gate(protocol,evaluated):
    """Keep the existing75% development gate over the complete actual inventory."""
    cases,episodes,count=representation_inventory(protocol,evaluated)
    required=(3*count+3)//4;successes=sum(r['success'] for r in episodes)
    if successes<required:
        raise ValueError('Require the predeclared75percent representation gate for EVERY registered method')
    families={}
    for row in episodes:
        spec=cases[row['case_id']].get('native_hazard') if cases is not None else None
        family=('known_roads' if cases is None else 'curve_parking' if spec is None
                else 'static_obstacle' if spec['kind']=='static_vehicle' else 'dynamic_interaction')
        totals=families.setdefault(family,{'successes':0,'cases':0})
        totals['cases']+=1;totals['successes']+=row['success']
    return {'successes':successes,'cases':count,'required_successes':required,'minimum_fraction':.75,
            'families':families,'scope':'Complete development initialization gate, not research acceptance or independent confirmation'}


def basic_motion_training_gate(protocol,evaluated):
    """Explicit B1 learning-entry profile, never full-task qualification."""
    cases,episodes,count=representation_inventory(protocol,evaluated)
    if protocol.get('methods')!=['B1'] or cases is None:
        raise ValueError('Basic-motion entry requires explicit complete B1 development conditions')
    basic=[row for row in episodes if cases[row['case_id']].get('native_hazard') is None]
    required=(3*len(basic)+3)//4
    if (len(basic)<4 or sum(row['success'] for row in basic)<required
            or any(not isinstance(row.get('termination_reason'),str) for row in basic)
            or any(row.get('termination_reason') in ('collision','out_of_bounds') for row in basic)):
        raise ValueError('Require75percent unobstructed motion success with zero basic-motion unsafe episodes')
    return {'profile':'basic_motion_training_v1','basic_successes':sum(r['success'] for r in basic),
        'basic_cases':len(basic),'required_basic_successes':required,'minimum_basic_fraction':.75,
        'full_task_successes':sum(r['success'] for r in episodes),'full_task_cases':count,
        'full_task_qualification_claimed':False,
        'scope':'PPO learning-entry sanity only. Full-task initialization may remain unqualified; no relaxation of final research acceptance or healthy-final-baseline claim.'}


def copy_cleanup_harness(archived,current,destination):
    """Allow only Bridge.clear lifecycle changes, not hidden actuation changes."""
    import ast
    def signature(path):
        tree=ast.parse(Path(path).read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='Bridge':
                node.body=[member for member in node.body if not isinstance(member,ast.FunctionDef) or member.name!='clear']
        return ast.dump(tree,include_attributes=False)
    if signature(archived)!=signature(current):
        raise ValueError('Cleanup harness override changes code outside Bridge.clear')
    change={'parent_sha256':sha(archived),'initialized_sha256':sha(current),
        'scope':'Bridge.clear only; actuation, sensing, scenario generation and policy semantics unchanged by AST comparison'}
    shutil.copy2(current,destination)
    return change


def supervised_input_cost(protocol,method):
    costs=[protocol['extra_supervision']['teacher_controls'],
           protocol['recovery'][method]['consumed_controls_lower_bound']]
    if any(type(value) is not int or value<0 for value in costs):
        raise ValueError('Supervised physical input costs must be nonnegative integers')
    return max(costs)


def fresh_target_kl(model, target_kl):
    """Register SB3's existing within-rollout KL stop before any PPO updates."""
    if (isinstance(target_kl,bool) or not np.isfinite(target_kl) or not .001<=target_kl<=.05
            or model.num_timesteps or model._n_updates or model.policy.optimizer.state):
        raise ValueError('Bounded KL tuning requires an untouched PPO optimizer')
    previous=model.target_kl
    model.target_kl=float(target_kl)
    return {'previous_target_kl':previous,'initialized_target_kl':float(target_kl),
            'scope':'Existing SB3 per-rollout epoch early stop; physical budget and endpoint unchanged. Not a hard bound on cumulative policy drift.'}


def prepare(parent, controls=20480, wall_seconds=7200, log_std=-2.3, guarded_progress=False,categorical_probability=.9,learning_rate=None,rollout_actions=None,initialization_gate='complete_inventory_v1',current_cleanup_harness=False,target_kl=None,critic_rate_profile=None,partition_inventory=None,ppo_reward_scale=None):
    parent = Path(parent).resolve()
    if type(controls) is not int or not 640 <= controls <= 100000 or not 900 <= wall_seconds <= 7200:
        raise ValueError('Invalid bounded native learning budget')
    if learning_rate is not None and (not np.isfinite(learning_rate) or not 1e-6<=learning_rate<=1e-3):
        raise ValueError('Invalid bounded PPO learning rate')
    if rollout_actions is not None and (type(rollout_actions) is not int or not 2<=rollout_actions<=4096):
        raise ValueError('Invalid bounded PPO rollout')
    if target_kl is not None and (isinstance(target_kl,bool) or not np.isfinite(target_kl) or not .001<=target_kl<=.05):
        raise ValueError('Invalid bounded PPO KL target')
    if critic_rate_profile is not None and (critic_rate_profile!='actor1_critic30_v1' or learning_rate is None or not 1e-6<=learning_rate<=1e-3/30):
        raise ValueError('Critic-rate profile requires its bounded explicit actor rate')
    if ppo_reward_scale is not None and (ppo_reward_scale!=.05 or critic_rate_profile or partition_inventory):
        raise ValueError('Fixed reward-unit diagnostic is a separate single-variable profile at scale.05')
    if initialization_gate not in ('complete_inventory_v1','basic_motion_training_v1'):
        raise ValueError('Unknown explicit development training-entry profile')
    require_finished_parent(parent)
    protocol = read(parent/'protocol.json')
    methods=protocol.get('methods')
    if (methods not in (['B1'],['H','H0']) or not protocol.get('evaluation_only') or not protocol.get('extra_supervision')):
        raise ValueError('Require a complete supervised-only direct or H/H0 diagnostic')
    if critic_rate_profile and methods!=['B1']:raise ValueError('Critic-rate diagnostic is registered for B1only')
    origins={};representation_gates={};parent_evaluation_controls={}
    for method in methods:
        parent_audit=audit(parent,method)
        if parent_audit!=read(parent/method/'endpoint_raw_audit.json'):
            raise ValueError('Parent raw audit changed')
        parent_evaluation_controls[method]=parent_audit['evaluation_control_frames_verified']
        evaluated=read(parent/method/'evaluation.json')
        representation_gates[method]=(basic_motion_training_gate if initialization_gate=='basic_motion_training_v1' else representation_gate)(protocol,evaluated)
        origin=parent/method/read(parent/method/'latest_checkpoint.json')['folder']
        metadata=verified_checkpoint_metadata(origin)
        if metadata['physics_steps'] or metadata['optimizer_steps']:
            raise ValueError('The parent is not a zero-PPO supervised initialization')
        if rollout_actions is not None:
            with zipfile.ZipFile(origin/'model.zip') as archive:learner=json.loads(archive.read('data'))
            if (rollout_actions==learner['n_steps'] or rollout_actions%learner['batch_size']
                    or controls%rollout_actions or controls%learner['n_steps']):
                raise ValueError('Matched rollout diagnostic requires a changed minibatch multiple and a budget divisible by both lengths')
        origins[method]=(origin,metadata)
    protocol['supervised_representation_gates']=representation_gates
    protocol['initialization_gate_profile']=initialization_gate
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    with ProcessLease(parent/'supervised_ppo_prepare.lock'):
        label='supervised_ppo_guarded' if guarded_progress else 'supervised_ppo'
        if learning_rate is not None:label+='_lr'+format(learning_rate,'.8g')
        if rollout_actions is not None:label+='_rollout'+str(rollout_actions)
        if target_kl is not None:label+='_kl'+format(target_kl,'.8g')
        if critic_rate_profile:label+='_critic30_v1'
        if partition_inventory:label+='_partitions_'+sha(Path(partition_inventory))[:16]
        if ppo_reward_scale is not None:label+='_reward_units_005_v1'
        if methods==['B1'] and log_std!=-2.3:label+='_std'+format(log_std,'.8g')
        if methods!=['B1'] and categorical_probability!=.9:label+='_prob'+format(categorical_probability,'.8g')
        if initialization_gate=='basic_motion_training_v1':label+='_basic_motion_v1'
        if current_cleanup_harness:label+='_cleanup_snapshot'
        registration=parent/(label+'_registration.json')
        if registration.exists():
            raise FileExistsError('Do not repeat the registered supervised-PPO continuation')
        for name, digest in protocol['source_hashes'].items():
            if sha(parent/'source'/name) != digest:
                raise ValueError('Parent frozen source changed')
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_supervised_ppo')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration', batch/'calibration')
        torch.set_num_threads(1)
        dataset=parent/'measured_demonstrations.npz'
        if not dataset.is_file() or sha(dataset)!=protocol['extra_supervision']['dataset_sha256']:
            raise ValueError('Registered teacher training views missing or changed')
        shutil.copy2(dataset,batch/'measured_demonstrations.npz')
        arrays=np.load(dataset,allow_pickle=False)
        interventions={};recovery={}
        for method,(origin,metadata) in origins.items():
            destination=batch/'resume'/method;shutil.copytree(origin,destination)
            reward_change=None
            if guarded_progress:
                original_config=read(origin/'config.json')
                changed_config=guarded_initialization(protocol,original_config,method)
                write_json(destination/'config.json',changed_config)
                reward_change={'parent_config_sha256':sha(origin/'config.json'),
                    'initialized_config_sha256':sha(destination/'config.json'),
                    'previous_reward':original_config['env']['reward'],'new_reward':changed_config['env']['reward'],
                    'scope':'New reward version before any PPO update; all non-reward environment fields unchanged'}
            model,rollout_change=fresh_rollout(destination/'model.zip',rollout_actions)
            if method=='B1':change=initial_exploration(model,log_std)
            else:
                with (destination/'normalization.pkl').open('rb') as stream:normalizer=pickle.load(stream)
                change=categorical_exploration(model,normalizer,arrays[method+'_observations'][arrays['train_mask']],categorical_probability)
            if learning_rate is not None:change['learning_rate_initialization']=fresh_learning_rate(model,learning_rate)
            if target_kl is not None:change['target_kl_initialization']=fresh_target_kl(model,target_kl)
            if critic_rate_profile:
                from navstack.critic_rate_policy import promote_fresh_critic_rate
                change['critic_rate_initialization']=promote_fresh_critic_rate(model)
            if rollout_change is not None:change['rollout_initialization']=rollout_change
            original_model_sha256=metadata['model_sha256']
            model.save(destination/'model.zip');metadata['model_sha256']=sha(destination/'model.zip')
            if ppo_reward_scale is not None:
                from navstack.fixed_reward_units import initialize_fixed_reward_units
                with (destination/'normalization.pkl').open('rb') as stream:normalizer=pickle.load(stream)
                change['PPO_reward_units_initialization']=initialize_fixed_reward_units(normalizer,ppo_reward_scale)
                with (destination/'normalization.pkl').open('wb') as stream:pickle.dump(normalizer,stream)
                metadata['normalization_sha256']=sha(destination/'normalization.pkl')
            write_json(destination/'metadata.json',metadata)
            if sha(destination/'normalization.pkl')!=metadata['normalization_sha256']:
                raise ValueError('Supervised observation statistics changed')
            verified_checkpoint_metadata(destination)
            interventions[method]={**change,'parent_model_sha256':original_model_sha256,
                'initialized_model_sha256':metadata['model_sha256'],'supervised_updates_added':0,
                'initialization_evaluation_controls_charged':parent_evaluation_controls[method]}
            if reward_change is not None:interventions[method]['reward_initialization']=reward_change
            prior=protocol['recovery'][method]
            recovery[method]={**prior,'checkpoint':'resume/'+method,'parent_batch':parent.name,
                'parent_checkpoint':origin.name,'lineage_controls':0,
                'consumed_controls_lower_bound':supervised_input_cost(protocol,method)+parent_evaluation_controls[method],
                'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False}
        for name in ('prepare_native_supervised_ppo.py','prepare_native_hazard_learning.py'):
            shutil.copy2(ROOT/'scripts'/name, batch/'source/scripts'/name)
        if critic_rate_profile:
            shutil.copy2(ROOT/'navstack/critic_rate_policy.py',batch/'source/navstack/critic_rate_policy.py')
        if ppo_reward_scale is not None:
            shutil.copy2(ROOT/'navstack/fixed_reward_units.py',batch/'source/navstack/fixed_reward_units.py')
            protocol['PPO_reward_units']={'version':'fixed_physical_reward_units_v1','scale':ppo_reward_scale,
                'scope':'Fixed learner reward units only; every environment reward/component and raw physical trace remains in original task units.'}
        if current_cleanup_harness:
            protocol['cleanup_harness_override']=copy_cleanup_harness(parent/'source/scripts/carla_bridge37.py',
                ROOT/'scripts/carla_bridge37.py',batch/'source/scripts/carla_bridge37.py')
        if partition_inventory:
            from scripts.native_partition_initialization import install_inventory
            install_inventory(protocol,batch/'source',partition_inventory)
            shutil.copy2(partition_inventory,batch/'native_partition_inventory.json')
            shutil.copy2(ROOT/'scripts/native_partition_initialization.py',batch/'source/scripts/native_partition_initialization.py')
        now = time.time()
        protocol.pop('evaluation_only')
        protocol.update(created_epoch=now, deadline_epoch=now+wall_seconds, parent_batch=parent.name,
            controlled_budget_per_method=controls, method_budgets={m:controls for m in methods},
            freeze_observation_statistics=True, checkpoint_thresholds=sorted({min(controls,n) for n in (1280,2560,controls)}),
            scope='Real native PPO after disclosed shared teacher supervision; frozen train-only statistics and explicit exploration initialization; not ordinary from-scratch PPO or formal hierarchy superiority',
            comparison='Same physical teacher source; action spaces, rates, supervised losses, exploration and compute separately disclosed. Not independent confirmation.',
            supervised_ppo_intervention={'methods':interventions,'supervised_updates_added':0,
                'statistics': 'Frozen at teacher training examples; correlated holdout excluded from fitting',
                'comparison_limit': 'Teacher access,20Hz continuous control and exploration differ from old categorical2Hz B1. All must be shared or controlled in formal comparisons.'},
            recovery=recovery)
        if methods==['B1']:protocol['supervised_ppo_intervention'].update(interventions['B1'])
        if initialization_gate=='basic_motion_training_v1':
            protocol['scope']='Real CARLA PPO learning from a basic-motion-qualified but full-task-unqualified B1 initializer; remaining hazards are training objectives. Not a final healthy-baseline or research-acceptance claim.'
        if rollout_actions is not None:
            protocol['rollout_actions']=rollout_actions
            protocol['rollout_length_experiment']={
                'version':'supervised_initial_collection_v1','methods':methods,
                'initialization':{m:interventions[m]['rollout_initialization'] for m in methods},
                'hypothesis':'Longer initial collection can observe complete episode rewards before the first actor update; '
                             'this is a development hypothesis, not evidence of improved driving or reliable bootstrap values.',
                'stopping_rule':'One fixed physical endpoint and all four original cases; no early checkpoint selection.'}
        protocol['resource_limits']['wall_seconds'] = wall_seconds
        protocol['source_hashes'] = {p.relative_to(batch/'source').as_posix(): sha(p)
            for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch/'protocol.json', protocol)
        write_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch/'protocol.json')})
        with registration.open('x', encoding='utf-8') as stream:
            json.dump({'batch': batch.name, 'protocol_sha256': sha(batch/'protocol.json')}, stream, indent=2)
        print(batch, flush=True)
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--controls', type=int, default=20480)
    parser.add_argument('--wall-seconds', type=int, default=7200)
    parser.add_argument('--log-std', type=float, default=-2.3)
    parser.add_argument('--guarded-progress',action='store_true')
    parser.add_argument('--categorical-probability',type=float,default=.9)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--rollout-actions',type=int)
    parser.add_argument('--initialization-gate',choices=['complete_inventory_v1','basic_motion_training_v1'],default='complete_inventory_v1')
    parser.add_argument('--current-cleanup-harness',action='store_true')
    parser.add_argument('--target-kl',type=float)
    parser.add_argument('--critic-rate-profile',choices=['actor1_critic30_v1'])
    parser.add_argument('--partition-inventory',type=Path,help='New versioned training/validation inventory,applied before the first PPO update')
    parser.add_argument('--ppo-reward-scale',type=float,choices=[.05],help='Separate fixed1/20learner-reward units diagnostic; raw task rewards unchanged')
    args = parser.parse_args()
    prepare(args.parent, args.controls, args.wall_seconds, args.log_std, args.guarded_progress,args.categorical_probability,args.learning_rate,args.rollout_actions,args.initialization_gate,args.current_cleanup_harness,args.target_kl,args.critic_rate_profile,args.partition_inventory,args.ppo_reward_scale)
