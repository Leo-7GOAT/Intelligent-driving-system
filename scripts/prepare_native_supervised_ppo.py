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


def guarded_initialization(protocol,metadata_config):
    """Explicit new reward version before ANY PPO update; never rewrite parent."""
    original=EnvConfig.from_dict(metadata_config['env'])
    if metadata_config['signature']!=signature(original):
        raise ValueError('Supervised initialization signature is inconsistent')
    apply_guarded_progress(protocol,original.to_dict())
    changed=training_config(protocol,'B1',original.vehicle)
    before,after=original.to_dict(),changed.to_dict()
    expected=dict(before['reward']);expected.update(progress_weight=.5,collision=-90.,out_of_bounds=-90.)
    if {k:v for k,v in before.items() if k!='reward'}!={k:v for k,v in after.items() if k!='reward'} or after['reward']!=expected:
        raise ValueError('Reward initialization changes unrelated task/control fields')
    return {**metadata_config,'env':after,'signature':signature(changed),
            'scope':'Same supervised actor and untouched PPO optimizer, explicitly initialized for guarded_progress_v1 before training'}


def prepare(parent, controls=20480, wall_seconds=7200, log_std=-2.3, guarded_progress=False):
    parent = Path(parent).resolve()
    if type(controls) is not int or not 640 <= controls <= 100000 or not 900 <= wall_seconds <= 7200:
        raise ValueError('Invalid bounded native learning budget')
    require_finished_parent(parent)
    protocol = read(parent/'protocol.json')
    if (protocol.get('methods') != ['B1'] or not protocol.get('evaluation_only')
            or not protocol.get('extra_supervision') or audit(parent, 'B1') != read(parent/'B1/endpoint_raw_audit.json')):
        raise ValueError('Require complete raw-audited supervised-only direct diagnostic')
    evaluated = read(parent/'B1/evaluation.json')
    if len(evaluated['episodes']) != 4 or sum(r['success'] for r in evaluated['episodes']) < 3:
        raise ValueError('Require the predeclared three-of-four known-road representation gate')
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    with ProcessLease(parent/'supervised_ppo_prepare.lock'):
        registration = parent/('supervised_ppo_guarded_registration.json' if guarded_progress else 'supervised_ppo_registration.json')
        if registration.exists():
            raise FileExistsError('Do not repeat the registered supervised-PPO continuation')
        for name, digest in protocol['source_hashes'].items():
            if sha(parent/'source'/name) != digest:
                raise ValueError('Parent frozen source changed')
        origin = parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder']
        metadata = verified_checkpoint_metadata(origin)
        if metadata['physics_steps'] or metadata['optimizer_steps']:
            raise ValueError('The parent is not a zero-PPO supervised initialization')
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_supervised_ppo')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration', batch/'calibration')
        destination = batch/'resume/B1'
        shutil.copytree(origin, destination)
        reward_change=None
        if guarded_progress:
            original_config=read(origin/'config.json')
            changed_config=guarded_initialization(protocol,original_config)
            write_json(destination/'config.json',changed_config)
            reward_change={'parent_config_sha256':sha(origin/'config.json'),
                'initialized_config_sha256':sha(destination/'config.json'),
                'previous_reward':original_config['env']['reward'],'new_reward':changed_config['env']['reward'],
                'scope':'New reward version before any PPO update; actor/critic architecture and all non-reward environment fields unchanged'}
        torch.set_num_threads(1)
        model = load_ppo(destination/'model.zip', device='cpu')
        change = initial_exploration(model, log_std)
        original_model_sha256 = metadata['model_sha256']
        model.save(destination/'model.zip')
        metadata['model_sha256'] = sha(destination/'model.zip')
        write_json(destination/'metadata.json', metadata)
        if sha(destination/'normalization.pkl') != metadata['normalization_sha256']:
            raise ValueError('Supervised observation statistics changed')
        verified_checkpoint_metadata(destination)
        for name in ('prepare_native_supervised_ppo.py','prepare_native_hazard_learning.py'):
            shutil.copy2(ROOT/'scripts'/name, batch/'source/scripts'/name)
        now = time.time()
        prior = protocol['recovery']['B1']
        protocol.pop('evaluation_only')
        protocol.update(created_epoch=now, deadline_epoch=now+wall_seconds, parent_batch=parent.name,
            controlled_budget_per_method=controls, method_budgets={'B1': controls},
            freeze_observation_statistics=True, checkpoint_thresholds=sorted({min(controls,n) for n in (1280,2560,controls)}),
            scope='Real native PPO after disclosed teacher supervision; frozen train-only statistics, calibrated Gaussian exploration; not ordinary from-scratch PPO or a hierarchy comparison',
            supervised_ppo_intervention={**change, 'parent_model_sha256': original_model_sha256,
                'initialized_model_sha256': metadata['model_sha256'], 'supervised_updates_added': 0,
                'statistics': 'Frozen at teacher training examples; correlated holdout excluded from fitting',
                'comparison_limit': 'Teacher access,20Hz continuous control and exploration differ from old categorical2Hz B1. All must be shared or controlled in formal comparisons.'},
            recovery={'B1': {**prior, 'checkpoint': 'resume/B1', 'parent_batch': parent.name,
                'parent_checkpoint': origin.name, 'lineage_controls': 0,
                'consumed_controls_lower_bound': protocol['extra_supervision']['teacher_controls'],
                'retained_optimizer_steps': 0, 'unknown_unflushed_parent_tail': False}})
        if reward_change is not None:
            protocol['supervised_ppo_intervention']['reward_initialization']=reward_change
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
    args = parser.parse_args()
    prepare(args.parent, args.controls, args.wall_seconds, args.log_std, args.guarded_progress)
