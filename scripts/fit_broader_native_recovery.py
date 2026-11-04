"""One canonical22 actor fit on nominal plus audited executed TRAIN recovery labels."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def recovery_masks(arrays, expected):
    import numpy as np
    mask, selected, groups = (np.asarray(arrays[k]) for k in ('fit_train_mask', 'teacher_executed', 'groups'))
    if (mask.dtype != np.bool_ or selected.dtype != np.bool_ or mask.shape != selected.shape
            or mask.shape != groups.shape or (mask & ~selected).any() or int(mask.sum()) != expected
            or set(groups[mask]) != set(range(2048, 2084))
            or any(np.count_nonzero(mask & (groups == g)) < 5 for g in range(2048, 2084))):
        raise ValueError('Require executed eligible teacher coverage in all36episodes, including failed outcomes')
    holdout = np.zeros(len(mask), dtype=bool)
    for g in range(2048, 2084):
        holdout[np.flatnonzero(mask & (groups == g))[::5]] = True
    return mask & ~holdout, holdout


def run(collection, views, reference_config, output, profile='original'):
    collection, views, reference_config, output = map(lambda p: Path(p).resolve(), (collection, views, reference_config, output))
    if output.exists():
        raise FileExistsError('Never overwrite or repeat a recovery fit')
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0, str(collection / 'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.native_cases import registered_native_cases
    from navstack.research import atomic_json, resources_ok
    from scripts.audit_native_method_endpoint import sha
    from scripts.collect_native_recovery import validate_collection
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    from navstack.config import EnvConfig
    require_finished_parent(collection)
    profiles = {'original':('broader_executed_recovery_actor_v1','',.02,1024,1024),
        'steering_v2':('broader_executed_recovery_actor_steering_v2','_steering',.01,1024,1024),
        'convergence_v3':('broader_executed_recovery_actor_convergence_v3','_convergence',.02,7168,8192)}
    if profile not in profiles: raise ValueError('Unknown fixed fit profile')
    expected_version, suffix, steering_scale, steps, retained_steps = profiles[profile]
    recipe_file = 'protocols/native_B1_broader_executed_recovery'+suffix+'_20261102.json'
    recipe_path = ROOT / recipe_file
    recipe = read(recipe_path)
    if (recipe['version'] != expected_version or recipe['seed'] != 22
            or recipe['steps'] != steps or recipe['learning_rate'] != 1e-4
            or recipe.get('steering_scale',.02) != steering_scale
            or recipe['collection_batch'] != collection.name
            or recipe['collection_protocol_sha256'] != sha(collection / 'protocol.json')
            or recipe['fit_script_sha256'] != sha(__file__)
            or recipe['helper_script_sha256'] != sha(ROOT / 'scripts/fit_broader_measured_actor.py')):
        raise ValueError('Predeclared fixed recovery recipe/source changed')
    p = read(collection / 'protocol.json')
    audit = read(collection / 'B1/collection_raw_audit.json')
    result = read(collection / 'B1/collection.json')
    strict = read(collection / 'B1/strict_delivery.json')
    origin = collection / 'resume/B1'
    metadata = verified_checkpoint_metadata(origin)
    if profile=='steering_v2':
        prior = ROOT / 'runs' / recipe['prior_failed_basic_batch']
        failed = read(prior / 'B1/evaluation.json')
        if (sha(prior / 'B1/endpoint_raw_audit.json') != recipe['prior_failed_basic_audit_sha256']
                or sum(e['success'] for e in failed['episodes']) != 3
                or sum(e['termination_reason']=='out_of_bounds' for e in failed['episodes']) != 1
                or metadata['model_sha256'] != recipe['source_model_sha256']):
            raise ValueError('Require the actual failed original objective and same fixed starting model')
    config = read(origin / 'config.json')
    validate_collection(p, EnvConfig.from_dict(config['env']), registered_native_cases(p))
    if (p['native_recovery_collection']['route_profile'] != 'broader_TRAIN36_v1'
            or audit['protocol_sha256'] != sha(collection / 'protocol.json')
            or audit['collection_result_sha256'] != sha(collection / 'B1/collection.json')
            or audit['dataset_sha256'] != sha(collection / 'B1/collection.npz')
            or not audit['all_measured_observations_bit_exact'] or not audit['all_learner_proposals_bit_exact']
            or not audit['all_source_choices_reconstructed'] or not audit['not_policy_evaluation']
            or not strict['strict_delivery_gate_passed']
            or strict['counts']['controls'] != audit['collection_controls']
            or metadata['model_sha256'] != p['native_recovery_collection']['source_model_sha256']):
        raise ValueError('Require completed independent measured-history and engine-delivery audits')
    helper_path = ROOT / 'scripts/fit_broader_measured_actor.py'
    module = importlib.util.spec_from_file_location('broader_measured_actor_helper', helper_path)
    helper = importlib.util.module_from_spec(module); module.loader.exec_module(helper)
    manifest = read(views / 'manifest.json')
    teacher = ROOT / 'runs' / manifest['sources'][0]['batch']
    teacher_protocol = read(teacher / 'protocol.json')
    if (teacher_protocol.get('data_role') != 'TRAIN_EXTRA_SUPERVISION'
            or teacher_protocol.get('native_validation') is not None
            or teacher_protocol['routes'] != p['routes']
            or teacher_protocol['native_case_manifest'] != p['native_case_manifest']
            or manifest['sources'][0]['protocol_sha256'] != sha(teacher / 'protocol.json')
            or manifest['sources'][0]['raw_audit_sha256'] != sha(teacher / 'raw_audit.json')
            or manifest['reference_config_sha256'] != sha(reference_config)
            or read(reference_config)['env'] != config['env']
            or read(reference_config)['signature'] != config['signature']):
        raise ValueError('Nominal data must have the same original TRAIN conditions and direct observation interface')
    families = [c.get('native_hazard', {}).get('kind', 'curve') for c in registered_native_cases(p)]
    x, y, groups, train, holdout, _ = helper.validated_arrays(views / 'measured_demonstrations.npz', manifest, families)
    with np.load(collection / 'B1/collection.npz', allow_pickle=False) as arrays:
        recovery_train, recovery_holdout = recovery_masks(arrays, audit['eligible_executed_teacher_labels'])
        recovery_groups = arrays['groups']
        x = np.concatenate((x, arrays['observations'])); y = np.concatenate((y, arrays['actions']))
        groups = np.r_[groups, recovery_groups]
        train = np.r_[train, recovery_train]; holdout = np.r_[holdout, recovery_holdout]
    torch.set_num_threads(1)
    resume_adam = resume_sampler = None
    if profile=='convergence_v3':
        prior_fit = ROOT / recipe['resume_fit']
        prior_registration, prior_result = read(prior_fit/'registration.json'), read(prior_fit/'complete.json')
        for name, digest in recipe['resume_hashes'].items():
            if sha(prior_fit/name) != digest: raise ValueError('Actual Adam/sampler/checkpoint resume input changed')
        if (prior_registration['steering_scale'] if 'steering_scale' in prior_registration else .02) != .02:
            raise ValueError('Convergence test resumes the original objective only')
        if (prior_result['actual_Adam_step_each'] != [1024]
                or prior_registration['collection_audit_sha256'] != sha(collection/'B1/collection_raw_audit.json')
                or prior_registration['nominal_manifest_sha256'] != sha(views/'manifest.json')
                or prior_registration['recovery_dataset_sha256'] != audit['dataset_sha256']):
            raise ValueError('Resume must use identical data and completed1024-step actor state')
        origin = prior_fit/'checkpoint'
        metadata = verified_checkpoint_metadata(origin)
        if metadata['model_sha256'] != recipe['source_model_sha256']:
            raise ValueError('Fixed convergence parent differs')
        resume_adam = torch.load(prior_fit/'auxiliary_Adam.pt',map_location='cpu',weights_only=True)
        resume_sampler = read(prior_fit/'sampler_state.json')
    model = load_ppo(origin / 'model.zip', device='cpu')
    with (origin / 'normalization.pkl').open('rb') as stream: norm = pickle.load(stream)
    rms = pickle.dumps(norm.obs_rms)
    output.mkdir()
    shutil.copy2(__file__, output / 'executed_fit.py'); shutil.copy2(helper_path, output / 'executed_helper.py')
    atomic_json(output / 'registration.json', dict(version=expected_version,
        recipe_sha256=sha(recipe_path), recipe_file=recipe_file, steering_scale=steering_scale,
        seed=22, steps=steps, retained_recovery_actor_steps=retained_steps, learning_rate=1e-4, collection=collection.name,
        collection_audit_sha256=sha(collection / 'B1/collection_raw_audit.json'),
        nominal_manifest_sha256=sha(views / 'manifest.json'), recovery_dataset_sha256=audit['dataset_sha256'],
        source_model_sha256=metadata['model_sha256'], source_normalization_sha256=metadata['normalization_sha256'],
        inherited_measured_actor_steps=1024+(retained_steps-steps), new_PPO_updates=0, new_controls=0,
        sampling='Equal total probability per successful nominal episode and each executed recovery episode. All36recoveryoutcomes included. Every fifth eligible recovery label held out;correlated diagnostic only.',
        selection='One fixed final canonical22endpoint;no model selection or automatic promotion.',
        source_hashes={n: sha(output / n) for n in ('executed_fit.py', 'executed_helper.py')}))
    started = time.monotonic()
    def health(row):
        okay, reason = resources_ok(ROOT)
        if not okay or time.monotonic() - started > 600:
            raise RuntimeError('Bounded actor fit stopped: ' + str(reason))
        atomic_json(output / 'heartbeat.json', row)
    fitted, optimizer, sampler = helper.fit_actor(model, norm.normalize_obs(x), y, groups, train, holdout,
        seed=22, steps=steps, callback=health, steering_scale=steering_scale,
        resume_optimizer=resume_adam, resume_sampler=resume_sampler)
    if pickle.dumps(norm.obs_rms) != rms: raise ValueError('Frozen RMS changed')
    destination = output / 'checkpoint'; shutil.copytree(origin, destination); model.save(destination / 'model.zip')
    metadata.update(model_sha256=sha(destination / 'model.zip'), inherited_measured_actor_steps=1024+(retained_steps-steps),
        additional_recovery_actor_steps=steps, retained_recovery_actor_steps=retained_steps, cumulative_measured_actor_steps=1024+retained_steps,
        recovery_fit_registration_sha256=sha(output / 'registration.json'))
    atomic_json(destination / 'metadata.json', metadata); verified_checkpoint_metadata(destination)
    torch.save(optimizer.state_dict(), output / 'auxiliary_Adam.pt'); atomic_json(output / 'sampler_state.json', sampler)
    fitted.update(model_sha256=metadata['model_sha256'], new_controls=0, new_PPO_updates=0,
        new_auxiliary_Adam_steps=steps, retained_recovery_actor_steps=retained_steps, cumulative_measured_actor_steps=1024+retained_steps,
        elapsed_seconds=time.monotonic() - started, research_acceptance='UNMET')
    atomic_json(output / 'complete.json', fitted)
    print(json.dumps({k: fitted[k] for k in ('model_sha256', 'new_auxiliary_Adam_steps', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True); parser.add_argument('--views', required=True)
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', choices=['original','steering_v2','convergence_v3'], default='original')
    args = parser.parse_args(); run(args.collection, args.views, args.reference_config, args.output, args.profile)
