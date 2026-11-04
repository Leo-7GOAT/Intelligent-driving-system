"""Build direct-policy views from complete audited CARLA teachers; no new driving.

Numeric high-level action IDs are deliberately not relabelled between catalogs.
These data are extra supervision, not PPO updates or independent validation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import signature, write_json
from navstack.measured_demo import measured_flat_env, measured_label, advance_measured_flat
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.audit_native_rule import audit
from scripts.evaluate_native_rule import require_finished_parent
from navstack.native_cases import registered_native_cases


def build(reference_config, teachers, output, wall_seconds=600):
    if type(wall_seconds) is not int or not 60<=wall_seconds<=1800:
        raise ValueError('Measured view construction requires a declared60..1800second wall budget')
    started = time.monotonic()
    code_identity = {'view_source_sha256': sha(ROOT / 'navstack/measured_demo.py'),
                     'script_sha256': sha(Path(__file__))}
    reference_config, output = Path(reference_config).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve existing measured-view datasets')
    teachers = [Path(path).resolve() for path in teachers]
    if not teachers or len(teachers) != len(set(teachers)):
        raise ValueError('Require unique completed teacher batches')
    cfg = EnvConfig.from_dict(read(reference_config)['env'])
    if signature(cfg) != read(reference_config)['signature']:
        raise ValueError('Reference configuration signature differs')
    provenance, calibrations = [], set()
    for teacher in teachers:
        require_finished_parent(teacher)
        verified = audit(teacher)
        if verified != read(teacher / 'raw_audit.json') or verified['cases_verified'] != len(registered_native_cases(read(teacher/'protocol.json'))):
            raise ValueError('Require every registered teacher case and its complete raw audit')
        protocol = read(teacher / 'protocol.json')
        for name, digest in protocol['source_hashes'].items():
            if sha(teacher / 'source' / name) != digest:
                raise ValueError('Teacher frozen source changed')
        donor = EnvConfig.from_dict(protocol['config'])
        if (donor.vehicle != cfg.vehicle or donor.control_dt != cfg.control_dt
                or donor.reward != cfg.reward or donor.actuator_observation != cfg.actuator_observation):
            raise ValueError('Teacher physical limits/reward/actuator observation differ from the target view')
        calibrations.add(protocol['calibration_sha256'])
        provenance.append({'batch': teacher.name, 'raw_audit_sha256': sha(teacher / 'raw_audit.json'),
                           'protocol_sha256': sha(teacher / 'protocol.json'), 'trace_sha256': verified['trace_sha256'],
                           'controls': verified['evaluation_controls_verified'],
                           'bridge_projection_source_sha256': protocol['source_hashes']['scripts/carla_bridge37.py'],
                           'target_config_differences': {k: [donor.to_dict().get(k), cfg.to_dict().get(k)]
                               for k in set(donor.to_dict()) | set(cfg.to_dict()) if donor.to_dict().get(k) != cfg.to_dict().get(k)}})
    if len(calibrations) != 1 or len({p['trace_sha256'] for p in provenance}) != len(provenance):
        raise ValueError('Calibrations differ or the same raw teacher trace was counted twice')
    output.mkdir(parents=True)
    write_json(output / 'registration.json', {'scope': 'Read-only measured direct views; no policy fit, control or update',
               'reference_config_sha256': sha(reference_config), 'sources': provenance, 'wall_seconds': wall_seconds})
    deadline = started + wall_seconds
    observations, labels, returns, masks, groups, cases = [], [], [], [], [], []
    def health():
        okay, reason = resources_ok(ROOT)
        if not okay or time.monotonic() >= deadline: raise RuntimeError('View reconstruction guard: ' + str(reason))
    try:
        for source_id, (teacher, source) in enumerate(zip(teachers, provenance)):
            health()
            rows, error = read_prefix(teacher / 'evaluation_trace.jsonl.gz')
            if error: raise ValueError(error)
            begins = {}
            paths = list((teacher / 'infrastructure').glob('*_responses.jsonl.gz'))
            if len(paths) != 1: raise ValueError('Require the single complete audited rule bridge stream')
            for path in paths:
                if sha(path) != read(teacher / 'raw_audit.json')['bridge_response_sha256']:
                    raise ValueError('Audited bridge stream changed')
                responses, error = read_prefix(path)
                if error: raise ValueError(error)
                begins.update({w['response']['episode_id']: w['response'] for w in responses if 'scene' in w.get('response', {})})
            count = 0
            for case_id in range(len(registered_native_cases(read(teacher/'protocol.json')))):
                decisions = [row for row in rows if row['case_id'] == case_id]
                ticks = [tick for row in decisions for tick in row['info']['trace']]
                initial = decisions[0]['info']['initial_scenario']
                begin = begins[ticks[0]['backend_evidence']['episode_id']]
                env = measured_flat_env(cfg, begin['scene'], initial, begin=begin)
                changed_frames, brake_projections = 0, 0
                group = len(cases)
                discounted = np.empty(len(ticks), dtype=np.float32); total = 0.
                for index in reversed(range(len(ticks))):
                    total = ticks[index]['reward'] + cfg.gamma_low * total
                    discounted[index] = total
                try:
                    for index, tick in enumerate(ticks):
                        if index % 128 == 0: health()
                        observations.append(env._observation())
                        label = measured_label(env, tick); labels.append(label)
                        request = tick['requested_command']
                        brake_projections += (abs(request['steering']) > cfg.vehicle.max_steer+1e-3
                            or not -cfg.vehicle.max_decel-1e-3 <= request['acceleration'] <= cfg.vehicle.max_accel+1e-3)
                        before = [asdict(actor) for actor in env.world.dynamic]
                        advance_measured_flat(env, tick, label)
                        changed_frames += before != [asdict(actor) for actor in env.world.dynamic]
                        masks.append(index % 5 != 0); groups.append(group)
                finally: env.close()
                returns.extend(discounted); count += len(ticks)
                metric = read(teacher / ('case_%04d.json' % case_id))
                cases.append({'group': group, 'source_id': source_id, 'case_id': case_id, 'controls': len(ticks),
                              'termination_reason': metric['termination_reason'], 'dynamic_actors': len(initial['dynamic']),
                              'measured_actor_changed_frames': changed_frames,
                              'acknowledged_brake_projection_controls': brake_projections})
                print(json.dumps(cases[-1]), flush=True)
            if count != source['controls']: raise ValueError('Not all teacher controls entered the dataset')
        dataset = output / 'measured_demonstrations.npz'
        np.savez_compressed(dataset, observations=np.asarray(observations, dtype=np.float32),
                            actions=np.asarray(labels, dtype=np.float32), returns=np.asarray(returns, dtype=np.float32),
                            train_mask=np.asarray(masks), groups=np.asarray(groups, dtype=np.int64))
        if code_identity != {'view_source_sha256': sha(ROOT / 'navstack/measured_demo.py'),
                             'script_sha256': sha(Path(__file__))}:
            raise ValueError('Conversion source changed during dataset construction')
        result = {'scope': 'Measured direct B1 views in one explicit target configuration. Extra supervision only; '
                           'all cases/failures included. No high-level action relabelling, fitting or new driving.',
                  'sources': provenance, 'cases': cases, 'teacher_unique_controls': len(labels),
                  'dataset_sha256': sha(dataset), 'reference_config_sha256': sha(reference_config),
                  **code_identity, 'conversion_wall_seconds': time.monotonic()-started,
                  'conversion_wall_budget_seconds': wall_seconds,
                  'brake_projection': 'Only known held-measured-steer brake requests may use their exact acknowledged '
                                      'rate/absolute-limited command. Raw requests and frame evidence remain unchanged; QP tolerance is not enlarged.',
                  'holdout': 'Every fifth control within each case; correlated diagnostics only',
                  'returns': 'Recorded teacher discounted returns, not a new reward profile or direct-policy counterfactual',
                  'new_controls': 0, 'supervised_optimizer_steps': 0, 'ppo_updates': 0}
        write_json(output / 'manifest.json', result)
        return result
    except BaseException as exc:
        write_json(output / 'failure.json', {'error': repr(exc), 'completed_cases': cases})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--teachers', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wall-seconds',type=int,default=600,help='Declared before conversion;60..1800seconds, resource floors unchanged')
    args = parser.parse_args()
    result = build(args.reference_config, args.teachers, args.output,args.wall_seconds)
    print(json.dumps({'teacher_unique_controls': result['teacher_unique_controls'], 'new_controls': 0, 'ppo_updates': 0}))
