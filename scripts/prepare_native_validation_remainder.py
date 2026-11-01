"""Resume only unfinished validation cases after the already diagnosed walker-spawn collision."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.prepare_native_evaluation_recovery import verified_interruption, evaluation_lineage
from scripts.prepare_native_recovery import verified_checkpoint_metadata, claim_recovery
from scripts.prepare_native_control_delivery_probe import replace_once


def prepare(parent):
    parent = Path(parent).resolve()
    if verified_interruption(parent) != 'native_crossing_walker_spawn_collision':
        raise ValueError('Require this specific diagnosed and cleaned-up initialization failure')
    if (parent / 'recovery_registration.json').exists():
        raise FileExistsError('Preserve the registered child; never rerun the parent')
    p = deepcopy(read(parent / 'protocol.json'))
    rows, error = read_prefix(parent / 'B1/evaluation_trace.jsonl.gz')
    if error:
        raise ValueError('Require a fully closed readable prefix; never repair original trace bytes')
    last = {r['case_id']: r for r in rows}
    completed = sorted(k for k, r in last.items() if r['info'].get('episode_metrics', {}).get('termination_reason') is not None)
    remaining = [i for i in range(len(p['native_case_manifest']['cases'])) if i not in completed]
    if completed != list(range(17)) or remaining != list(range(17, 24)):
        raise ValueError('Require the actual recorded17completed and7unfinished validation conditions')
    consumed = sum(len(r['info']['trace']) for r in rows)
    audit = read(parent / 'interruption_audit.json')
    if audit['methods']['B1']['evaluation_controls_readable_lower_bound'] != consumed:
        raise ValueError('Partial evaluation costs differ')
    for name, digest in p['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Parent source changed')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_validation_remainder')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent / 'calibration', batch / 'calibration')
    shutil.copytree(origin, batch / 'resume/B1')
    if (parent / 'measured_demonstrations.npz').exists():
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    selected = [deepcopy(p['native_case_manifest']['cases'][i]) for i in remaining]
    used_routes = sorted({r['route_index'] for r in selected})
    p['routes'] = [p['routes'][i] for i in used_routes]
    for case in selected:
        case['route_index'] = used_routes.index(case['route_index'])
    p['native_case_manifest']['cases'] = selected
    # Reuse the already validated clearance fix: exact XY/yaw, +.2m Z only
    # before the existing30gravity-settlement ticks; no executed-state edit.
    name = 'scripts/carla_bridge37.py'
    text = (batch / 'source' / name).read_text(encoding='utf-8')
    text = replace_once(text, "                try:\n                    hazard=self.world.spawn_actor(bp,self.transform(x,y,yaw))", "                walker_transform=self.transform(x,y,yaw)\n                walker_transform.location.z+=.2\n                try:\n                    hazard=self.world.spawn_actor(bp,walker_transform)")
    marker = "                self.walker_direction=self.native_hazard['walker_direction_local']"
    text = replace_once(text, marker, marker + "\n                self.native_hazard={**self.native_hazard,'walker_spawn_initialization':{\n                    'version':'native_walker_clearance_020_v1','requested_world_z':walker_transform.location.z,\n                    'clearance_added_m':.2,'settle_ticks':30}}")
    (batch / 'source' / name).write_text(text, encoding='utf-8')
    worker = 'scripts/native_road_learning.py'
    text = (batch / 'source' / worker).read_text(encoding='utf-8')
    text = replace_once(text, 'evaluation.reset(seed=7301+case_id,', "evaluation.reset(seed=7301+protocol['evaluation_original_case_ids'][case_id],")
    (batch / 'source' / worker).write_text(text, encoding='utf-8')
    p['evaluation_original_case_ids'] = remaining
    p['validation_remainder'] = {'parent_batch': parent.name, 'parent_protocol_sha256': sha(parent / 'protocol.json'),
        'retained_completed_original_case_ids': completed, 'original_case_ids': remaining,
        'original_route_indices': used_routes, 'parent_controls_charged': consumed,
        'initialization_version': 'native_walker_clearance_020_v1', 'model_sha256': metadata['model_sha256'],
        'limits': 'Only unfinished cases; completed failures remain retained. Vertical spawn initialization changes before unchanged settlement;do not claim a uniform-harness formal comparison. No new learning or final conditions.'}
    p['recovery'] = {'B1': evaluation_lineage(p['recovery']['B1'], read(parent / 'B1/summary.json'), parent.name, origin.name, 'B1', consumed)}
    now = time.time()
    p.update(scope='Seven remaining existing validation conditions after a preserved native walker-spawn failure. Explicit +.2m pre-control spawn clearance;original XY/yaw,settlement and evaluation seeds retained. Zero learning.',
             created_epoch=now, deadline_epoch=now + 3600,
             resource_limits={**p['resource_limits'], 'wall_seconds': 3600, 'automatic_retries': 0})
    for relative in (name, worker):
        p['source_hashes'][relative] = sha(batch / 'source' / relative)
    registered_native_cases(p)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
    claim_recovery(parent, batch.name)
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    prepare(parser.parse_args().parent)
