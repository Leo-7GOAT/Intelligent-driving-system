"""Audit whole successful TRAIN episodes for possible direct-control teacher reuse; never emit labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks
from scripts.audit_native_control_delivery import control_errors


def analyze(manifest=ROOT / 'protocols/native_matched_broader_feedback_pairs_20261101.json'):
    pairs = read(Path(manifest))['pairs']
    if [p['seed'] for p in pairs] != [11, 22, 33, 44, 55]:
        raise ValueError('All original pairs required')
    allcases, total, records = set(), 0, []
    for method in ('H', 'H0'):
        for pair in pairs:
            batch = ROOT / pair[method.lower()]
            audit = read(batch / method / 'endpoint_raw_audit.json')
            if Path(audit['bridge_response_stream']).name != audit['bridge_response_stream']:
                raise ValueError('Bridge stream must be a basename')
            path = batch / 'infrastructure' / audit['bridge_response_stream']
            if sha(path) != audit['bridge_response_sha256'] or sha(batch / 'protocol.json') != audit['protocol_sha256']:
                raise ValueError('Audited bridge/protocol changed')
            responses, error = read_prefix(path)
            if error:
                raise ValueError(error)
            actual, acks, owners, begins = {}, {}, {}, {}
            for wrapper in responses:
                response = wrapper['response']
                episode, request = response.get('episode_id'), response.get('request_id')
                if episode is not None and request is not None:
                    acks[(episode, request)] = response
                if 'scene' in response:
                    begins[episode] = response
                if 'vehicle_control' in response:
                    key = (episode, response['frame'])
                    if key in actual:
                        raise ValueError('Duplicate physical frame')
                    actual[key], owners[key] = response, path.name
            episodes, used, previous, ids, count, files = {}, set(), {}, set(), 0, []
            for trace in sorted((batch / method).glob('transitions_*.jsonl.gz')):
                rows, error = read_prefix(trace)
                if error:
                    raise ValueError(error)
                files.append({'file': trace.name, 'sha256': sha(trace)})
                for row in rows:
                    if 'case_id' in row or 'episode_id' not in row:
                        raise ValueError('Only native TRAIN transitions are accepted; not evaluation decisions')
                    count += _verify_ticks(row['trace'], actual, acks, owners, used, previous, ids, 'ff_state_v1')
                    uuid = row['trace'][0]['backend_evidence']['episode_id']
                    e = episodes.setdefault(uuid, dict(episode=uuid, controls=0, bad_ack=False,
                        termination=None, case=begins[uuid]['scene']['condition']))
                    e['controls'] += len(row['trace'])
                    e['termination'] = row['termination_reason']
                    for tick in row['trace']:
                        response = actual[(uuid, tick['backend_frame'])]
                        applied = response['applied_vehicle_control']
                        e['bad_ack'] |= (bool(control_errors(response['vehicle_control'], applied))
                            or applied['gear'] != 1 or applied['manual_gear_shift'] is not True
                            or applied['hand_brake'] is not False or applied['reverse'] is not False)
            if count != audit['training_control_frames_verified']:
                raise ValueError('Not every audited TRAIN control was accounted for')
            total += count
            chosen = [e for e in episodes.values() if e['termination'] == 'success' and not e['bad_ack']]
            cases = sorted({e['case'] for e in chosen})
            allcases.update(cases)
            records.append(dict(seed=pair['seed'], method=method, batch=batch.name,
                protocol_sha256=sha(batch / 'protocol.json'), raw_audit_sha256=sha(batch / method / 'endpoint_raw_audit.json'),
                bridge_sha256=sha(path), training_files=files, charged_source_TRAIN_controls=count,
                eligible_controls=sum(e['controls'] for e in chosen), eligible_cases=cases, eligible_episodes=chosen,
                excluded_ack_mismatch_episodes=sum(e['bad_ack'] for e in episodes.values())))
    return dict(scope='Whole-successful-TRAIN-episode eligibility;no dataset/fit/new controls;unchanged all-frame engine acknowledgement rule. H-only25/36insufficient;H0TRAINsources added explicitly.',
        sources=records, source_TRAIN_controls=total, covered_TRAIN_cases=sorted(allcases),
        coverage_complete=allcases == set(range(36)), new_controls=0, new_optimizer_updates=0, validation_labels=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'protocols/native_matched_broader_feedback_pairs_20261101.json')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve previous eligibility receipt')
    result = analyze(args.manifest)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('coverage_complete', 'covered_TRAIN_cases', 'source_TRAIN_controls')}))
