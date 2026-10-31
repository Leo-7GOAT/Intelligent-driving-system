"""Install only partition handling before the first physical PPO update."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from navstack.native_cases import registered_native_cases
from navstack.native_partitions import validate_native_partitions

REPLACEMENTS=[
 ('            cases = registered_native_cases(protocol)\n','            cases = registered_native_cases(protocol)\n            evaluation_cases = registered_native_cases(protocol,\'evaluation\') if protocol.get(\'native_validation\') else cases\n'),
 ('evaluation=RecordedRoadEnv(config,client,cases,output) if evaluation_only else CarlaNavigationEnv(config,client,cases)',
  'evaluation=RecordedRoadEnv(config,client,evaluation_cases,output) if evaluation_only else CarlaNavigationEnv(config,client,evaluation_cases)'),
 ('for case_id,case in enumerate(cases):','for case_id,case in enumerate(evaluation_cases):'),
 ('                    write_json(output/"evaluation.json",{"scope":evaluation_scope,',
  '                    if protocol.get(\'native_validation\'):\n                        evaluation_scope=\'Frozen deterministic probes on registered geometry-separated development validation roads; zero updates,not final confirmation\'\n                    write_json(output/"evaluation.json",{"scope":evaluation_scope,\n                        "partition":\'validation\' if protocol.get(\'native_validation\') else \'development\','),
 ('"episodes":len(cases)}),flush=True)','"episodes":len(evaluation_cases)}),flush=True)')]

def worker_with_partitions(original):
    for old,new in REPLACEMENTS:
        if original.count(old)!=1:raise ValueError('Frozen worker structure differs; partition overlay rejected')
        original=original.replace(old,new)
    return original

def install_inventory(protocol,source,inventory_path):
    inventory_path=Path(inventory_path);inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
    candidate=deepcopy(protocol)
    for key in ('routes','native_case_manifest','native_validation'):candidate[key]=deepcopy(inventory[key])
    verification=validate_native_partitions(candidate)
    original=registered_native_cases(protocol);changed=registered_native_cases(candidate)
    if changed[:len(original)]!=original:
        raise ValueError('New training partition must retain every original teacher condition and its index')
    source=Path(source);worker=source/'scripts/native_road_learning.py'
    changed_worker=worker_with_partitions(worker.read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[1]
    overrides=['navstack/native_cases.py','navstack/native_partitions.py','scripts/native_road_learning.py']
    for name in overrides[:2]:
        target=source/name;target.write_bytes((root/name).read_bytes())
    worker.write_text(changed_worker,encoding='utf-8')
    for key in ('routes','native_case_manifest','native_validation'):protocol[key]=candidate[key]
    protocol.pop('case_manifest_sha256',None)
    protocol['native_partition_initialization']={'inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        'verification':verification,'source_overrides':overrides,'original_teacher_conditions_preserved':len(original),
        'scope':'New training distribution before any PPO update; unchanged initialized weights,teacher data,low-level physics,reward or normalization. Original12case gate is not qualification on the new36training or24validation conditions.'}
    return protocol['native_partition_initialization']
