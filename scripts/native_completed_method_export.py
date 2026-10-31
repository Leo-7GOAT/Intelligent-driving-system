"""Explicit interrupted-batch export contract;never select among completed arms."""
from pathlib import Path
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata

VERSION='completed_method_with_incomplete_siblings_v1'


def completed_method_profile(batch,protocol,method,check_processes=True):
    batch=Path(batch)
    registered=protocol['methods']
    if (method not in registered or len(registered)<2 or len(set(registered))!=len(registered)
            or any(not isinstance(m,str) or Path(m).name!=m or m in ('.','..') for m in registered)):
        raise ValueError('Require one explicit completed method with registered incomplete siblings')
    if check_processes:
        from scripts.prepare_native_shared_brake import require_closed_method_parent
        require_closed_method_parent(batch)
    guard=read(batch/'guard_final.json')
    if type(guard.get('exit_code')) is not int or guard['exit_code']==0:
        raise ValueError('Require a retained terminal batch failure,not a successful subset')
    if read(batch/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
        raise ValueError('Verified owned-server cleanup is mandatory')
    if (batch/'summary.json').exists() and read(batch/'summary.json').get('status')=='complete':
        raise ValueError('A complete batch must export every registered method')
    evidence=['guard_final.json','interruption_audit.json']
    failures=[name for name in ('failure.json','launcher_failure.json') if (batch/name).is_file()]
    if not failures:raise ValueError('Retain the original structured failure evidence')
    evidence+=failures
    interrupted=read(batch/'interruption_audit.json')['methods']
    if set(interrupted)!=set(registered) or interrupted[method]['state']!='complete':
        raise ValueError('The original interruption audit must identify every registered method')
    incomplete=[m for m in registered if m!=method]
    for sibling in incomplete:
        if interrupted[sibling]['state'] not in ('interrupted','not_started','missing','unstarted'):
            raise ValueError('Cannot omit another completed registered method')
        if (batch/sibling/'evaluation.json').exists() or (batch/sibling/'endpoint_raw_audit.json').exists():
            raise ValueError('Another evaluated method cannot be silently excluded')
    folder=batch/method;summary=read(folder/'summary.json')
    origin=folder/read(folder/'latest_checkpoint.json')['folder']
    metadata=verified_checkpoint_metadata(origin)
    if (summary.get('stopped_reason') is not None or summary.get('reload_exact') is not True
            or summary['physics_steps']<=0 or summary['optimizer_steps']<=0
            or any(metadata[k]!=summary[k] for k in ('physics_steps','optimizer_steps','lineage_control_steps'))
            or sha(folder/'model.zip')!=metadata['model_sha256']):
        raise ValueError('Require the complete trained endpoint,not an earlier selected checkpoint')
    inventory=protocol.get('native_validation',{}).get('manifest',protocol.get('native_case_manifest'))
    count=len(inventory['cases']) if inventory is not None else len(protocol['routes'])
    episodes=read(folder/'evaluation.json')['episodes']
    if len(episodes)!=count or sorted(r['case_id'] for r in episodes)!=list(range(count)):
        raise ValueError('Every condition including failures must be exported')
    return {'version':VERSION,'registered_methods':registered,'completed_methods':[method],
        'incomplete_methods':incomplete,'batch_complete':False,'batch_summary_present':(batch/'summary.json').exists(),
        'failure_evidence_sha256':{name:sha(batch/name) for name in evidence},
        'original_interruption_states':{m:interrupted[m] for m in registered},
        'incomplete_file_sha256':{m:{p.relative_to(batch).as_posix():sha(p)
            for p in sorted((batch/m).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
            for m in incomplete},
        'scope':'Only the completed method is scored/reproduced. Original protocol,all sibling method files/checkpoints/raw prefixes and failure evidence are retained;the batch is not declared complete.'}


def validate_exported_inventory(batch,protocol,manifest):
    """Run before loading any model;normal complete-archive behavior stays strict."""
    profile=manifest.get('completed_method_export')
    if profile is None:
        if protocol['methods']!=manifest['methods']:
            raise ValueError('Registered methods differ from exported inventory')
        return False
    if (not isinstance(profile,dict) or manifest.get('schema_version')!=2 or profile.get('version')!=VERSION
            or manifest.get('methods')!=profile.get('completed_methods')
            or len(manifest['methods'])!=1):
        raise ValueError('Invalid explicit completed-method export profile')
    expected=completed_method_profile(batch,protocol,manifest['methods'][0],check_processes=False)
    if profile!=expected:raise ValueError('Interrupted-batch export evidence changed')
    inventory=manifest.get('retained_incomplete_method_files')
    if not isinstance(inventory,dict) or set(inventory)!=set(profile['incomplete_methods']):
        raise ValueError('Every incomplete sibling inventory must be explicit')
    for method,names in inventory.items():
        actual=sorted(name for name in manifest['files'] if name.startswith(manifest['batch']+'/'+method+'/'))
        if sorted(names)!=actual or len(set(names))!=len(names):
            raise ValueError('Incomplete sibling data/checkpoint inventory changed')
    return True
