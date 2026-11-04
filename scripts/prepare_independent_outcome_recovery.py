"""Retain an interrupted independent cohort learner; never repeat completed jobs."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from navstack.fixed_lowlevel_training import registered
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_evaluation_recovery import verified_interruption
from scripts.prepare_native_recovery import verified_checkpoint_metadata,claim_recovery
from scripts.fixed_shared_native_rule import without_functions
from scripts.run_independent_outcome_cohort import resource_admission


def prepare(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    reason=verified_interruption(parent)
    p=read(parent/'protocol.json');audit=read(parent/'interruption_audit.json')
    if output.exists() or (parent/'recovery_registration.json').exists():raise FileExistsError('Preserve recovery attempts')
    if p.get('independent_outcome_cohort',{}).get('version')!='independent_outcome_TRAIN48_v1':raise ValueError('Wrong cohort')
    if p['controlled_budget_per_method']!=30720 or p['methods']!=['H']:raise ValueError('Require original fixed course')
    folder=parent/'H'/read(parent/'H/latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(folder)
    if not 0<meta['physics_steps']==meta['lineage_control_steps']<30720:raise ValueError('Invalid retained checkpoint')
    resources=resource_admission(output.parent)
    for name,digest in p['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen parent source changed:'+name)
    gate='navstack/fixed_lowlevel_training.py'
    if without_functions(ROOT/gate,{'registered'})!=without_functions(parent/'source'/gate,{'registered'}):
        raise ValueError('Admission overlay changes unrelated code')
    output.mkdir();(output/'infrastructure').mkdir()
    for name in ('source','calibration'):shutil.copytree(parent/name,output/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(folder,output/'resume/H');shutil.copy2(ROOT/gate,output/'source'/gate)
    prior=p['recovery']['H'];status=audit['methods']['H'];remaining=30720-meta['physics_steps']
    p['recovery']={'H':dict(checkpoint='resume/H',parent_batch=parent.name,parent_checkpoint=folder.name,
        lineage_controls=meta['lineage_control_steps'],
        consumed_controls_lower_bound=max(meta['charged_control_steps_lower_bound'],prior['consumed_controls_lower_bound']+status['consumed_controls_lower_bound']),
        retained_optimizer_steps=prior['retained_optimizer_steps']+meta['optimizer_steps'],
        unknown_unflushed_parent_tail=status['unknown_unflushed_tail'],
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])}
    p['fixed_cohort_recovery']=dict(version='saved_outcome_cohort_recovery_v1',original_target=30720,
        saved_controls=meta['physics_steps'],parent_protocol_sha256=sha(parent/'protocol.json'),
        interruption_audit_sha256=sha(parent/'interruption_audit.json'))
    p['recovery_disclosure']=dict(reason=reason,readable_parent_controls=status['consumed_controls_lower_bound'],
        retained_parent_controls=meta['physics_steps'],discarded_readable_tail=status['consumed_controls_lower_bound']-meta['physics_steps'],
        uncertainty='Unflushed tail remains unknown;all parent costs retained as lower bounds. Not exact uninterrupted or exact matched-interaction evidence.')
    now=time.time();p.update(controlled_budget_per_method=remaining,method_budgets={'H':remaining},
        checkpoint_thresholds=list(range(320,remaining+1,320))+[remaining],parent_batch=parent.name,
        created_epoch=now,deadline_epoch=now+14400,resource_admission=resources,
        scope='Interrupted independent TRAIN48 learner recovery;retained Adam/RNG/RMS,remaining original lineage budget,all costs disclosed')
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    if not registered(p):raise ValueError('Incomplete recovery registration')
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    atomic_json(output/'parent_interruption_audit.json',audit)
    claim_recovery(parent,output.name)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(prepare(args.parent,args.output),flush=True)
