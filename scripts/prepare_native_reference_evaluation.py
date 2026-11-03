"""Freeze one learned reference-context endpoint on complete known inventories."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.reference_context_policy import VERSION
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(parent,output,phase):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    require_finished_parent(parent)
    if output.exists():raise FileExistsError('Preserve each frozen endpoint evaluation')
    old=read(parent/'protocol.json')
    if old.get('reference_context')!={'version':VERSION} or old['methods']!=['H','H0']:
        raise ValueError('Require the registered learned reference-context pair')
    if phase not in ('static','known_validation'):raise ValueError('Unknown development partition')
    p=deepcopy(old);p.pop('freeze_observation_statistics',None)
    p.pop('lineage_targets',None);p.pop('native_validation',None)
    if phase=='static':
        reference=read(ROOT/'runs/20261102T101413887816Z_native_diverse_TRAIN36_teacher/protocol.json')
        p['routes']=deepcopy(reference['routes']);p['native_case_manifest']=deepcopy(reference['native_case_manifest'])
        p['native_case_manifest']['cases']=deepcopy(reference['native_case_manifest']['cases'][12:24])
    else:
        reference=read(ROOT/'runs/20261103T043242Z_native_recovery/protocol.json')
        p['native_validation']=deepcopy(reference['native_validation'])
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<16*1024**3:raise RuntimeError('Evaluation staging drive headroom insufficient')
    output.mkdir();(output/'infrastructure').mkdir()
    for name in ('source','calibration'):
        shutil.copytree(parent/name,output/name,ignore=shutil.ignore_patterns('__pycache__'))
    # Add only the read-only auditor;retain executed policy/observation/physical
    # source bytes from training without a deployment-time implementation swap.
    name='scripts/audit_native_reference_inference.py'
    shutil.copy2(ROOT/name,output/'source'/name);p['source_hashes'][name]=sha(output/'source'/name)
    for method in ('H','H0'):
        folder=parent/method;summary=read(folder/'summary.json')
        budget=old['method_budgets'][method]
        if (summary['stopped_reason'] is not None or not budget<=summary['physics_steps']<budget+320
                or summary['optimizer_steps']<=0):
            raise ValueError('Require the full registered training budget endpoint')
        source=folder/read(folder/'latest_checkpoint.json')['folder'];metadata=verified_checkpoint_metadata(source)
        if metadata['physics_steps']!=summary['physics_steps']:raise ValueError('Checkpoint not the final fixed budget endpoint')
        shutil.copytree(source,output/'resume'/method)
        prior=old['recovery'][method]
        p['recovery'][method]=dict(checkpoint='resume/'+method,parent_batch=parent.name,
            parent_checkpoint=method+'/'+source.name,lineage_controls=summary['lineage_control_steps'],
            consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'],
            retained_optimizer_steps=prior['retained_optimizer_steps']+summary['optimizer_steps'],
            unknown_unflushed_parent_tail=prior.get('unknown_unflushed_parent_tail',False),
            model_sha256=metadata['model_sha256'],normalization_sha256=metadata['normalization_sha256'])
    now=time.time();wall=7200 if phase=='static' else 3600
    p.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0,'H0':0},
        deterministic_training_route_evaluation=True,record_initial_parameters=False,checkpoint_thresholds=[],
        created_epoch=now,deadline_epoch=now+wall,
        resource_limits={**p['resource_limits'],'wall_seconds':wall,'automatic_retries':0},
        scope='Frozen learned reference-context pair;complete '+phase+' development inventory only. No new learning,no final conditions.',
        reference_context_endpoint=dict(phase=phase,original_teacher_static_case_ids=list(range(12,24)) if phase=='static' else None,
            new_PPO_updates=0,new_final_conditions=0,source_model_selection='Final predeclared training budget only'))
    atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    print(str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['static','known_validation'],required=True)
    a=p.parse_args();prepare(a.parent,a.output,a.phase)
