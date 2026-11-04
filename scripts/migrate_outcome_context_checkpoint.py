"""Version a checkpoint for learned outcome feedback without changing old logits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.independent_clip_ppo import load_ppo
from navstack.outcome_context_policy import migrate_learner,VERSION,FAILURE_GAIN_SCALE
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.collect_native_recovery import tensor_digest
from scripts.evaluate_native_rule import require_finished_parent


def migrate(batch,output,ablate_failure_feedback=False):
    batch,output=Path(batch).resolve(),Path(output).resolve();require_finished_parent(batch)
    if output.exists():raise FileExistsError('Preserve each schema migration')
    source=batch/'H'/read(batch/'H/latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(source)
    if sha(source/'model.zip')!=sha(batch/'H/model.zip'):raise ValueError('Require evaluated initializer bytes')
    before=load_ppo(source/'model.zip',device='cpu');rng=torch.get_rng_state().clone();after=migrate_learner(before,outcome_ablated=ablate_failure_feedback)
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('Migration advanced global Torch RNG')
    old,new=dict(before.policy.named_parameters()),dict(after.policy.named_parameters())
    if len(old)!=32 or len(new)!=34:raise ValueError('Unexpected parameter inventory')
    for name,value in old.items():
        if (not torch.equal(value,new[name]) or tensor_digest(before.policy.optimizer.state[value])
                !=tensor_digest(after.policy.optimizer.state[new[name]])):
            raise ValueError('Original tensor/Adam changed:'+name)
    extras=sorted(set(new)-set(old))
    if any(torch.count_nonzero(new[n]) or float(after.policy.optimizer.state[new[n]]['step'])!=0 for n in extras):
        raise ValueError('New learning coordinates must start at zero')
    shutil.copytree(source,output);after.save(output/'model.zip')
    config=read(output/'config.json');config['observation_encoding']={'version':VERSION};config['failure_gain_scale']=FAILURE_GAIN_SCALE
    config['failure_feedback_ablated']=ablate_failure_feedback
    atomic_json(output/'config.json',config)
    meta.update(model_sha256=sha(output/'model.zip'),observation_encoding=VERSION,failure_gain_scale=FAILURE_GAIN_SCALE,
        policy_parameter_states=34,checkpoint_schema_intervention=True)
    meta['failure_feedback_ablated']=ablate_failure_feedback
    atomic_json(output/'metadata.json',meta)
    reloaded=load_ppo(output/'model.zip',device='cpu')
    if any(not torch.equal(v,reloaded.policy.state_dict()[n]) for n,v in after.policy.state_dict().items()):
        raise ValueError('Migrated model failed reload')
    receipt=dict(version=VERSION,parent_batch=batch.name,old_model_sha256=sha(source/'model.zip'),new_model_sha256=sha(output/'model.zip'),
        normalization_sha256=sha(output/'normalization.pkl'),source_checkpoint_metadata_sha256=sha(source/'metadata.json'),
        original32_parameter_and_Adam_states_exact=True,new_parameter_names=extras,new_parameters_and_Adam_steps_zero=True,
        copied_resume_RNG_sha256=sha(output/'resume_state.pkl'),global_Torch_RNG_unchanged=True,
        failure_gain_scale=FAILURE_GAIN_SCALE,failure_feedback_ablated=ablate_failure_feedback,script_sha256=sha(Path(__file__)),
        policy_source_sha256=sha(ROOT/'navstack/outcome_context_policy.py'),new_controls=0,new_updates=0,
        limits='Schema/optimizer initialization only;requires native-state equivalence proof and actual new learning before any benefit claim.')
    atomic_json(output/'outcome_migration.json',receipt);print({k:receipt[k] for k in ['old_model_sha256','new_model_sha256','new_parameter_names']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ablate-failure-feedback',action='store_true')
    a=p.parse_args();migrate(a.batch,a.output,a.ablate_failure_feedback)
