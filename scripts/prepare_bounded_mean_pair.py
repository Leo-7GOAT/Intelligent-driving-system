"""Register a fixed-endpoint paired response to measured action-mean saturation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def prepare(parent, variant='tanh_pair'):
    import torch
    from stable_baselines3 import PPO
    from navstack.bounded_policy import BoundedMeanActorCriticPolicy, SoftsignMeanActorCriticPolicy
    torch.set_num_threads(1)
    parent=Path(parent).resolve();old=read(parent/'protocol.json');complete=read(parent/'completed.json')
    pointer='bounded_mean_pair' if variant=='tanh_pair' else 'softsign_mean_development'
    names=('unbounded_mean','bounded_mean') if variant=='tanh_pair' else ('softsign_mean',)
    if (ROOT/'runs'/f'{pointer}.json').exists():raise RuntimeError('Inspect registered experiment; no duplicate initialization')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    if variant=='tanh_pair':
        endpoint=next(r for r in complete['results'] if r['branch']=='lower_rate' and r['target']==old['milestones'][-1])
        origin=parent/'lower_rate';validation=read(origin/f"validation_{endpoint['target']}.json")
        checkpoint=Path(validation['checkpoint']);start=endpoint['actual_physics_steps']
        expected_hash=validation['model_sha256']
    else:
        # Reuse completed controls as historical paired comparators. The new
        # branch starts at THEIR original checkpoint, not their trained endpoint.
        assert {b['name'] for b in old['branches']}=={'unbounded_mean','bounded_mean'}
        origin=parent/'unbounded_mean';checkpoint=origin/'initial_checkpoint'
        start=old['inherited_controls']
        expected_hash=old['initial_checkpoint_hashes']['unbounded_mean']['model.zip']
        assert len(complete['results'])==2*len(old['milestones'])
    assert hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest()==expected_hash
    for relative,digest in old['source_hashes'].items():
        assert hashlib.sha256((parent/'source'/relative).read_bytes()).hexdigest()==digest
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+pointer)
    batch.mkdir();shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('navstack/bounded_policy.py','navstack/research.py','scripts/prepare_bounded_mean_pair.py'):
        shutil.copy2(ROOT/relative,batch/'source'/relative)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()}
    initial_hashes={}
    for name in names:
        folder=batch/name;folder.mkdir();destination=folder/'initial_checkpoint'
        shutil.copytree(checkpoint,destination)
        for filename in ('ppo_options.json','resolved_config.json','version.json'):
            shutil.copy2(origin/filename,folder/filename)
        if name in ('bounded_mean','softsign_mean'):
            original=PPO.load(checkpoint/'model.zip',device='cpu')
            policy_class=BoundedMeanActorCriticPolicy if name=='bounded_mean' else SoftsignMeanActorCriticPolicy
            changed=PPO.load(checkpoint/'model.zip',device='cpu',custom_objects={'policy_class':policy_class})
            for key,value in original.policy.state_dict().items():
                assert torch.equal(value,changed.policy.state_dict()[key])
            for key,state in original.policy.optimizer.state_dict()['state'].items():
                for field,value in state.items():
                    other=changed.policy.optimizer.state_dict()['state'][key][field]
                    assert torch.equal(value,other) if torch.is_tensor(value) else value==other
            changed.save(destination/'model.zip')
        initial_hashes[name]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}
        assert all(initial_hashes[name][key]==value for key,value in hashes.items() if key!='model.zip')
        summary=read(destination/'training_summary.json')
        write_json(folder/'latest.json',dict(summary,checkpoint=str(destination)))
    now=time.time()
    protocol={**old,'created_epoch':now,'deadline_epoch':now+3600,'inherited_controls':start,
        'parent_batch':parent.name,'parent_checkpoint':checkpoint.name,
        'scope':'One-seed development policy-parameterization transfer; correlated branches with all inherited costs, not fresh training or architecture superiority',
        'branches':[{'name':name,'learning_rate':.0001} for name in names],
        'milestones':[start+n for n in (0,100000,300000,600000)],
        'hypothesis':'Keeping the Gaussian mean inside actuator bounds restores deceleration exploration and improves static-task learning without losing parking',
        'primary_diagnostic':'Both fixed static cases at every endpoint, with complete ten-case outcomes and all eight parking probes; no best-checkpoint selection',
        'intervention':('Only bounded branch replaces Gaussian mean mu with tanh(mu).' if variant=='tanh_pair' else 'Only new branch replaces Gaussian mean mu with softsign(mu); completed identity/tanh branches are retained in the parent batch, not repeated.')+' Identical parameter tensors, optimizer, RNG, environment state and normalization; unchanged Gaussian likelihood, PPO objective, std, action clipping, reward, cases and learning rate.',
        'inherited_total_cost_note':'The 5,098,576-control lineage includes selected historical checkpoint branching; all discarded original-v19 and learning-rate sibling costs remain in parent evidence and are not zero.',
        'checkpoint_hashes':hashes,'initial_checkpoint_hashes':initial_hashes,
        'resource_limits':{'wall_seconds':3600,'disk_min_GiB':12,'ram_min_GiB':4,'cpu_threads':1,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'source_hashes':{p.relative_to(batch/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch/'source').rglob('*.py')}}
    if variant=='softsign':
        protocol['completed_comparator_batch']=parent.name
        protocol['completed_comparator_protocol_sha256']=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest()
        protocol['hypothesis']='Polynomial-tail bounded mean retains usable gradients at measured inherited raw means near five; test static completion at every fixed endpoint, retaining all negatives'
    write_json(batch/'protocol.json',protocol)
    write_json(batch/'initial_pairing_audit.json',{'weights_equal':True,'optimizer_equal':True,'all_other_checkpoint_files_equal':True,
        'warning':'Function changes immediately although tensors match. The zero-update endpoint measures this transfer discontinuity.'})
    write_json(ROOT/'runs'/f'{pointer}.json',{'batch':str(batch)})
    print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--variant',choices=['tanh_pair','softsign'],default='tanh_pair')
    a=p.parse_args();prepare(a.parent,a.variant)
