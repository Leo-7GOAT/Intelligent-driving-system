"""Fit only177new acceleration parameters; preserve every original steering weight."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_flat_reflection import reflect_raw
from scripts.fit_flat_feedback_contrast import exact


def run(output,continue_fit=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the terminal acceleration fit')
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    parent=ROOT/'runs/20261101T100704463570Z_native_reflection_feedback_probe';sys.path.insert(0,str(parent/'source'))
    import navstack
    navstack.__path__.append(str(ROOT/'navstack'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.terminal_acceleration_policy import promote,INDICES
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    require_finished_parent(parent)
    prior=ROOT/'runs/B1_seed22_reflection_feedback_extension_20261101';pr=read(prior/'complete.json')
    if sha(prior/'fit_inputs.npz')!=pr['fit_inputs_sha256']:raise ValueError('Acknowledged teacher inputs changed')
    with np.load(prior/'fit_inputs.npz',allow_pickle=False) as d:
        obs=d['observations'].copy();labels=d['actions'][:,0].copy();groups=d['groups'].copy()
    query=ROOT/'runs/B1_terminal_velocity_queries_20261101';q=read(query/'manifest.json')
    if sha(query/'terminal_velocity_queries.npz')!=q['dataset_sha256']:raise ValueError('Terminal query inputs changed')
    with np.load(query/'terminal_velocity_queries.npz',allow_pickle=False) as d:
        qo=d['observations'].copy();qy=d['actions'].copy();qg=d['groups'].copy()
    query_obs=np.concatenate((qo,reflect_raw(qo)));query_labels=np.r_[qy,qy];query_groups=np.r_[qg+24,qg+28]
    inputs=np.concatenate((obs,query_obs));targets=np.r_[labels,query_labels];allgroups=np.r_[groups,query_groups]
    origin=prior/'checkpoint';metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!='72d465ab4e553c189fd04c5e472a5fcc7cc0eafa371ee7c5c62f6eb9e0ab5844':raise ValueError('Use the declared complementary steering reference')
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    normalized=norm.normalize_obs(inputs);z=torch.as_tensor(normalized[:,INDICES])
    with torch.no_grad():base=torch.cat([model.policy.get_distribution(torch.as_tensor(normalized[i:i+256])).distribution.mean for i in range(0,len(inputs),256)]).clone()
    previous={k:v.clone() for k,v in model.policy.state_dict().items()};old_optimizer=deepcopy(model.policy.optimizer.state_dict())
    continuation=None
    if continue_fit is not None:
        continue_fit=Path(continue_fit).resolve();continuation=read(continue_fit/'rejected.json')
        if (continuation['source_model_sha256']!=metadata['model_sha256'] or continuation['steps']!=512
                or continuation['terminal_query_manifest_sha256']!=sha(query/'manifest.json')):
            raise ValueError('Only one fixed512step continuation with the same anchors/queries is supported')
        metadata=verified_checkpoint_metadata(continue_fit/'checkpoint')
        model=load_ppo(continue_fit/'checkpoint/model.zip',device='cpu');model.policy.set_training_mode(False)
        intervention=continuation['intervention']
    else:intervention=promote(model)
    head=model.policy.action_net.policy_net_terminal
    for p in model.policy.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    y=torch.as_tensor(targets);weights=np.zeros(len(inputs))
    for group in range(32):
        active=allgroups==group
        if not active.any():raise ValueError('Missing teacher or hypothetical TRAINgroup')
        weights[active]=(.5/(24 if group<24 else 8))/active.sum()
    query_begin=len(obs);critical=(query_obs[:,20]<=1)&(query_obs[:,0]>=.75)&(query_labels<-.05)
    if not critical.any():raise ValueError('Require actual terminal overspeed coverage in hypothetical queries')
    output.mkdir();shutil.copy2(__file__,output/'executed_fit.py');shutil.copy2(ROOT/'navstack/terminal_acceleration_policy.py',output/'executed_policy.py')
    registration={'scope':__doc__,'script_sha256':sha(Path(__file__)),'policy_source_sha256':sha(ROOT/'navstack/terminal_acceleration_policy.py'),
        'source_model_sha256':metadata['model_sha256'],'source_normalization_sha256':metadata['normalization_sha256'],
        'terminal_query_manifest_sha256':sha(query/'manifest.json'),'old_teacher_fit_inputs_sha256':pr['fit_inputs_sha256'],
        'physical_teacher_rows':3424,'virtual_reflected_teacher_rows':3424,'hypothetical_terminal_rows':len(query_obs),
        'hypothetical_rows_are_executed':False,'steps':512,'learning_rate':.003,'seed':22,
        'loss':'Acceleration MSE;half total weight on24teacher/reflection groups,half on8terminal-velocity query/reflection groups.',
        'frozen':'Every original actor and critic tensor,log_std,old PPOAdam groups/states/counters,RMS. Only177new NNacceleration parameters fit;no runtime distance/braking rule.',
        'screen':'Terminal query MSE reduced>=75%;all24teacher groups acceleration RMS<=max(.1,2*previousRMS);>=90%correct braking sign on teacher-negative overspeed queries within1m. All steering means exact. Native gate unchanged.',
        'new_controls':0,'new_PPO_updates':0,'intervention':intervention}
    atomic_json(output/'registration.json',registration)
    np.savez_compressed(output/'fit_inputs.npz',observations=inputs,targets=targets,groups=allgroups,weights=weights,
        base_action_means=base.numpy(),critical_terminal_mask=critical)
    optimizer=torch.optim.Adam(head.parameters(),lr=.003);rng=np.random.default_rng(22);started=time.monotonic();history=[]
    if continuation:
        if sha(continue_fit/'auxiliary_Adam.pt')!=continuation['auxiliary_Adam_sha256']:raise ValueError('Auxiliary optimizer changed')
        optimizer.load_state_dict(torch.load(continue_fit/'auxiliary_Adam.pt',map_location='cpu',weights_only=True))
        for _ in range(512):rng.choice(len(z),256,replace=True,p=weights)
        registration.update(inherited_terminal_head_Adam_steps=512,auxiliary_Adam_and_sampler_resumed_exact=True,
            fixed_reference_model_sha256='72d465ab4e553c189fd04c5e472a5fcc7cc0eafa371ee7c5c62f6eb9e0ab5844',
            continuation_parent_result_sha256=sha(continue_fit/'rejected.json'),
            continuation_reason='Terminal MSE and teacher retention pass,but16/18critical braking signs fall short of the unchanged90%gate;one fixed512step extension,no intermediate selection.')
        atomic_json(output/'registration.json',registration)
    for step in range(1,513):
        if time.monotonic()-started>120:raise TimeoutError('Terminal head CPU budget expired')
        indices=rng.choice(len(z),256,replace=True,p=weights)
        pred=base[indices,0]+head(z[indices]).squeeze(1);loss=(pred-y[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%64==0:
            okay,reason=resources_ok(ROOT)
            if not okay:raise RuntimeError(str(reason))
            row={'steps':step,'acceleration_MSE':float(loss.detach())};history.append(row);print(json.dumps(row),flush=True)
    for p in model.policy.parameters():p.requires_grad_(True)
    if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in previous.items()):raise ValueError('Original steering/actor/critic weights changed')
    state=model.policy.optimizer.state_dict()
    if not exact(old_optimizer['state'],state['state']) or old_optimizer['param_groups']!=state['param_groups'][:1]:raise ValueError('Original PPOAdam state changed')
    with torch.no_grad():
        final=base[:,0]+head(z).squeeze(1)
        # Full-policy verification includes all teacher and hypothetical inputs.
        means=torch.cat([model.policy.get_distribution(torch.as_tensor(normalized[i:i+256])).distribution.mean for i in range(0,len(inputs),256)])
    if not torch.equal(means[:,1],base[:,1]):raise ValueError('Steering function changed')
    np.testing.assert_allclose(means[:,0].numpy(),final.numpy(),atol=2e-7,rtol=1e-6)
    metrics=[]
    for group in range(24):
        active=allgroups==group;before=float(torch.sqrt((base[active,0]-y[active]).square().mean()));after=float(torch.sqrt((final[active]-y[active]).square().mean()))
        metrics.append({'group':group,'RMS_before':before,'RMS_after':after,'ceiling':max(.1,2*before)})
    before_mse=float((base[query_begin:,0]-y[query_begin:]).square().mean());after_mse=float((final[query_begin:]-y[query_begin:]).square().mean())
    sign=float((final[query_begin:][critical]<0).float().mean())
    passed=after_mse<=.25*before_mse and all(r['RMS_after']<=r['ceiling'] for r in metrics) and sign>=.9
    checkpoint=output/'checkpoint';shutil.copytree(origin,checkpoint);model.save(checkpoint/'model.zip');torch.save(optimizer.state_dict(),output/'auxiliary_Adam.pt')
    metadata.update(model_sha256=sha(checkpoint/'model.zip'),terminal_acceleration_profile=intervention,
        new_terminal_head_Adam_steps=1024 if continuation else 512,new_terminal_head_parameters=177,
        note='Original steering/critic/actor exactly retained;177new separately learned acceleration residual parameters. Old PPOAdam retained,new PPOparameter group initially empty;fresh rollout required.')
    atomic_json(checkpoint/'metadata.json',metadata);verified_checkpoint_metadata(checkpoint)
    result={**registration,'model_sha256':metadata['model_sha256'],'fit_inputs_sha256':sha(output/'fit_inputs.npz'),
        'auxiliary_Adam_sha256':sha(output/'auxiliary_Adam.pt'),'teacher_retention':metrics,
        'terminal_query_MSE_before':before_mse,'terminal_query_MSE_after':after_mse,
        'critical_queries':int(critical.sum()),'critical_braking_sign_fraction':sign,
        'steering_means_bit_exact':len(inputs),'offline_screen_passed':passed,'history':history,
        'elapsed_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
    atomic_json(output/('complete.json' if passed else 'rejected.json'),result)
    print(json.dumps({k:result[k] for k in ('model_sha256','terminal_query_MSE_before','terminal_query_MSE_after','critical_braking_sign_fraction','offline_screen_passed')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--continue-fit');a=p.parse_args();run(a.output,a.continue_fit)
