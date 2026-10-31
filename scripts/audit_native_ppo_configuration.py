"""Compare the complete saved learner recipe before native PPO starts."""
import argparse
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.independent_clip_ppo import load_ppo


def profile(checkpoint):
    checkpoint=Path(checkpoint);model=load_ppo(checkpoint/'model.zip',device='cpu')
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError('Require a zero-PPO saved initializer')
    with (checkpoint/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    state=norm.__dict__
    values={name:getattr(model,name) for name in ('n_steps','n_epochs','batch_size','gamma','gae_lambda',
        'ent_coef','vf_coef','max_grad_norm','normalize_advantage','target_kl')}
    values.update(learning_rate_schedule=[float(model.lr_schedule(x)) for x in (1.,.5,0.)],
        clip_schedule=[float(model.clip_range(x)) for x in (1.,.5,0.)],
        clip_vf_schedule=None if model.clip_range_vf is None else [float(model.clip_range_vf(x)) for x in (1.,.5,0.)],
        optimizer_group_rates=[float(g['lr']) for g in model.policy.optimizer.param_groups],
        optimizer_class=type(model.policy.optimizer).__module__+'.'+type(model.policy.optimizer).__name__,
        optimizer_defaults={k:v for k,v in model.policy.optimizer.defaults.items() if k not in ('params',)},
        policy_class=type(model.policy).__module__+'.'+type(model.policy).__name__,
        observation_shape=list(model.observation_space.shape),action_space=type(model.action_space).__name__,
        action_count=int(model.action_space.n),gradient_clip_mode=model.__dict__.get('gradient_clip_mode','global'),
        normalizer={'class':type(norm).__module__+'.'+type(norm).__name__,
            **{k:state.get(k) for k in ('norm_obs','norm_reward','clip_obs','clip_reward','epsilon','fixed_reward_scale')}})
    return values


def differences(actual,expected):
    return {k:{'actual':actual.get(k),'expected':expected.get(k)} for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)}


def verify(checkpoint,reference):
    actual,expected=profile(checkpoint),profile(reference);diff=differences(actual,expected)
    if diff:raise ValueError('Saved PPOconfiguration mismatch:'+json.dumps(diff,sort_keys=True))
    return {'complete_saved_PPO_profile_matches_reference':True,'profile':actual,
        'scope':'Pristine learner recipe only;weights,RNG and statistics may differ across independent training seeds'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--reference',required=True)
    a=p.parse_args();print(json.dumps(verify(a.checkpoint,a.reference)))
