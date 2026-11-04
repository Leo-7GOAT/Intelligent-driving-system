from types import SimpleNamespace
import numpy as np
import pytest
import torch
from navstack.outcome_context_policy import LearnedOutcomeBias,outcome_context


def test_measured_failure_is_action_aligned_and_wait_or_success_is_not_penalized():
    env=SimpleNamespace(_reference_selection=None,_outcome_last_action=2,execution_outcome=np.array([0,1,0,0,1,0]))
    context=outcome_context(env)
    assert context[2]==-1 and np.count_nonzero(context)==1 and context[0]==0
    env.execution_outcome[1]=0;assert not outcome_context(env).any()
    env.execution_outcome[1]=1;env._outcome_last_action=0;assert not outcome_context(env).any()


def test_failure_history_accumulates_without_masking_and_clears_on_accepted_reference():
    env=SimpleNamespace(_reference_selection=None,_outcome_last_action=30,_outcome_failed_actions={2},execution_outcome=np.array([0,1,0,0,1,0]))
    assert set(np.flatnonzero(outcome_context(env)==-1))=={2,30}
    assert env._outcome_failed_actions=={2}  # observation is read-only
    env.execution_outcome=np.array([1,0,0,0,0,0])
    assert not outcome_context(env).any()


def test_failure_gain_is_zero_initially_and_learns_a_sign_without_an_action_override():
    head=LearnedOutcomeBias(100.,1000.);logits=torch.tensor([[0.]+[.1]*30]);context=torch.zeros_like(logits);context[0,2]=-1
    torch.testing.assert_close(head(logits,context),logits,rtol=0,atol=0)
    loss=torch.log_softmax(head(logits,context),dim=1)[0,2];loss.backward()
    assert head.failure_gain.grad.item()<0
    with torch.no_grad():head.failure_gain.fill_(.01)
    assert head(logits,context)[0,2]<logits[0,2]
    with torch.no_grad():head.failure_gain.fill_(-.01)
    assert head(logits,context)[0,2]>logits[0,2]
    with pytest.raises(ValueError):head(logits,context*.5)


@pytest.mark.parametrize('ablated',[False,True])
@pytest.mark.parametrize('failure_off',[False,True])
def test_migration_preserves_original_model_adam_rng_and_roundtrip_learning(tmp_path,ablated,failure_off):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from navstack.config import EnvConfig
    from navstack.reference_context_policy import ReferenceContextObservationMixin,ReferenceContextVecNormalize,ReferenceContextPolicy,AblatedReferenceContextPolicy
    from navstack.outcome_context_policy import migrate_learner,OutcomeContextObservationMixin,VERSION
    from test_route_catalog_environment import RouteFixture,config
    torch.set_num_threads(1)
    class OriginalFixture(ReferenceContextObservationMixin,RouteFixture):pass
    class OutcomeFixture(OutcomeContextObservationMixin,RouteFixture):pass
    cfg=EnvConfig.from_dict({**config(feedback='continuation').to_dict(),'route_goal_candidate':True,'reference_reuse':True})
    raw=DummyVecEnv([lambda:OriginalFixture(cfg)]);norm=ReferenceContextVecNormalize(raw,norm_reward=False)
    old=PPO(AblatedReferenceContextPolicy if ablated else ReferenceContextPolicy,norm,n_steps=2,batch_size=2,n_epochs=1,device='cpu',seed=11)
    old.learn(2);norm.training=False
    with torch.no_grad():
        old.policy.action_net.reference_bias.gain.fill_(.03);old.policy.value_net.context_weight.fill_(.02)
    before_rng=torch.get_rng_state().clone();new=migrate_learner(old,outcome_ablated=failure_off)
    assert torch.equal(before_rng,torch.get_rng_state())
    a=torch.as_tensor(norm.normalize_obs(raw.reset())).repeat(3,1);a[:,-31:]=0;a[0,-5]=1
    z=a.clone();z[1,-29]=-1;z[1,-28]=-1
    with torch.no_grad():
        assert torch.equal(old.policy.get_distribution(a).distribution.probs,new.policy.get_distribution(z).distribution.probs)
        assert torch.equal(old.policy.predict_values(a),new.policy.predict_values(z))
    old_parameters=dict(old.policy.named_parameters());new_parameters=dict(new.policy.named_parameters())
    assert len(old_parameters)==32 and len(new_parameters)==34
    for name,parameter in new_parameters.items():
        if name in old_parameters:
            assert torch.equal(parameter,old_parameters[name])
            for key,value in old.policy.optimizer.state[old_parameters[name]].items():
                actual=new.policy.optimizer.state[parameter][key]
                assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
        else:
            assert not torch.count_nonzero(parameter) and new.policy.optimizer.state[parameter]['step']==0
    new.save(tmp_path/'model.zip');norm.save(tmp_path/'norm.pkl')
    wrapped=DummyVecEnv([lambda:OutcomeFixture(cfg)]);restored=ReferenceContextVecNormalize.load(tmp_path/'norm.pkl',wrapped);restored.training=False
    loaded=PPO.load(tmp_path/'model.zip',env=restored,device='cpu')
    assert loaded.policy.reference_context_version==VERSION and loaded.policy.failure_gain_scale==1000.
    assert loaded.policy.outcome_ablated is failure_off
    steps={name:float(loaded.policy.optimizer.state[param]['step']) for name,param in loaded.policy.named_parameters()}
    loaded.learn(2,reset_num_timesteps=False)
    assert all(float(loaded.policy.optimizer.state[param]['step'])==steps[name]+1 for name,param in loaded.policy.named_parameters())
    restored.close();norm.close()


def test_failure_ablation_preserves_reuse_feedback_but_removes_failure_gradient():
    head=LearnedOutcomeBias(100.,1000.,ablated=True)
    with torch.no_grad():head.gain.fill_(.01);head.failure_gain.fill_(.02)
    logits=torch.zeros(1,31);context=torch.zeros_like(logits);context[0,2]=-1;context[0,26]=1
    result=head(logits,context);result.sum().backward()
    assert result[0,2]==0 and result[0,26]==1
    assert head.failure_gain.grad==0 and head.gain.grad!=0
