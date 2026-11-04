"""Check the new independent-start learning path,not native performance."""
from copy import deepcopy
import numpy as np
import pytest
import torch


def test_new_cohort_budget_keeps_the_entire_physical_profile_required():
    from navstack.fixed_lowlevel_training import REGISTRATION_COHORT, registered, outcome_registered
    from navstack.mpc_reference_effort import settings
    from test_fixed_lowlevel_training import protocol
    p=protocol();p.update(fixed_lowlevel_training=deepcopy(REGISTRATION_COHORT),
        controlled_budget_per_method=30720,method_budgets={'H':30720},
        reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'),
        reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        mpc_reference_effort=dict(version='reference_centered_mpc_effort_v1'),
        reference_context=dict(version='action_aligned_outcome_context_v1'),
        outcome_context_registration=dict(version='action_aligned_outcome_context_v1',failure_gain_scale=1000.,
            parameter_states=34,new_supervision=0,failure_feedback_ablated=False))
    assert registered(p) and outcome_registered(p) and settings(p)
    for change in (dict(controlled_budget_per_method=10240),dict(method_budgets={'H':30720.}),
                   dict(reference_point_constraints=None),dict(mpc_reference_effort=None)):
        assert not registered({**p,**change})
        if change.get('mpc_reference_effort', 'present') is None:
            assert settings({**p,**change}) is False
        else:
            with pytest.raises(ValueError):settings({**p,**change})


@pytest.mark.parametrize('failure_off',[False,True])
def test_pristine_migration_preserves_outputs_and_first_original_adam_update(failure_off):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from navstack.road_candidate_policy import EndpointClearRoadPolicy,RoadCandidateVecNormalize
    from navstack.reference_context_policy import original_name
    from scripts.run_independent_outcome_cohort import migrate_pristine
    from test_route_catalog_environment import RouteFixture,config
    torch.set_num_threads(1)
    raw=DummyVecEnv([lambda:RouteFixture(config(feedback='continuation'))])
    norm=RoadCandidateVecNormalize(raw,norm_reward=False)
    old=PPO(EndpointClearRoadPolicy,norm,n_steps=2,batch_size=2,n_epochs=1,seed=11,device='cpu')
    try:
        assert not old.policy.optimizer.state
        rng=torch.get_rng_state().clone();new,newnorm,added=migrate_pristine(old,norm,failure_off)
        assert torch.equal(rng,torch.get_rng_state()) and not old.policy.optimizer.state
        assert len(added)==32 and len(new.policy.optimizer.state)==34
        a=torch.as_tensor(norm.normalize_obs(raw.reset()))
        b=torch.as_tensor(newnorm.normalize_obs(np.c_[raw.envs[0]._observation()[None],np.zeros((1,31),dtype=np.float32)]))
        np.testing.assert_array_equal(a.numpy(),b[:,:-31].numpy())
        with torch.no_grad():
            assert torch.equal(old.policy.get_distribution(a).distribution.probs,new.policy.get_distribution(b).distribution.probs)
            assert torch.equal(old.policy.predict_values(a),new.policy.predict_values(b))
        for model,obs in ((old,a),(new,b)):
            model.policy.optimizer.zero_grad()
            values,logs,_=model.policy.evaluate_actions(obs,torch.tensor([2]))
            (values.square().mean()-logs.mean()).backward();model.policy.optimizer.step()
        previous=dict(old.policy.named_parameters())
        for name,value in new.policy.named_parameters():
            mapped=original_name(name)
            if mapped is not None:
                torch.testing.assert_close(value,previous[mapped],rtol=0,atol=0)
    finally:raw.close()
