from copy import deepcopy
from dataclasses import asdict,replace
import pytest
from navstack.shared_rule_registration import registered,VERSION
from navstack.env import rule_action
from test_route_catalog_environment import RouteFixture,config


def protocol():
    return dict(fixed_shared_rule=dict(version=VERSION,failed_action_penalty_m=5.,goal_wait_requires_heading=True),
        methods=['B3'],evaluation_only=True,controlled_budget_per_method=0,
        method_budgets={'B3':0},training_physics_steps=0,optimizer_updates=0)


def test_rule_admission_is_strictly_evaluation_only_and_zero_learning():
    p=protocol();assert registered(p)
    for key,value in [('evaluation_only',False),('training_physics_steps',1),('optimizer_updates',1),
                      ('controlled_budget_per_method',False),('methods',['H']),('method_budgets',{'B3':False})]:
        q=deepcopy(p);q[key]=value;assert not registered(q)


def test_H_and_B3_use_identical_outcome_observation_constructor():
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.outcome_context_policy import OutcomeContextObservationMixin
    class Observed(OutcomeContextObservationMixin,RouteFixture):pass
    base=config().to_dict();base['reference_reuse']=True
    h=Observed(EnvConfig.from_dict(base));b=Observed(EnvConfig.from_dict({**base,'method':'B3'}))
    try:
        a,_=h.reset(seed=11);z,_=b.reset(seed=11)
        np.testing.assert_array_equal(a,z)
        assert a.shape==z.shape==(4732,)
        b._outcome_failed_actions={2};h._outcome_failed_actions={2}
        np.testing.assert_array_equal(h._observation(),b._observation())
        assert b._observation()[-29]==-1
    finally:h.close();b.close()


def test_failed_action_preference_is_finite_and_does_not_step_or_mask():
    env=RouteFixture(config())
    try:
        env.reset(seed=11,options={'start_x':20.});state=asdict(env.state)
        baseline=rule_action(env);assert baseline!=0
        env._outcome_failed_actions={baseline}
        assert rule_action(env)==baseline  # Historical default remains unchanged.
        assert rule_action(env,failed_action_penalty=5.)!=baseline
        env._outcome_failed_actions=set(range(1,31))
        assert rule_action(env,failed_action_penalty=5.)==baseline
        assert asdict(env.state)==state and env.reference is None
    finally:env.close()


def test_strong_rule_does_not_wait_only_because_position_is_close(monkeypatch):
    import numpy as np
    from navstack import planning
    env=RouteFixture(config())
    try:
        env.reset(seed=11,options={'start_x':23.8});env.plant._state=replace(env.plant.state,yaw=.6)
        monkeypatch.setattr(planning,'analytic_connection',lambda *args:np.array([[23.8,16.,.6],[24.,16.,0.]]))
        assert rule_action(env)==0
        assert rule_action(env,require_goal_heading=True)!=0
    finally:env.close()


@pytest.mark.parametrize('penalty',[-1.,float('nan'),6.,True])
def test_unregistered_rule_preference_is_rejected(penalty):
    with pytest.raises(ValueError):rule_action(None,failed_action_penalty=penalty)
