"""Synthetic reference lifecycle tests; these are not CARLA success evidence."""
from dataclasses import replace
import numpy as np
import pytest
from navstack.config import EnvConfig
from navstack.env import rule_action
from navstack.experiments import signature
from navstack.world import MovingObstacle
from test_route_catalog_environment import RouteFixture,config


def prepared():
    settings=EnvConfig.from_dict({**config(feedback='continuation').to_dict(),
                                  'route_goal_candidate':True,'reference_reuse':True})
    env=RouteFixture(settings);env.reset(seed=11)
    mode,detail=env._prepare_reference(26)
    assert mode=='reference'
    reference=env.reference
    x,y,v,yaw=reference.sample([.5])[0]
    env.plant._state=replace(env.state,x=float(x),y=float(y),v=float(v),yaw=float(yaw),time=.5)
    return env,reference


def test_same_selected_goal_preserves_timed_reference_and_explicit_wait_clears_it(monkeypatch):
    env,reference=prepared();target,speed=env.decode_action(26)
    before=[v.copy() for v in (reference.x,reference.y,reference.t,reference.v)]
    assert env.reusable_reference(target,speed)
    def forbidden(*args):raise AssertionError('Valid retained reference unexpectedly searched again')
    monkeypatch.setattr(env.planner,'plan',forbidden)
    assert rule_action(env)==26
    assert env._prepare_reference(26)[1]['status']=='reference_reused'
    assert env.reference is reference and reference.start_time==0.
    for a,b in zip(before,(reference.x,reference.y,reference.t,reference.v)):np.testing.assert_array_equal(a,b)
    assert not env.reusable_reference(target,.75)
    assert not env.reusable_reference((target[0],target[1]+.1,target[2]),speed)
    env._prepare_reference(0)
    assert env.reference is None and not env.reusable_reference(target,speed)


@pytest.mark.parametrize('fault',['lateral','heading','clock_lag','expired','static','dynamic'])
def test_reuse_rechecks_current_tracking_remaining_body_and_current_actor_prediction(fault):
    env,reference=prepared();target,speed=env.decode_action(26)
    if fault=='lateral':env.plant._state=replace(env.state,y=env.state.y+.3)
    elif fault=='heading':env.plant._state=replace(env.state,yaw=.3)
    elif fault=='clock_lag':env.plant._state=replace(env.state,time=3.)
    elif fault=='expired':env.plant._state=replace(env.state,time=float(reference.t[-1])+1.)
    elif fault=='static':env.world.static.append((16.,15.,17.,17.))
    else:env.world.dynamic.append(MovingObstacle(12.,16.,0.,0.))
    assert not env.reusable_reference(target,speed)


def test_reuse_changes_checkpoint_signature_and_requires_explicit_route_semantics():
    baseline=config();new=EnvConfig.from_dict({**baseline.to_dict(),'reference_reuse':True})
    assert signature(baseline)!=signature(new)
    with pytest.raises(ValueError):EnvConfig(reference_reuse=True)
