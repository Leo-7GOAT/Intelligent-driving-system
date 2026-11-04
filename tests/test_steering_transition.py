import math
from types import SimpleNamespace
import numpy as np
import pytest
from navstack.config import VehicleConfig,PlannerConfig
from navstack.planning import HybridAStar,steering_transition_primitive
from navstack.steering_transition import VERSION,settings,prepare_reference
from navstack.world import World


def test_slew_primitive_cannot_instantly_reverse_measured_steering():
    v=VehicleConfig();speed=1.5
    path,end=steering_transition_primitive((10.,10.,0.),-.25,.5,.8,v,speed,.12)
    assert end==pytest.approx(.07)
    ds=.8/len(path)
    curvature=np.diff(np.r_[0.,path[:,2]])/ds
    steering=np.arctan(v.wheelbase*curvature)
    assert steering[0]<-.24 and abs(steering[0]+.25)<v.steer_rate*ds/speed
    assert np.max(np.abs(np.diff(steering)))*speed/ds<=v.steer_rate+1e-10
    assert path[0,1]<10. and np.isfinite(path).all()


def test_straight_transition_and_full_body_obstacle_rejection():
    v=VehicleConfig();cfg=PlannerConfig(max_seconds=None,max_nodes=200,analytic_expansion=True)
    planner=HybridAStar(v,cfg)
    a=planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25),initial_steering=0.,reference_speed=1.5)
    assert a.status=='success' and np.max(np.abs(a.path[:,1]-10.))<1e-10
    b=planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25,static=[(9.,0.,10.,25.)]),initial_steering=0.,reference_speed=1.5)
    assert b.status!='success'
    with pytest.raises(ValueError):planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25),initial_steering=.8,reference_speed=1.5)


def test_optin_only_and_restores_planner_on_exception():
    valid=dict(reference_steering_transition={'version':VERSION},evaluation_only=True,methods=['H'])
    assert settings({}) is None and settings(valid)
    with pytest.raises(ValueError):settings({**valid,'evaluation_only':False})
    owner=object();env=SimpleNamespace(planner=owner,config=SimpleNamespace(vehicle=VehicleConfig(),planner=PlannerConfig()),
        state=SimpleNamespace(v=1.8,steering=-.2),decode_action=lambda a:((10,10,0),1.5),_rule_plan_cache='stale')
    def fail(action):
        assert env.planner is not owner and env._rule_plan_cache is None
        raise RuntimeError('failure')
    with pytest.raises(RuntimeError):prepare_reference(env,2,fail,settings(valid))
    assert env.planner is owner
