import math
from types import SimpleNamespace
import numpy as np
import pytest
from navstack.config import VehicleConfig,PlannerConfig
from navstack.planning import HybridAStar,steering_transition_primitive
from navstack.steering_transition import VERSION,VERSION_BUDGET,settings,prepare_reference
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
    with pytest.raises(ValueError):planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25),initial_steering=math.pi/2,reference_speed=1.5)
    measured=planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25),initial_steering=.50362,reference_speed=1.5)
    assert measured.status in ('success','timeout','no_path')


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


def test_failure_memo_binds_full_steering_speed_and_geometry_across_private_views():
    from navstack.planner_memo import ExactSteeringFailureMemo
    v=VehicleConfig();cfg=PlannerConfig(max_seconds=None,max_nodes=5)
    memo=ExactSteeringFailureMemo(HybridAStar(v,cfg))
    def world():return World(25,25,static=[(12.,0.,13.,25.)])
    start,goal=(6.,10.,0.),(18.,10.,0.)
    first=memo.plan(start,goal,world(),initial_steering=0.,reference_speed=1.5)
    assert first.status=='timeout' and first.expanded_nodes==5
    cached=memo.plan(start,goal,world(),initial_steering=0.,reference_speed=1.5)
    assert cached.cache_hit and cached.expanded_nodes==0 and cached.cached_expanded_nodes==5
    for steer,speed in [(1e-12,1.5),(0.,1.5000000001)]:
        changed=memo.plan(start,goal,world(),initial_steering=steer,reference_speed=speed)
        assert not getattr(changed,'cache_hit',False)
    changed=world();changed.static[0]=(11.9,0.,13.,25.)
    assert not getattr(memo.plan(start,goal,changed,initial_steering=0.,reference_speed=1.5),'cache_hit',False)
    moving=world();moving.dynamic=[object()]
    assert not getattr(memo.plan(start,goal,moving,initial_steering=0.,reference_speed=1.5),'cache_hit',False)
    assert memo.bypasses==1


def test_budget_version_requires_exact_registration():
    p=dict(reference_steering_transition=dict(version=VERSION_BUDGET,max_nodes=32000),evaluation_only=True,methods=['H'])
    assert settings(p)['max_nodes']==32000
    for budget in [32000.,7200,True]:
        with pytest.raises(ValueError):settings({**p,'reference_steering_transition':dict(version=VERSION_BUDGET,max_nodes=budget)})


def test_success_memo_is_optin_exact_and_returns_independent_path_arrays():
    from navstack.planner_memo import ExactSteeringFailureMemo
    v=VehicleConfig();c=PlannerConfig(max_seconds=None,max_nodes=200)
    planner=ExactSteeringFailureMemo(HybridAStar(v,c),cache_success=True)
    def call(steer=0.,speed=1.5,world=None):
        return planner.plan((6.,10.,0.),(10.,10.,0.),world or World(25,25),initial_steering=steer,reference_speed=speed)
    first=call();second=call()
    assert first.status==second.status=='success' and second.cache_hit and second.expanded_nodes==0
    assert second.cached_expanded_nodes==first.expanded_nodes
    np.testing.assert_array_equal(first.path,second.path)
    second.path[0]=999
    np.testing.assert_array_equal(first.path,call().path)
    assert not getattr(call(steer=1e-12),'cache_hit',False)
    assert not getattr(call(speed=1.5000000001),'cache_hit',False)
    moving=World(25,25);moving.dynamic=[object()]
    assert not getattr(call(world=moving),'cache_hit',False)
