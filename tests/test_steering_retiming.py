from dataclasses import replace
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.trajectory import parameterize
from navstack.steering_retiming import retime


def reference():
    s=np.linspace(0,12,601);yaw=.3*np.sin(s*.8)
    x=np.r_[0,np.cumsum(np.cos((yaw[:-1]+yaw[1:])/2)*np.diff(s))]
    y=np.r_[0,np.cumsum(np.sin((yaw[:-1]+yaw[1:])/2)*np.diff(s))]
    vehicle=VehicleConfig(wheelbase=2.,max_steer=.6,steer_rate=.15)
    return vehicle,parameterize(np.c_[x,y,yaw],2.5,0.,vehicle,stop=True)


def test_retiming_preserves_geometry_and_enforces_continuous_interpolated_curvature_rate():
    vehicle,original=reference();before=original.kappa.copy()
    result=retime(original,vehicle,0.)
    for name in ('x','y','yaw','s'):np.testing.assert_array_equal(getattr(result,name),getattr(original,name))
    np.testing.assert_array_equal(original.kappa,before)
    assert result.v[0]==0 and result.v[-1]==0 and result.start_time==original.start_time
    assert result.t[-1]>original.t[-1]
    for i,dt in enumerate(np.diff(result.t)):
        k=np.linspace(result.kappa[i],result.kappa[i+1],101)
        rates=vehicle.wheelbase*abs(result.kappa[i+1]-result.kappa[i])/dt/(1+(vehicle.wheelbase*k)**2)
        assert rates.max()<=vehicle.steer_rate+1e-10
    assert result.a.max()<=vehicle.max_accel+1e-9
    assert result.a.min()>=-vehicle.max_decel-1e-9


def test_retiming_rejects_incompatible_actual_speed_instead_of_lowering_measurement():
    vehicle,original=reference();speed=original.v.copy();speed[0]=2.5
    with pytest.raises(ValueError,match='initial_speed_incompatible'):
        retime(replace(original,v=speed),vehicle,0.)
    assert speed[0]==2.5


def test_retiming_straight_reference_keeps_speed_profile():
    vehicle=VehicleConfig();p=np.c_[np.linspace(0,10,51),np.zeros((51,2))]
    original=parameterize(p,1.5,0.,vehicle,stop=True)
    result=retime(original,vehicle,0.)
    np.testing.assert_allclose(result.v,original.v)
    np.testing.assert_allclose(result.t,original.t)


def test_actual_wheel_overshoot_and_small_signed_velocity_are_preserved():
    vehicle=VehicleConfig();p=np.c_[np.linspace(0,10,51),np.zeros((51,2))]
    original=parameterize(p,1.5,-4.9e-5,vehicle,stop=True)
    result=retime(original,vehicle,-.5008)
    assert result.v[0]==-4.9e-5
    assert np.arctan(vehicle.wheelbase*result.kappa[0])==pytest.approx(-.5008)
    assert np.all(np.diff(result.t)>0) and np.all(result.v[1:]>=0)
    assert result.a.max()<=vehicle.max_accel+1e-9
    assert result.a.min()>=-vehicle.max_decel-1e-9


def test_joint_search_fallback_keeps_policy_speed_and_restores_temporary_retimer(monkeypatch):
    from types import SimpleNamespace
    from navstack.config import PlannerConfig
    from navstack.world import World
    from navstack.planning import HybridAStar,PlanResult
    from navstack.steering_transition import settings,prepare_reference
    from navstack.steering_retiming import VERSION
    from navstack.reference_compute import VERSION as COMPUTE
    protocol=dict(methods=['H'],evaluation_only=True,
        reference_steering_transition=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000),
        reference_road_refinement=dict(version='sampled_native_reference_refinement_v1',max_attempts=3),
        reference_compute_profile=dict(version=COMPUTE),
        reference_joint_speed=dict(version=VERSION,spatial_speed_ceiling=.25))
    speeds=[];vehicle=VehicleConfig();world=World(30,30)
    path=np.c_[np.linspace(6,12,51),np.full(51,10.),np.zeros(51)]
    def plan(self,start,goal,world,**kw):
        speeds.append(kw['reference_speed'])
        return PlanResult('success' if kw['reference_speed']==.25 else 'no_path',
            path.copy() if kw['reference_speed']==.25 else np.empty((0,3)),6. if kw['reference_speed']==.25 else None,1,.1,
            None if kw['reference_speed']==.25 else 'open_set_exhausted')
    monkeypatch.setattr(HybridAStar,'plan',plan)
    owner=HybridAStar(vehicle,PlannerConfig(max_seconds=None));prior=object()
    env=SimpleNamespace(planner=owner,config=SimpleNamespace(vehicle=vehicle,planner=owner.config),
        state=SimpleNamespace(v=0.,steering=0.),decode_action=lambda action:((12.,10.,0.),1.5),
        _reference_retimer=prior,_rule_plan_cache=None)
    def prepare(action):
        result=env.planner.plan((6.,10.,0.),(12.,10.,0.),world)
        env.reference=env._reference_retimer(parameterize(result.path,env.decode_action(action)[1],0.,vehicle))
        return 'reference',dict(status=result.status,expanded_nodes=result.expanded_nodes)
    mode,detail=prepare_reference(env,26,prepare,settings(protocol))
    assert mode=='reference' and speeds==[1.5,.25]
    assert env.reference.v.max()==1.5 and env.decode_action(26)[1]==1.5
    assert detail['joint_speed_search']['total_expanded_nodes']==2
    assert env.planner is owner and env._reference_retimer is prior
    with pytest.raises(ValueError):settings({**protocol,'evaluation_only':False})
