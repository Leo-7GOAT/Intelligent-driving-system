import math
import pytest
from navstack.native_hazards import resolve_native_hazard,verify_native_hazard_scene


def layout(yaw=0.):
    return {'path':[[6.+i*math.cos(yaw),10.+i*math.sin(yaw),yaw] for i in range(25)],'width':40.,'height':40.}


def test_native_hazard_uses_route_arclength_heading_and_crossing_normal():
    spec={'version':'native_hazard_v1','kind':'crossing_walker','distance_m':10.,'lateral_m':-3.5,'speed_m_s':.8,'start_delay_s':2.5}
    hazard=resolve_native_hazard(layout(math.pi/2),spec)
    assert hazard['spawn_pose']==pytest.approx([9.5,20.,math.pi/2])
    assert hazard['walker_direction_local']==pytest.approx([-1.,0.])
    assert hazard['task_family']=='dynamic_interaction' and hazard['start_delay_s']==2.5
    parked=resolve_native_hazard(layout(),{'version':'native_hazard_v1','kind':'static_vehicle','distance_m':10.})
    assert parked['spawn_pose']==[16.,10.,0.] and parked['walker_speed_m_s']==0.


@pytest.mark.parametrize('change',[{'distance_m':2.},{'distance_m':23.},{'distance_m':float('nan')},
    {'lateral_m':0.},{'speed_m_s':2.},{'start_delay_s':-1.},{'extra':0},{'version':'unknown'}])
def test_invalid_native_hazards_reject_before_rpc_or_actor_creation(change):
    spec={'version':'native_hazard_v1','kind':'crossing_walker',**change}
    with pytest.raises(ValueError):resolve_native_hazard(layout(),spec)


def test_hazard_scene_rejects_phantom_actor_and_wrong_measured_position():
    spec={'version':'native_hazard_v1','kind':'static_vehicle'}
    observed={**resolve_native_hazard(layout(),spec),'actor_id':42,'actor_type':'vehicle.tesla.model3'}
    verify_native_hazard_scene(layout(),spec,observed,[[14.,9.,18.,11.]],[])
    with pytest.raises(ValueError):verify_native_hazard_scene(layout(),spec,observed,[],[])
    with pytest.raises(ValueError):verify_native_hazard_scene(layout(),spec,observed,[[24.,9.,28.,11.]],[])
    with pytest.raises(ValueError):verify_native_hazard_scene(layout(),spec,{**observed,'actor_id':0},[[14.,9.,18.,11.]],[])


def test_native_dynamic_frame_evidence_cannot_drop_or_invent_actor_measurements():
    from copy import deepcopy
    from scripts.audit_native_method_endpoint import _verify_ticks
    measured={'episode_id':'e','request_id':2,'frame':31,'imu_frame':31,'state':{'time':.05,'x':1.},
              'command':{'acceleration':0.,'steering':0.},
              'native_dynamic_actors':[{'actor_id':42,'actor_type':'walker.pedestrian.0001','state':{'x':2.,'y':3.,'vx':0.,'vy':.8,'radius':.5}}]}
    evidence={k:deepcopy(measured[k]) for k in ('episode_id','request_id','frame','imu_frame','native_dynamic_actors')}
    tick={'state':measured['state'],'command':measured['command'],'backend_frame':31,'backend_evidence':evidence}
    def check(row,raw):return _verify_ticks([row],{('e',31):raw},{},{},set(),{},set())
    assert check(tick,measured)==1
    missing=deepcopy(tick);del missing['backend_evidence']['native_dynamic_actors']
    with pytest.raises(ValueError,match='missing or fabricated'):check(missing,measured)
    altered=deepcopy(tick);altered['backend_evidence']['native_dynamic_actors'][0]['state']['x']+=1
    with pytest.raises(ValueError,match='differs'):check(altered,measured)
    raw=deepcopy(measured);del raw['native_dynamic_actors']
    with pytest.raises(ValueError,match='missing or fabricated'):check(tick,raw)
