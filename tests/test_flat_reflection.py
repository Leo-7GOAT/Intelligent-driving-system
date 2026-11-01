from copy import deepcopy
from dataclasses import replace
import numpy as np
from navstack.measured_demo import measured_flat_env
from navstack.road_grid import road_digest
from navstack.vehicle import Command,VehicleState
from navstack.world import MovingObstacle
from scripts.diagnose_flat_reflection import reflect_raw
from test_measured_demo import fixture


def test_reflection_matches_rebuilt_geometry_and_is_an_involution():
    live,original,_=fixture()
    scene={'width':live.world.width,'height':live.world.height,'goal':list(live.goal),
        'static':[[12.,21.,14.,23.]],'road_grid':deepcopy(live.world.road.payload),
        'native_layout':{'path':[[6.,16.,0.],[15.,17.,.15],[24.,19.,.25]]}}
    scene['goal']=scene['native_layout']['path'][-1]
    initial=deepcopy(live.initial_scenario);initial.update(goal=scene['goal'],static=scene['static'])
    road=scene['road_grid'];road.pop('sha256')
    for key in ('drivable','collision_drivable'):
        if key in road:
            cells=np.asarray(road[key]);cells[2:4,2:4]=0;road[key]=cells.tolist()
    road['sha256']=road_digest(road);initial['road_sha256']=road['sha256']
    original.close();original=measured_flat_env(live.config,scene,initial)
    state=VehicleState(8.13,14.123,.17,1.2,.1,.2,2.)
    original.plant.state=state;original.plant.previous_command=Command(.3,.06)
    original.last_behavior=np.array([.1,.12,0.,0.]);original.execution_outcome=np.arange(6,dtype=np.float32)/10
    original.world.dynamic=[MovingObstacle(12.,16.,.2,.8,.5)]
    mirrored=deepcopy(scene);height=scene['height']
    mirrored['goal']=[scene['goal'][0],height-scene['goal'][1],-scene['goal'][2]]
    mirrored['static']=[[a,height-d,c,height-b] for a,b,c,d in scene['static']]
    mirrored['native_layout']['path']=[[x,height-y,-yaw] for x,y,yaw in scene['native_layout']['path']]
    payload=mirrored['road_grid'];payload.pop('sha256')
    for key in ('drivable','collision_drivable'):
        if key in payload:payload[key]=np.asarray(payload[key])[::-1,:].tolist()
    payload['sha256']=road_digest(payload)
    mi=deepcopy(initial);mi.update(goal=mirrored['goal'],static=mirrored['static'],road_sha256=payload['sha256'])
    twin=measured_flat_env(live.config,mirrored,mi)
    twin.plant.state=replace(state,y=height-state.y,yaw=-state.yaw,steering=-state.steering)
    twin.plant.previous_command=Command(.3,-.06);twin.last_behavior=np.array([.1,-.12,0.,0.])
    twin.execution_outcome=original.execution_outcome.copy()
    twin.world.dynamic=[MovingObstacle(12.,height-16.,.2,-.8,.5)]
    try:
        x=original._observation()[None,:];expected=twin._observation()[None,:]
        np.testing.assert_allclose(reflect_raw(x),expected,atol=2e-6,rtol=1e-6)
        np.testing.assert_array_equal(reflect_raw(reflect_raw(x)),x)
    finally:live.close();original.close();twin.close()
