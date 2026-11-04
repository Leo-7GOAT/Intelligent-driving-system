"""Independent bicycle integration checks all returned geometric families."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from navstack.config import EnvConfig,PlannerConfig,VehicleConfig
from navstack.dubins_guide import dubins_distance,dubins_families
from navstack.experiments import signature
from navstack.planning import HybridAStar,primitive
from navstack.vehicle import wrap
from navstack.world import World
from navstack.road_grid import RoadGrid
from navstack.trajectory import parameterize


def endpoint(start,word,lengths,radius):
    pose=np.asarray(start,float)
    for kind,length in zip(word,lengths):
        if length>1e-12:
            steer=0 if kind=='S' else math.atan((1 if kind=='L' else -1)/radius)
            pose=primitive(pose,steer,length,1.,.1)[-1]
    return pose


def test_every_dubins_family_reaches_random_endpoints_by_exact_bicycle_arcs():
    rng=np.random.default_rng(21);families=set()
    for _ in range(80):
        start=rng.uniform(-5,5,3);goal=rng.uniform(-5,5,3);radius=float(rng.uniform(.5,4))
        choices=dubins_families(start,goal,radius)
        assert choices
        for word,lengths in choices:
            families.add(word);assert all(math.isfinite(v) and v>=0 for v in lengths)
            end=endpoint(start,word,lengths,radius)
            np.testing.assert_allclose(end[:2],goal[:2],atol=1e-8)
            assert abs(wrap(end[2]-goal[2]))<1e-8
        assert dubins_distance(start,goal,radius)>=math.dist(start[:2],goal[:2])-1e-8
    assert families=={'LSL','RSR','RSL','LSR','RLR','LRL'}


def test_dubins_known_lengths_and_rigid_motion_scaling():
    assert dubins_distance((0,0,0),(10,0,0),2)==pytest.approx(10)
    assert dubins_distance((0,0,0),(2,2,math.pi/2),2)==pytest.approx(math.pi)
    assert dubins_distance((1,2,.7),(1,2,.7+2*math.pi),2)==0
    a=(0.,0.,.8);b=(3.,-2.,-1.1);radius=1.7
    def moved(p):
        co,si=math.cos(.6),math.sin(.6)
        return (p[0]*co-p[1]*si+8,p[0]*si+p[1]*co-5,p[2]+.6)
    length=dubins_distance(a,b,radius)
    assert dubins_distance(moved(a),moved(b),radius)==pytest.approx(length)
    assert dubins_distance((0,0,a[2]),(9,-6,b[2]),radius*3)==pytest.approx(length*3)
    with pytest.raises(ValueError):dubins_distance(a,b,0)


def test_versioned_guide_keeps_goal_gates_and_default_serialization():
    old=EnvConfig();new=EnvConfig(planner=PlannerConfig(heuristic_mode='dubins_v1'))
    assert 'heuristic_mode' not in old.to_dict()['planner']
    assert EnvConfig.from_dict(old.to_dict()).to_dict()==old.to_dict()
    assert 'heuristic_mode' not in signature(old)['planner']
    assert signature(old)!=signature(new)
    vehicle=VehicleConfig();world=World(30,30)
    for mode in ('euclidean','dubins_v1'):
        planner=HybridAStar(vehicle,PlannerConfig(max_seconds=None,analytic_expansion=True,
            vectorized_primitives=True,heuristic_mode=mode))
        goal=(18.,15.,.3)
        result=planner.plan((5.,10.,0.),goal,world)
        assert result.status=='success'
        assert all(world.free(p,vehicle) for p in result.path)
        assert math.dist(result.path[-1,:2],goal[:2])<=planner.config.goal_tolerance
        assert abs(wrap(result.path[-1,2]-goal[2]))<=planner.config.yaw_tolerance
        nearby=planner.plan((18.,15.,.31),goal,world)
        assert nearby.status=='success' and len(nearby.path)==1


def test_measured_development_wait_has_a_valid_path_despite_legacy_open_set_exhaustion():
    case=json.loads((Path(__file__).parent/'fixtures/native_wait_state_v1.json').read_text())
    config=EnvConfig.from_dict(case['config']);data=case['world']
    world=World(data['width'],data['height'],data['static'],road=RoadGrid(data['road_grid']))
    pose=tuple(case['state'][k] for k in ('x','y','yaw'))
    original=HybridAStar(config.vehicle,config.planner).plan(pose,case['goal'],world)
    assert original.status=='no_path' and original.failure_reason=='open_set_exhausted'
    config.planner.heuristic_mode='dubins_v1'
    guided=HybridAStar(config.vehicle,config.planner).plan(pose,case['goal'],world)
    assert guided.status=='success' and guided.expanded_nodes<original.expanded_nodes
    reference=parameterize(guided.path,1.5,case['state']['v'],config.vehicle,case['state']['time'],True)
    assert reference.v[-1]==0
    assert all(world.free((x,y,yaw),config.vehicle) for x,y,yaw in zip(reference.x,reference.y,reference.yaw))
