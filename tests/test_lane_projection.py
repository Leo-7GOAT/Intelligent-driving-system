"""Analytic circular-lane fixtures, not actual CARLA validation."""
import math
from types import SimpleNamespace
from navstack.lane_projection import refined_lane_point, local_driving_union, DrivingSurfaceIndex


class Lane:
    road_id=1
    section_id=0
    lane_type='Driving'
    lane_width=3.5
    def __init__(self,road,lane_id,s):
        self.road,self.lane_id,self.s=road,lane_id,s
        r=10+3.5*(lane_id-1);a=s/10
        self.transform=SimpleNamespace(location=SimpleNamespace(x=r*math.cos(a),y=r*math.sin(a),z=0))
    def get_left_lane(self):
        return Lane(self.road,1,self.s) if self.lane_id==2 else None
    def get_right_lane(self):
        return Lane(self.road,2,self.s) if self.lane_id==1 else None


class Road:
    def get_waypoint(self,location,**kwargs):return Lane(self,1,5.08)
    def get_waypoint_xodr(self,road_id,lane_id,s):
        return Lane(self,lane_id,s) if 0<=s<50 else None


def location(radius,z=0):
    return SimpleNamespace(x=radius*math.cos(.5),y=radius*math.sin(.5),z=z)


def test_refinement_removes_longitudinal_projection_bias_without_dilation():
    road=Road();w=road.get_waypoint(location(11.749))
    result=refined_lane_point(road,w,location(11.749))
    assert abs(result['s']-5)<1e-4 and abs(result['distance']-1.749)<1e-7
    assert result['inside'] and result['half_width']==1.75
    assert not refined_lane_point(road,w,location(11.751))['inside']


def test_neighbour_union_outside_and_height_rejection():
    road=Road()
    result=local_driving_union(road,location(11.8),'Driving')
    assert result['inside']
    assert not result['candidates'][0]['inside'] and result['candidates'][1]['inside']
    assert not local_driving_union(road,location(15.26),'Driving')['inside']
    assert not local_driving_union(road,location(11.8,z=2),'Driving')['inside']


def test_index_checks_disconnected_road_ids_and_preserves_outer_boundary():
    class SeparateRoads(Road):
        def make(self,road_id,s):
            w=Lane(self,road_id,s);w.road_id=road_id
            w.get_left_lane=lambda:None;w.get_right_lane=lambda:None
            return w
        def get_waypoint(self,point,**kwargs):
            return None if not kwargs.get('project_to_road',True) else self.make(1,5.08)
        def get_waypoint_xodr(self,road_id,lane_id,s):
            return self.make(road_id,s) if 0<=s<50 else None
        def generate_waypoints(self,spacing):
            return [self.make(road_id,s) for road_id in (1,2) for s in range(50)]
    road=SeparateRoads();index=DrivingSurfaceIndex(road,'Driving')
    assert not local_driving_union(road,location(11.8),'Driving')['inside']
    assert index.contains(location(11.8))
    assert not index.contains(location(15.26))
    assert not index.contains(location(11.8,z=2))
