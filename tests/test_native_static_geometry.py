import math
import numpy as np
import pytest
from navstack.native_hazards import projected_box_strips,resolve_native_hazard,verify_native_hazard_scene
from test_native_hazards import layout


def vertices(center=(16.,10.),yaw=math.pi/4):
    points=[]
    for x,y in [(-2.4,-1.08),(-2.4,1.08),(2.4,1.08),(2.4,-1.08)]:
        points.append([center[0]+x*math.cos(yaw)-y*math.sin(yaw),center[1]+x*math.sin(yaw)+y*math.cos(yaw)])
    return points+points


def test_strip_union_conservatively_covers_rotated_measured_box_without_whole_box_corners():
    points=vertices();rectangles=projected_box_strips(points)
    assert len(rectangles)==12
    # Dense body samples include every edge/corner. Coverage is also guaranteed
    # by the extrema-of-convex-slice construction, not inferred from this grid.
    for x in np.linspace(-2.4,2.4,41):
        for y in np.linspace(-1.08,1.08,23):
            px=16.+(x-y)/math.sqrt(2);py=10.+(x+y)/math.sqrt(2)
            assert any(a<=px<=c and b<=py<=d for a,b,c,d in rectangles)
    old=(max(p[0] for p in points)-min(p[0] for p in points))*(max(p[1] for p in points)-min(p[1] for p in points))
    area=sum((c-a)*(d-b) for a,b,c,d in rectangles)
    assert 4.8*2.16<=area<.7*old


def test_static_geometry_registration_and_audit_bind_actual_vertices_to_every_strip():
    spec={'version':'native_hazard_v1','kind':'static_vehicle','static_geometry':'projected_box_strips12_v1'}
    observed={**resolve_native_hazard(layout(),spec),'actor_id':42,'actor_type':'vehicle.tesla.model3',
              'static_body_vertices_local_xy':vertices()}
    rectangles=projected_box_strips(observed['static_body_vertices_local_xy'])
    verify_native_hazard_scene(layout(),spec,observed,rectangles,[])
    wrong=[r[:] for r in rectangles];wrong[3][1]+=.1
    with pytest.raises(ValueError,match='strip cover'):verify_native_hazard_scene(layout(),spec,observed,wrong,[])
    with pytest.raises(ValueError):verify_native_hazard_scene(layout(),spec,observed,rectangles[:1],[])
    legacy=resolve_native_hazard(layout(),{'version':'native_hazard_v1','kind':'static_vehicle'})
    assert 'static_geometry' not in legacy
    with pytest.raises(ValueError):resolve_native_hazard(layout(),{**spec,'kind':'crossing_walker'})


@pytest.mark.parametrize('points,count', [([],12),([[1.,2.]]*4,12),([[1.,float('nan')]]*4,12),([1.,2.,3.,4.],12),(vertices(),21),(vertices(),True)])
def test_invalid_body_projection_or_unbounded_observation_count_is_rejected(points,count):
    with pytest.raises(ValueError):projected_box_strips(points,count)
