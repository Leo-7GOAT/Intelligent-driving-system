"""Necessary XY connectivity under a shared road/static geometry model.

The occupied grid centers are a SUBSET of obstacle points. Their EDT is an
upper bound on distance to the complete obstacle union. A query cell is
removed only when EDT(center)+half_diagonal < body_inscribed_radius, proving
that every axle point in that cell collides. Eight-neighbor connectivity of
the remaining cells therefore over-approximates continuous car feasibility.
Connected does not certify a kinematic/physical path. Disconnected certifies
only impossibility under this declared road/static model, not CARLA meshes.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt,label


def necessary_connectivity(road,static,start,goal,vehicle,goal_tolerance):
    if not math.isfinite(goal_tolerance) or goal_tolerance<0:raise ValueError('Finite nonnegative goal region required')
    radius=min(vehicle.front,vehicle.rear,vehicle.width/2)
    dy,dx=road.dy,road.dx
    yy,xx=np.meshgrid((np.arange(road.size)+.5)*dy,(np.arange(road.size)+.5)*dx,indexing='ij')
    clear=road.collision_drivable.copy()
    for x0,y0,x1,y1 in static:
        clear &= ~((xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1))
    # Padding centers lie outside the allowed map and are valid obstacle points.
    distance=distance_transform_edt(np.pad(clear,1,constant_values=False),sampling=(dy,dx))[1:-1,1:-1]
    half_diagonal=math.hypot(dx,dy)/2
    possible=distance+half_diagonal>=radius
    components,count=label(possible,np.ones((3,3),dtype=np.uint8))
    sx,sy=float(start[0]),float(start[1]);gx,gy=float(goal[0]),float(goal[1])
    if not 0<=sx<road.width or not 0<=sy<road.height:raise ValueError('Start outside declared map')
    source=int(components[min(road.size-1,int(sy/dy)),min(road.size-1,int(sx/dx))])
    goal_cells=np.hypot(xx-gx,yy-gy)<=goal_tolerance+half_diagonal
    reached=source!=0 and bool(np.any(goal_cells&(components==source)))
    return dict(necessary_connected=bool(reached),model_infeasibility_certificate=not reached,
        body_inscribed_radius_m=radius,query_half_diagonal_m=half_diagonal,
        conservative_possible_cells=int(possible.sum()),components=int(count),start_component=source,
        goal_region_cells=int(goal_cells.sum()),goal_position_tolerance_m=goal_tolerance,
        scope='Necessary XY over-approximation under declared closed road cells/static rectangles only;connected is not a feasibility certificate.')
