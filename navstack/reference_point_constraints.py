"""Planning-only exact native counterexamples checked against the reference tube."""
import math
import numpy as np

VERSION='native_counterexample_point_tube_v1'
ROUNDTRIP_TOLERANCE=1e-5  # float32 native world-coordinate round-trip, metres
MAX_POINTS=8192


def settings(protocol):
    value=protocol.get('reference_point_constraints')
    if value is None:return None
    from .steering_retiming import settings as joint_settings
    if value!={'version':VERSION} or not joint_settings(protocol):
        raise ValueError('Exact point constraints require registered frozen joint-speed evaluation')
    return value


def signature(world):
    points=getattr(world,'_reference_point_constraints',None)
    if points is None:return None
    if getattr(world,'_reference_point_constraint_version',None)!=VERSION:
        raise ValueError('Unknown planning point-constraint model')
    array=np.asarray(points,dtype=np.float64).reshape(-1,2)
    if len(array)>MAX_POINTS or not np.isfinite(array).all():raise ValueError('Invalid bounded planning point set')
    return VERSION,ROUNDTRIP_TOLERANCE,array.shape,array.tobytes()


def local_points(points,layout):
    angle=math.radians(layout['angle']);c,s=math.cos(angle),math.sin(angle);ox,oy=layout['origin']
    result=set()
    for point in points:
        if len(point)!=3 or not all(math.isfinite(v) for v in point):raise ValueError('Invalid native counterexample')
        dx,dy=point[0]-ox,point[1]-oy
        result.add((dx*c+dy*s,dx*s-dy*c))
    return tuple(sorted(result))


def allowed(poses,vehicle,points,margin=0.):
    """Full body avoids known points at all nine unchanged query offsets.

    This adds constraints only at measured rejected locations, without eroding
    the whole road raster. Native validation of the entire reference remains
    required; satisfying these points is not a road-safety certificate.
    """
    poses=np.asarray(poses,dtype=float);points=np.asarray(points,dtype=float)
    if poses.ndim!=2 or poses.shape[1]!=3 or not np.isfinite(poses).all():raise ValueError('Finite N x3 poses required')
    if not math.isfinite(margin) or margin<0:raise ValueError('Nonnegative finite point margin required')
    valid=np.ones(len(poses),dtype=bool)
    if points.size==0:return valid
    if points.ndim!=2 or points.shape[1]!=2 or len(points)>8192 or not np.isfinite(points).all():
        raise ValueError('Bounded finite local point list required')
    base_yaw=poses[:,2];c,s=np.cos(base_yaw),np.sin(base_yaw)
    pad=margin+ROUNDTRIP_TOLERANCE
    for first in range(0,len(points),128):
        chunk=points[first:first+128]
        for lat in (-.05,0.,.05):
            dx=chunk[None,:,0]-poses[:,None,0]+lat*s[:,None]
            dy=chunk[None,:,1]-poses[:,None,1]-lat*c[:,None]
            for turn in (-.05,0.,.05):
                co,si=np.cos(base_yaw+turn)[:,None],np.sin(base_yaw+turn)[:,None]
                longitudinal=dx*co+dy*si;lateral=-dx*si+dy*co
                inside=((longitudinal>=-vehicle.rear-pad)&(longitudinal<=vehicle.front+pad)
                    &(abs(lateral)<=vehicle.width/2+pad))
                valid &= ~inside.any(axis=1)
    return valid
