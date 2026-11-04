"""Observable map-route errors; never change measured state or choose a command."""
import math
import numpy as np

VERSION='known_route_frenet4_v1'


def route_error_features(state,route,vehicle):
    path=np.asarray(route,dtype=float)
    if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all():
        raise ValueError('Require finite known-route x/y/yaw samples')
    delta=np.diff(path[:,:2],axis=0);length=np.linalg.norm(delta,axis=1)
    if np.any(length<=1e-8):raise ValueError('Route contains zero-length segments')
    point=np.array([state.x,state.y]);fraction=np.clip(((point-path[:-1,:2])*delta).sum(1)/length**2,0.,1.)
    projected=path[:-1,:2]+fraction[:,None]*delta
    index=int(np.argmin(np.square(projected-point).sum(1)))
    yaw=np.unwrap(path[:,2]);heading=float(yaw[index]+fraction[index]*(yaw[index+1]-yaw[index]))
    error=point-projected[index]
    lateral=-math.sin(heading)*error[0]+math.cos(heading)*error[1]
    angle=math.atan2(math.sin(heading-state.yaw),math.cos(heading-state.yaw))
    curvature=float((yaw[index+1]-yaw[index])/length[index])
    return np.array([lateral/3.,math.sin(angle),math.cos(angle),curvature*vehicle.min_turn_radius],dtype=np.float32)
