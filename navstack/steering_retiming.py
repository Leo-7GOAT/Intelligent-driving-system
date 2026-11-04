"""Opt-in sampled-reference retiming with steering-rate and acceleration bounds.

Geometry is unchanged. The first feedforward curvature is anchored to the
measured wheel angle. Bounds apply to MPC's piecewise-linear curvature in time;
they are a planning model, not a guarantee about native tracking or road safety.
"""
from dataclasses import replace
import math
import numpy as np

VERSION='measured_steering_rate_retiming_v2'


def settings(protocol):
    value=protocol.get('reference_joint_speed')
    if value is None:return None
    from .reference_compute import enabled
    from .fixed_lowlevel_training import registered
    if (value!={'version':VERSION,'spatial_speed_ceiling':.25}
            or not (protocol.get('evaluation_only') is True or registered(protocol)) or not enabled(protocol)):
        raise ValueError('Joint-speed prototype requires registered exact-compute frozen evaluation')
    return value


def retime(reference,vehicle,initial_steering):
    if not math.isfinite(initial_steering) or abs(initial_steering)>=math.pi/2:
        raise ValueError('retiming_invalid_measured_steering')
    s=np.asarray(reference.s,dtype=float);kappa=np.array(reference.kappa,dtype=float,copy=True)
    speed=np.array(reference.v,dtype=float,copy=True)
    if (len(s)<3 or s.shape!=kappa.shape or s.shape!=speed.shape
            or not np.isfinite(np.r_[s,kappa,speed]).all()
            or np.any(np.diff(s)<=0) or np.any(speed[1:]<0)):
        raise ValueError('retiming_invalid_reference')
    initial_speed=float(speed[0]);speed[0]=abs(initial_speed)
    kappa[0]=math.tan(initial_steering)/vehicle.wheelbase
    if np.max(abs(kappa[1:]))>1/vehicle.min_turn_radius+.015:
        raise ValueError('retiming_curvature_limit')
    ds=np.diff(s);left,right=kappa[:-1],kappa[1:]
    nearest=np.where(left*right<=0,0.,np.minimum(abs(left),abs(right)))
    # max |d atan(L*kappa)/dt| on each linear-kappa interval, not merely
    # endpoint angle differences (which underestimate the rate near zero).
    minimum_time=vehicle.wheelbase*abs(right-left)/(vehicle.steer_rate*(1+(vehicle.wheelbase*nearest)**2))
    caps=np.full(len(ds),vehicle.max_speed,dtype=float)
    np.divide(ds,minimum_time,out=caps,where=minimum_time>0)
    caps=np.minimum(caps,vehicle.max_speed)
    speed[:-1]=np.minimum(speed[:-1],caps);speed[1:]=np.minimum(speed[1:],caps)
    speed=np.minimum(speed,vehicle.max_speed)
    # Propagate braking constraints before fixing the actual measured root.
    for i in range(len(speed)-2,-1,-1):
        speed[i]=min(speed[i],math.sqrt(speed[i+1]**2+2*vehicle.max_decel*ds[i]))
    if abs(initial_speed)>speed[0]+1e-10:
        raise ValueError('retiming_initial_speed_incompatible_with_caps')
    speed[0]=initial_speed
    for i in range(1,len(speed)):
        speed[i]=min(speed[i],math.sqrt(speed[i-1]**2+2*vehicle.max_accel*ds[i-1]))
    denom=speed[:-1]+speed[1:]
    if np.any(denom<=1e-10):raise ValueError('retiming_zero_speed_segment')
    intervals=2*ds/denom
    if np.any(intervals+1e-10<minimum_time):raise ValueError('retiming_steering_rate_violation')
    times=np.r_[0.,np.cumsum(intervals)];acceleration=np.r_[np.diff(speed)/intervals,0.]
    if np.any(acceleration>vehicle.max_accel+1e-9) or np.any(acceleration<-vehicle.max_decel-1e-9):
        raise ValueError('retiming_acceleration_violation')
    return replace(reference,kappa=kappa,v=speed,a=acceleration,t=times)
