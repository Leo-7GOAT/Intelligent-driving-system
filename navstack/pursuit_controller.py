"""Timed pure pursuit plus acceleration PI,for fixed-reference comparisons only."""
import math,time
import numpy as np
from .mpc import ControlResult
from .vehicle import Command,brake,wrap

PARAMETERS={'lookahead_base_m':1.5,'lookahead_speed_seconds':.3,
            'speed_kp':1.2,'speed_ki':.3,'progress_kp':.5,'integral_limit':1.}


class TimedPursuitPI:
    def __init__(self,vehicle,dt=.05,**parameters):
        if set(parameters)-set(PARAMETERS):raise ValueError('Unknown pursuit controller parameter')
        self.parameters={**PARAMETERS,**parameters}
        if not math.isfinite(dt) or dt<=0 or any(not math.isfinite(x) or x<=0 for x in self.parameters.values()):
            raise ValueError('Positive finite controller period/gains required')
        self.vehicle,self.dt,self.integral=vehicle,dt,0.

    def step(self,state,reference,previous_command):
        started=time.perf_counter();c=self.vehicle;p=self.parameters
        if reference is None:
            return ControlResult(brake(state,c),'failed',{'reason':'no_reference','qp_seconds':0.,'control_seconds':time.perf_counter()-started})
        points=np.c_[reference.x,reference.y];segments=np.diff(points,axis=0)
        fraction=np.clip(np.sum((np.array([state.x,state.y])-points[:-1])*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-12),0,1)
        projected=points[:-1]+fraction[:,None]*segments
        nearest=int(np.argmin(np.linalg.norm(projected-[state.x,state.y],axis=1)))
        progress=float(reference.s[nearest]+fraction[nearest]*(reference.s[nearest+1]-reference.s[nearest]))
        lookahead=p['lookahead_base_m']+p['lookahead_speed_seconds']*abs(state.v)
        target_s=min(float(reference.s[-1]),progress+lookahead)
        tx=float(np.interp(target_s,reference.s,reference.x));ty=float(np.interp(target_s,reference.s,reference.y))
        alpha=wrap(math.atan2(ty-state.y,tx-state.x)-state.yaw)
        steering=math.atan2(2*c.wheelbase*math.sin(alpha),max(1e-3,math.hypot(tx-state.x,ty-state.y)))
        steering=float(np.clip(steering,max(-c.max_steer,previous_command.steering-c.steer_rate*self.dt),min(c.max_steer,previous_command.steering+c.steer_rate*self.dt)))
        relative_time=state.time+self.dt-reference.start_time
        timed_s=float(np.interp(relative_time,reference.t,reference.s))
        timed_v=float(np.interp(relative_time,reference.t,reference.v))
        remaining=max(0.,float(reference.s[-1])-progress)
        target_v=float(np.clip(timed_v+p['progress_kp']*(timed_s-progress),0,min(c.max_speed,math.sqrt(2*c.max_decel*remaining))))
        error=target_v-state.v
        integral=float(np.clip(self.integral+error*self.dt,-p['integral_limit'],p['integral_limit']))
        feedforward=float(np.interp(relative_time,reference.t,reference.a))
        acceleration=feedforward+p['speed_kp']*error+p['speed_ki']*integral
        if -c.max_decel<=acceleration<=c.max_accel or acceleration*error<0:self.integral=integral
        acceleration=float(np.clip(acceleration,-c.max_decel,c.max_accel))
        return ControlResult(Command(acceleration,steering),'solved',{
            'controller':'timed_pursuit_pi_v1','qp_applicable':False,'qp_seconds':0.,
            'progress_m':progress,'lookahead_m':lookahead,'target_speed_m_s':target_v,
            'control_seconds':time.perf_counter()-started})
