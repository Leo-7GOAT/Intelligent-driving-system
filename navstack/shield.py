"""Common inexpensive execution brake. No search, MPC, or hidden future state."""
import math
from copy import copy
import numpy as np
from .vehicle import Command


def risk_brake(state, command, world, vehicle):
    visible = copy(world)
    visible.dynamic = [o for o in world.dynamic if math.hypot(o.x-state.x,o.y-state.y)<=15]
    # Constant current curvature, conservative reaction + stopping distance.
    horizon = .25 + max(0.,state.v)/vehicle.max_decel
    curvature = math.tan(state.steering)/vehicle.wheelbase
    for t in np.linspace(0,horizon,8):
        distance=max(0.,state.v)*t
        angle=curvature*distance
        dx=distance if abs(curvature)<1e-8 else math.sin(angle)/curvature
        dy=0. if abs(curvature)<1e-8 else (1-math.cos(angle))/curvature
        pose=(state.x+math.cos(state.yaw)*dx-math.sin(state.yaw)*dy,
              state.y+math.sin(state.yaw)*dx+math.cos(state.yaw)*dy,state.yaw+angle)
        if visible.collision(pose,vehicle,float(t)) or visible.out_of_bounds(pose,vehicle):
            return Command(-vehicle.max_decel,state.steering),True
    return command,False
