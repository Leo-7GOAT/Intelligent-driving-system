"""Development-only goal-frame subgoal proposal; no policy or execution override.

All candidates are exposed, including infeasible ones. The selected target must
still pass the same Hybrid A*, time parameterization and MPC execution chain.
"""
import math

import numpy as np


def route_frame_candidates(state, route, vehicle, include_final_goal=False):
    """Propose local poses along a known route without moving measured state.

    Route samples are (x, y, yaw) in metres/radians. Nearest-segment projection
    locates a reference arc length ONLY; neither state nor executed trajectory
    is projected. This route prior must be disclosed and shared by comparators.
    Every proposal still needs swept-body planning and actual control execution.
    """
    path=np.asarray(route,dtype=float)
    if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all():
        raise ValueError('Route needs at least two finite x/y/yaw samples')
    delta=np.diff(path[:,:2],axis=0);length=np.linalg.norm(delta,axis=1)
    if np.any(length<=1e-8):raise ValueError('Route must not contain zero-length segments')
    arc=np.r_[0.,np.cumsum(length)]
    fraction=np.clip(((np.array([state.x,state.y])-path[:-1,:2])*delta).sum(axis=1)/length**2,0,1)
    projected=path[:-1,:2]+fraction[:,None]*delta
    nearest=int(np.argmin(np.linalg.norm(projected-np.array([state.x,state.y]),axis=1)))
    progress=arc[nearest]+fraction[nearest]*length[nearest]
    heading=np.unwrap(path[:,2]);result=[(None,0.)]
    for lookahead in (6.,9.,12.):
        target_s=min(arc[-1],progress+lookahead)
        x=float(np.interp(target_s,arc,path[:,0]));y=float(np.interp(target_s,arc,path[:,1]))
        yaw=float(np.interp(target_s,arc,heading));yaw=math.atan2(math.sin(yaw),math.cos(yaw))
        # Same terminal offset envelope as goal-frame proposals. It is a
        # necessary geometric reservation, not a path-feasibility guarantee.
        radius=vehicle.min_turn_radius*1.2
        ratio=min(1.,(arc[-1]-target_s)/(2*radius))
        maximum=2*radius*(1-math.sqrt(max(0.,1-ratio**2)))
        for offset in (-3.,-1.5,0.,1.5,3.):
            offset=max(-maximum,min(maximum,offset))
            target=(x-math.sin(yaw)*offset,y+math.cos(yaw)*offset,yaw)
            for speed in (.75,1.5):result.append((target,speed))
    if include_final_goal:
        # Versioned action semantics: replace only the furthest centre pair.
        # The actual goal is already shared task information. It still needs a
        # complete swept-body plan and MPC; this does not move or stop the ego.
        goal=tuple(float(v) for v in path[-1])
        result[25:27]=[(goal,.75),(goal,1.5)]
    return tuple(result)


def goal_frame_candidates(state, goal, vehicle=None):
    co, si = math.cos(goal[2]), math.sin(goal[2])
    remaining = (goal[0] - state.x) * co + (goal[1] - state.y) * si
    result = [(None, 0.0)]
    for lookahead in (6.0, 9.0, 12.0):
        distance_to_goal = max(0.0, remaining - lookahead)
        for offset in (-3.0, -1.5, 0.0, 1.5, 3.0):
            if vehicle is not None:
                # A pair of opposite-curvature forward arcs can return from
                # this offset to the final heading within the remaining x span.
                # Use a 20% larger turn radius as a tracking reserve. This is a
                # geometric candidate envelope, not a claim about dynamic safety.
                radius = vehicle.min_turn_radius * 1.2
                ratio = min(1.0, distance_to_goal / (2*radius))
                maximum_offset = 2*radius*(1-math.sqrt(max(0.0,1-ratio*ratio)))
                offset = max(-maximum_offset,min(maximum_offset,offset))
            target = (goal[0] - co * distance_to_goal - si * offset,
                      goal[1] - si * distance_to_goal + co * offset, goal[2])
            for speed in (.75, 1.5):
                result.append((target, speed))
    return tuple(result)
