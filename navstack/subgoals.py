"""Development-only goal-frame subgoal proposal; no policy or execution override.

All candidates are exposed, including infeasible ones. The selected target must
still pass the same Hybrid A*, time parameterization and MPC execution chain.
"""
import math


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
