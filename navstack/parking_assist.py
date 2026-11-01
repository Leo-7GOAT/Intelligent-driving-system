"""Opt-in common physical parking assistance; no state or success modification."""
import math
from .vehicle import Command,wrap

VERSION='inside_goal_pose_overspeed_brake_v2'


def goal_parking_command(state,goal,command,config,progressed):
    if not config.goal_parking_assist or not progressed or abs(state.v)<=config.goal_speed:
        return command,False,False
    eligible=(math.hypot(state.x-goal[0],state.y-goal[1])<=config.goal_tolerance
        and abs(wrap(state.yaw-goal[2]))<=config.goal_yaw_tolerance)
    if not eligible or command.acceleration<=-config.vehicle.max_decel:
        return command,False,eligible
    return Command(-config.vehicle.max_decel,command.steering),True,True
