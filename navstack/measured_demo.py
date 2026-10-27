"""Direct-policy views of measured static CARLA teacher frames, without a plant."""
from collections import deque
from types import SimpleNamespace
import math
import numpy as np

from .env import NavigationEnv
from .imitation import command_action
from .road_grid import RoadGrid
from .vehicle import Command, VehicleState
from .world import World


def measured_flat_env(config, scene, initial):
    if (config.method != 'B1' or config.flat_hold != 1 or config.vehicle.input_delay_steps
            or config.flat_action_catalog != 'continuous' or config.flat_longitudinal_mode != 'acceleration'):
        raise ValueError('Measured direct views require hold1 continuous acceleration/steering')
    if initial['dynamic'] or scene.get('dynamic'):
        raise ValueError('This diagnostic supports measured static scenes only')
    road = RoadGrid(scene['road_grid'])
    if road.sha256 != initial['road_sha256'] or scene['goal'] != initial['goal'] or scene['static'] != initial['static']:
        raise ValueError('Measured task geometry differs')
    env = NavigationEnv(config)
    env.world = World(scene['width'], scene['height'], scene['static'], road=road)
    env.world.road_collision_backend = config.road_collision_backend
    env.goal = initial['goal']
    env.navigation_route = scene['native_layout']['path']
    env.last_behavior = np.zeros(4)
    env.execution_outcome = np.zeros(6, dtype=np.float32)
    # Intentionally has no step method: only already measured state is assigned.
    env.plant = SimpleNamespace(state=VehicleState(**initial['initial']), previous_command=Command(),
                               delay_queue=deque(), evidence=initial.get('actuator_evidence', {}))
    return env


def measured_label(env, tick):
    """Apply the production action bounds to small QP feasibility residuals.

    Requests and acknowledged controls remain unchanged in the raw evidence.
    Residuals above1e-3 are rejected instead of becoming clipped supervision.
    """
    command = tick['requested_command']
    vehicle = env.config.vehicle
    values = (command['acceleration'], command['steering'])
    if (not all(math.isfinite(v) for v in values)
            or not -vehicle.max_decel-1e-3 <= values[0] <= vehicle.max_accel+1e-3
            or abs(values[1]) > vehicle.max_steer+1e-3):
        raise ValueError('Teacher command exceeds the direct-policy physical limits')
    after = VehicleState(**tick['state'])
    if abs(after.time-env.state.time-env.config.control_dt) > 1e-6:
        raise ValueError('Missing, duplicated or reordered measured control')
    return command_action(command, vehicle)


def advance_measured_flat(env, tick, action):
    """Method-correct history for one direct control; never execute a command."""
    expected = measured_label(env, tick)
    if not np.array_equal(np.asarray(action), expected):
        raise ValueError('Direct label differs from the measured requested command')
    before = env.state
    env.last_behavior = np.r_[expected, 0., 0.]
    env.plant.state = VehicleState(**tick['state'])
    env.plant.previous_command = Command(**tick['command'])
    env.plant.evidence = tick.get('backend_evidence', {})
    env.execution_outcome = np.array([0., 0., 0., 0., 0.,
        (env.state.v-before.v)/env.config.vehicle.max_speed], dtype=np.float32)
