"""Direct-policy views of measured CARLA teacher frames, without a plant."""
from collections import deque
from types import SimpleNamespace
import math
import numpy as np

from .env import NavigationEnv
from .imitation import command_action
from .road_grid import RoadGrid
from .vehicle import Command, VehicleState
from .world import World, MovingObstacle


def _actor_snapshot(actors, identities=None):
    if not isinstance(actors, list):
        raise ValueError('Measured dynamic actor snapshot is missing')
    keys, states = [], []
    for actor in actors:
        identity = (actor.get('actor_id'), actor.get('actor_type'))
        state = actor.get('state', {})
        if (type(identity[0]) is not int or identity[0] <= 0 or not isinstance(identity[1], str)
                or not identity[1].startswith(('walker.', 'vehicle.'))
                or set(state) != {'x', 'y', 'vx', 'vy', 'radius'}
                or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in state.values())
                or state['radius'] <= 0):
            raise ValueError('Invalid measured dynamic actor identity/state')
        keys.append(identity); states.append(state)
    if len(set(keys)) != len(keys) or identities is not None and tuple(keys) != identities:
        raise ValueError('Measured dynamic actor identity/order changed')
    return tuple(keys), states


def measured_flat_env(config, scene, initial, *, begin=None):
    if (config.method != 'B1' or config.flat_hold != 1 or config.vehicle.input_delay_steps
            or config.flat_action_catalog != 'continuous' or config.flat_longitudinal_mode != 'acceleration'):
        raise ValueError('Measured direct views require hold1 continuous acceleration/steering')
    if (initial['dynamic'] or scene.get('dynamic')) and begin is None:
        raise ValueError('Dynamic views require the measured bridge reset acknowledgement')
    identities, states = None, []
    if begin is not None:
        identities, states = _actor_snapshot(begin.get('native_dynamic_actors', [] if begin.get('dynamic') == [] else None))
        if (states != initial['dynamic'] or begin.get('dynamic') != states
                or begin.get('state') != initial['initial'] or begin.get('scene') != scene
                or type(begin.get('frame')) is not int or not isinstance(begin.get('episode_id'), str)):
            raise ValueError('Measured reset actor/task/frame identity differs')
    road = RoadGrid(scene['road_grid'])
    if road.sha256 != initial['road_sha256'] or scene['goal'] != initial['goal'] or scene['static'] != initial['static']:
        raise ValueError('Measured task geometry differs')
    env = NavigationEnv(config)
    env.world = World(scene['width'], scene['height'], scene['static'], road=road)
    env.world.road_collision_backend = config.road_collision_backend
    env.world.dynamic = [MovingObstacle(**state) for state in states]
    env.goal = initial['goal']
    env.navigation_route = scene['native_layout']['path']
    env.last_behavior = np.zeros(4)
    env.execution_outcome = np.zeros(6, dtype=np.float32)
    # Intentionally has no step method: only already measured state is assigned.
    env.plant = SimpleNamespace(state=VehicleState(**initial['initial']), previous_command=Command(),
                               delay_queue=deque(), evidence=initial.get('actuator_evidence', {}))
    if begin is not None:
        env._measured_actor_ids = identities
        env._measured_episode = begin['episode_id']
        env._measured_frame = begin['frame']
    return env


def measured_label(env, tick):
    """Apply the production action bounds to small QP feasibility residuals.

    Requests and acknowledged controls remain unchanged in the raw evidence.
    Residuals above1e-3 are rejected, except the explicitly verified production
    projection of a brake holding a measured wheel-angle overshoot.
    """
    command = tick['requested_command']
    vehicle = env.config.vehicle
    values = (command['acceleration'], command['steering'])
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Teacher command exceeds the direct-policy physical limits')
    outside = (not -vehicle.max_decel-1e-3 <= values[0] <= vehicle.max_accel+1e-3
               or abs(values[1]) > vehicle.max_steer+1e-3)
    if outside:
        # Braking holds the measured wheel angle, which can slightly overshoot
        # its commanded limit. Accept only the exact recorded bridge projection
        # of this known brake command, not an enlarged QP feasibility tolerance.
        if (tick.get('controller_status') not in ('dynamic_brake', 'plan_failed', 'wait', 'parking_brake')
                or abs(values[0]+vehicle.max_decel) > 1e-12
                or abs(values[1]-env.state.steering) > 1e-12):
            raise ValueError('Teacher command exceeds the direct-policy physical limits')
        limit = vehicle.steer_rate*env.config.control_dt
        expected = {'acceleration': -vehicle.max_decel,
                    'steering': max(-vehicle.max_steer, min(vehicle.max_steer,
                        max(env.plant.previous_command.steering-limit,
                            min(env.plant.previous_command.steering+limit, values[1]))))}
        if any(abs(tick['command'][name]-value) > 1e-12 for name,value in expected.items()):
            raise ValueError('Brake acknowledgement is not the recorded production projection')
        command = tick['command']
    after = VehicleState(**tick['state'])
    if abs(after.time-env.state.time-env.config.control_dt) > 1e-6:
        raise ValueError('Missing, duplicated or reordered measured control')
    if hasattr(env, '_measured_actor_ids'):
        evidence = tick.get('backend_evidence', {})
        _actor_snapshot(evidence.get('native_dynamic_actors', [] if not env._measured_actor_ids else None), env._measured_actor_ids)
        if (evidence.get('episode_id') != env._measured_episode
                or evidence.get('frame') != env._measured_frame+1
                or tick.get('backend_frame') != evidence.get('frame')):
            raise ValueError('Measured dynamic episode/frame sequence differs')
    label = command_action(command, vehicle)
    if hasattr(env, '_measured_actor_ids'):
        # A bounded direct request must produce the recorded acknowledgement
        # through the actual bridge's acceleration/rate/absolute projection.
        # Float32 labels introduce only their explicit numeric quantization.
        acceleration = float(label[0])*(vehicle.max_accel if label[0] >= 0 else vehicle.max_decel)
        steering = float(label[1])*vehicle.max_steer
        limit = vehicle.steer_rate*env.config.control_dt
        projected = {'acceleration': max(-vehicle.max_decel, min(vehicle.max_accel, acceleration)),
                     'steering': max(-vehicle.max_steer, min(vehicle.max_steer,
                         max(env.plant.previous_command.steering-limit,
                             min(env.plant.previous_command.steering+limit, steering))))}
        if any(abs(tick['command'][name]-value) > 1e-6 for name,value in projected.items()):
            raise ValueError('Direct label does not reproduce the measured bridge command projection')
    return label


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
    if hasattr(env, '_measured_actor_ids'):
        _, states = _actor_snapshot(env.plant.evidence.get('native_dynamic_actors', [] if not env._measured_actor_ids else None), env._measured_actor_ids)
        env.world.dynamic = [MovingObstacle(**state) for state in states]
        env._measured_frame = env.plant.evidence['frame']
    env.execution_outcome = np.array([0., 0., 0., 0., 0.,
        (env.state.v-before.v)/env.config.vehicle.max_speed], dtype=np.float32)
