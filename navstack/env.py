from dataclasses import asdict
import math
import time
from copy import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig
from .vehicle import VehiclePlant, Command, brake, wrap
from .world import make_scenario
from .planning import HybridAStar, GridAStarPlanner, primitive
from .trajectory import parameterize, dynamic_safe, tracking_error
from .mpc import MPC
from .reward import RewardCalculator


def action_catalog():
    # Immutable metric ego-frame candidate poses, independent of map dimensions.
    result = [(0.0, 0.0, 0.0, 0.0)]  # deliberate wait/brake
    for length in (3.0, 6.0, 9.0):
        for curvature in (-0.12, -0.06, 0.0, 0.06, 0.12):
            angle = curvature * length
            x = length if curvature == 0 else math.sin(angle) / curvature
            y = 0 if curvature == 0 else (1 - math.cos(angle)) / curvature
            for speed in (0.75, 1.5):
                result.append((x, y, angle, speed))
    return tuple(result)


ACTIONS = action_catalog()
OBS_VERSION = "physical-v2-local11-static20-neighbors3-delay5"

FLAT_SPEEDS=(0.,.75,1.5)
FLAT_STEERS=(-1.,-.5,-.2,0.,.2,.5,1.)


def flat_action_values(action,config):
    """Decode a versioned direct control action, without planning or a teacher."""
    values=np.asarray(action)
    if config.flat_action_catalog == 'speed_steer_3x7_v1':
        if (values.shape!=(2,) or not np.issubdtype(values.dtype,np.integer) or
                not 0<=values[0]<3 or not 0<=values[1]<7):
            raise ValueError('Invalid categorical speed/steering action')
        speed=min(config.vehicle.max_speed,FLAT_SPEEDS[int(values[0])])
        return np.array([2*speed/config.vehicle.max_speed-1,FLAT_STEERS[int(values[1])]],dtype=np.float32)
    values=np.asarray(action,dtype=np.float32)
    if values.shape!=(2,) or not np.isfinite(values).all():raise ValueError('invalid continuous action')
    return np.clip(values,-1,1)


def candidate_goal_witness(target, goal, world, vehicle):
    """Sufficient static continuation evidence, including an already-reached pose."""
    from .planning import analytic_connection
    if not world.free(target, vehicle):
        return False
    if math.dist(target[:2], goal[:2]) < 1e-6 and abs(wrap(target[2]-goal[2])) < 1e-6:
        return True
    return analytic_connection(target, goal, world, vehicle) is not None


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or EnvConfig()
        c = self.config
        self.action_space = (
            spaces.Box(-1.0, 1.0, (2,), np.float32)
            if c.method == "B1"
            else spaces.Discrete(len(ACTIONS))
        )
        if c.method=='B1' and c.flat_action_catalog=='speed_steer_3x7_v1':
            self.action_space=spaces.MultiDiscrete([3,7])
        # Same information for every policy: 163 local features + 20 known static
        # rectangles (four ego-frame corners + mask) + four map boundary corners.
        obs_size = 605 if c.candidate_observation else (351 if c.execution_feedback == "off" else 450)
        if c.road_observation:
            obs_size += 64*64
        if c.actuator_observation != "off":
            obs_size += 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.planner = (GridAStarPlanner if c.method == "B2" else HybridAStar)(
            c.vehicle, c.planner
        )
        self.reward_calculator = RewardCalculator(c.reward, c.control_dt)
        self.reset_count = 0
        self.total_physics_steps = 0
        self.total_actions = 0
        self.done = True

    @property
    def state(self):
        return self.plant.state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world, self.plant, self.goal, self.scenario_name = self._create_episode(options)
        self.world.road_collision_backend = self.config.road_collision_backend
        self.navigation_route = None
        if self.config.subgoal_mode == "route_frame_terminal":
            route = getattr(self, "scene_evidence", {}).get("native_layout", {}).get("path")
            if route is None:
                raise ValueError("Route catalog needs an explicit known navigation route")
            self.navigation_route = tuple(tuple(row) for row in route)
            # Validate before recording an episode or exposing a policy observation.
            self._route_candidates()
        initial = self.plant.state
        if len(self.world.static) > 20:
            raise ValueError("static-map observation capacity exceeded")
        if self.config.road_observation and self.world.road is None:
            raise ValueError("Measured road observations require measured road geometry")
        self.controller = MPC(
            self.config.vehicle, self.config.control_dt, self.config.mpc_horizon
        )
        self.reset_count += 1
        self.done = False
        self.collision_history = False
        self.ever_controlled = False
        self.last_behavior = np.zeros(4)
        self.reference = None
        self._reference_selection = None
        self._rule_plan_cache = None
        self.rule_diagnostics = None
        self.execution_outcome = np.zeros(6, dtype=np.float32)
        self.feedback_seconds = 0.0
        self.start_distance = math.hypot(
            initial.x - self.goal[0], initial.y - self.goal[1]
        )
        self.episode = {
            "reward": 0.0,
            "discounted_return": 0.0,
            "physics_steps": 0,
            "actions": 0,
            "distance": 0.0,
            "plan_failures": 0,
            "mpc_failures": 0,
            "interventions": 0,
            "reward_components": {},
            "plan_times": [],
            "control_times": [],
            "qp_times": [],
            "control_cycle_times": [],
            "decision_cycle_times": [],
            "lateral_errors": [],
            "heading_errors": [],
            "speed_errors": [],
            "jerk": [],
            "lateral_acceleration": [],
            "risk": [],
        }
        self.initial_scenario = {
            "name": self.scenario_name,
            "initial": asdict(initial),
            "goal": self.goal,
            "static": list(self.world.static),
            "dynamic": [asdict(o) for o in self.world.dynamic],
        }
        if self.world.road is not None:
            self.initial_scenario["road_sha256"] = self.world.road.sha256
        if self.config.actuator_observation != "off":
            self._actuator_observation()
            self.initial_scenario["actuator_evidence"] = {k: self.plant.evidence[k]
                for k in ("actuator_observation", "speed_setpoint", "actuation_state")}
        return self._observation(), {"scenario": self.scenario_name, "success": False}

    def _create_episode(self, options=None):
        route_length = None
        route_stage_active = False
        if self.config.split == "train":
            completed_routes = []
            for stage in self.config.route_stages:
                if self.total_physics_steps < stage["until_physics_steps"]:
                    route_length = stage["route_m"]
                    route_stage_active = True
                    break
                completed_routes.append(stage["route_m"])
            probability = self.config.route_rehearsal_probability
            if completed_routes and probability and self.np_random.random() < probability:
                # Physical on-policy episodes, not replay-buffer or expert data.
                # Old tasks remain available after the final mixed stage begins.
                route_length = float(self.np_random.choice(completed_routes))
                route_stage_active = True
        if self.config.curriculum and self.config.split == "train":
            route_length = 6.0 if self.total_physics_steps < 50000 else (16.0 if self.total_physics_steps < 150000 else None)
        if self.config.split == "train" and self.config.training_route_length is not None:
            route_length = self.config.training_route_length
        if self.config.split == "train" and self.config.training_route_range is not None:
            route_length = float(self.np_random.uniform(*self.config.training_route_range))
        if options and "diagnostic_route_length" in options:
            route_length = float(options["diagnostic_route_length"])
            if not 2 <= route_length <= self.config.width - 11:
                raise ValueError("diagnostic route does not fit map")
        scenario_config = self.config
        if route_stage_active or (self.config.split == "train" and self.config.scenario == "mixed"
                and self.total_physics_steps < self.config.empty_pretraining_steps):
            # Same full-length task for every architecture; only scene difficulty
            # is staged. Switch at episode boundaries, never reset in env.step.
            scenario_config = copy(self.config)
            scenario_config.scenario = "empty"
        world, initial, goal, name = make_scenario(scenario_config, self.np_random, route_length)
        return world, VehiclePlant(self.config.vehicle, initial), goal, name

    def decode_action(self, action):
        if self.config.subgoal_mode == "route_frame_terminal":
            return self._route_candidates()[int(action)]
        if self.config.subgoal_mode.startswith("goal_frame"):
            from .subgoals import goal_frame_candidates
            return goal_frame_candidates(self.state, self.goal, self.config.vehicle if self.config.subgoal_mode == "goal_frame_terminal" else None)[int(action)]
        x, y, yaw, speed = ACTIONS[int(action)]
        s = self.state
        co, si = math.cos(s.yaw), math.sin(s.yaw)
        return (s.x + co * x - si * y, s.y + si * x + co * y, wrap(s.yaw + yaw)), speed

    def _route_candidates(self):
        from .subgoals import route_frame_candidates
        route = getattr(self, "navigation_route", None)
        if route is None:
            raise ValueError("Route catalog needs an explicit known navigation route")
        candidates = route_frame_candidates(self.state, route, self.config.vehicle, self.config.route_goal_candidate)
        if (math.dist(route[-1][:2], self.goal[:2]) > 1e-5
                or abs(wrap(route[-1][2]-self.goal[2])) > 1e-5):
            raise ValueError("Navigation route endpoint differs from task goal")
        return candidates

    def _observation(self):
        s = self.state
        co, si = math.cos(s.yaw), math.sin(s.yaw)

        def local(dx, dy):
            return co * dx + si * dy, -si * dx + co * dy

        queue = [
            value
            for command in self.plant.delay_queue
            for value in (command.acceleration, command.steering)
        ]
        queue += [0.0] * (10 - len(queue))
        ego = [
            s.v,
            s.steering,
            s.acceleration,
            self.plant.previous_command.acceleration,
            self.plant.previous_command.steering,
            float(len(self.plant.delay_queue)),
        ] + queue
        dx, dy = local(self.goal[0] - s.x, self.goal[1] - s.y)
        angle = wrap(self.goal[2] - s.yaw)
        goal = [
            dx,
            dy,
            math.sin(angle),
            math.cos(angle),
            math.hypot(dx, dy),
            max(0, self.config.deadline - s.time),
            s.time,
        ]
        grid = []
        for yy in np.linspace(-7.5, 7.5, 11):
            for xx in np.linspace(-3, 12, 11):
                grid.append(
                    float(
                        self.world.occupied(
                            s.x + co * xx - si * yy, s.y + si * xx + co * yy
                        )
                    )
                )
        neighbors = []
        visible = sorted(
            (o for o in self.world.dynamic if math.hypot(o.x - s.x, o.y - s.y) <= 15),
            key=lambda o: (o.x - s.x) ** 2 + (o.y - s.y) ** 2,
        )[:3]
        for o in visible:
            px, py = local(o.x - s.x, o.y - s.y)
            vx, vy = local(o.vx - s.v * co, o.vy - s.v * si)
            neighbors.extend([px, py, vx, vy, 1.0])
        neighbors += [0.0] * (15 - len(neighbors))
        known_map = []
        for x0, y0, x1, y1 in self.world.static:
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                known_map.extend(local(x - s.x, y - s.y))
            known_map.append(1.0)
        known_map += [0.0] * (180 - len(known_map))
        for x, y in (
            (0, 0),
            (self.world.width, 0),
            (self.world.width, self.world.height),
            (0, self.world.height),
        ):
            known_map.extend(local(x - s.x, y - s.y))
        observation = np.array(
            ego + goal + grid + neighbors + list(self.last_behavior) + known_map,
            dtype=np.float32,
        )
        if self.config.execution_feedback != "off":
            started = time.perf_counter()
            extra = self._execution_feedback() if self.config.execution_feedback in ("enabled", "continuation") else np.zeros(99)
            if self.config.candidate_observation:
                # Candidate geometry is policy input, never an action selector.
                # Fixed scales preserve the same meaning across candidate slots.
                geometry = [[0., 0., 0., 1., 0.]]
                for action in range(1, 31):
                    target, speed = self.decode_action(action)
                    tx, ty = local(target[0] - s.x, target[1] - s.y)
                    heading = wrap(target[2] - s.yaw)
                    geometry.append([tx / 15., ty / 15., math.sin(heading), math.cos(heading), speed / 3.])
                candidates = np.c_[geometry, np.asarray(extra[:93]).reshape(31, 3)]
                if self.config.local_candidate_filter or self.config.endpoint_candidate_filter:
                    from .candidate_filter import local_witness_mask,endpoint_clear_mask
                    self._local_candidate_mask=(endpoint_clear_mask if self.config.endpoint_candidate_filter else local_witness_mask)(candidates)
                    self._local_candidate_mask_state=self.state
                extra = np.r_[candidates.ravel(), extra[93:]]
            self.feedback_seconds = time.perf_counter() - started
            observation = np.r_[observation, extra].astype(np.float32)
        if self.config.road_observation:
            # Full raster in the same local map axes as the four already-visible
            # map corners. All architectures receive every cell used by planning.
            observation = np.r_[observation, (~self.world.road.drivable).ravel()].astype(np.float32)
        if self.config.actuator_observation != "off":
            observation = np.r_[observation, self._actuator_observation()].astype(np.float32)
        return observation

    def _actuator_observation(self):
        from .carla_actuation import actuator_state_values
        return actuator_state_values(self.plant.evidence)

    def _execution_feedback(self):
        """Common geometric witness features plus last measured execution outcome.

        No action is replaced or removed. Connector rejection is unknown, not a
        proof that Hybrid A* cannot find another path. B1 can receive the same
        features; H0 receives a same-sized zero vector for the ablation.
        """
        from .planning import analytic_connection
        s, vehicle = self.state, self.config.vehicle
        features = [1.0, 1.0, 0.0]  # intentional wait, no progress claim
        cache = {}
        for action in range(1,31):
            target, speed = self.decode_action(action)
            if target not in cache:
                clear = self.world.free(target,vehicle)
                path = analytic_connection((s.x,s.y,s.yaw),target,self.world,vehicle) if clear else None
                continuation = (candidate_goal_witness(target,self.goal,self.world,vehicle)
                    if clear and self.config.execution_feedback == 'continuation' else False)
                cache[target] = (clear,path is not None,continuation)
            clear, witness, continuation = cache[target]
            # Travel-time proxy remains derivable from the shared target/speed
            # geometry. New signatures expose a sufficient continuation witness
            # in that slot; rejection means unknown and never masks an action.
            third = float(continuation) if self.config.execution_feedback == 'continuation' else math.dist((s.x,s.y),target[:2])/max(.3,speed)/45
            features.extend((float(clear),float(witness),third))
        return np.r_[features,self.execution_outcome]

    def reusable_reference(self, target, speed):
        """Retain only the same explicitly selected, still valid timed reference.

        No state projection, retiming, target replacement or policy override.
        Dynamic predictions start at the current measured state of each actor.
        """
        if not self.config.reference_reuse:return False
        reference=getattr(self,'reference',None)
        if (reference is None or getattr(self,'_reference_selection',None)!=(target,speed)):
            return False
        state=self.state;vehicle=self.config.vehicle
        remaining_end=reference.start_time+float(reference.t[-1])
        if remaining_end<=state.time+self.config.control_dt:
            return False
        error=tracking_error(state,reference)
        if (error['lateral']>.25 or abs(error['heading'])>.2
                or not self.world.free((state.x,state.y,state.yaw),vehicle)):
            return False
        # Check the complete remaining spatial reference, including the current
        # interpolated pose, rather than only sparse prediction-time samples.
        keep=reference.start_time+reference.t>=state.time
        poses=zip(reference.x[keep],reference.y[keep],reference.yaw[keep])
        current=reference.sample([state.time])[0]
        if (math.hypot(current[0]-state.x,current[1]-state.y)>.75
                or not self.world.free((current[0],current[1],current[3]),vehicle)
                or not all(self.world.free(p,vehicle) for p in poses)):
            return False
        horizon=min(self.config.prediction_seconds,remaining_end-state.time)
        times=np.linspace(0.,horizon,max(2,math.ceil(horizon/self.config.control_dt)+1))
        predicted=reference.sample(state.time+times)
        visible=[o for o in self.world.dynamic if math.hypot(o.x-state.x,o.y-state.y)<=15]
        return all(self.world.clearance((p[0],p[1],p[3]),vehicle,o,float(t))>.12
                   for t,p in zip(times,predicted) for o in visible)

    def _prepare_reference(self, action):
        if int(action) == 0:
            self.reference = None
            self._reference_selection = None
            return "wait", None
        target, speed = self.decode_action(action)
        s = self.state
        terminal_prefix = False
        if (self.config.subgoal_mode.startswith("goal_frame") or self.config.subgoal_mode == "route_frame_terminal") and self.config.parking:
            terminal_prefix = math.dist(target[:2], self.goal[:2]) < 1e-6
            if (terminal_prefix and math.hypot(s.x-self.goal[0], s.y-self.goal[1]) <= self.config.planner.goal_tolerance
                    and abs(wrap(s.yaw-self.goal[2])) <= self.config.goal_yaw_tolerance):
                self.reference=None;self._reference_selection=None
                return "parking_brake", {"status":"terminal_brake", "planning_time":0., "terminal_goal_prefix":True}
        elif self.config.terminal_goal_prefix and self.config.parking:
            length = (3.0, 6.0, 9.0)[(int(action) - 1) // 10]
            curvature = ACTIONS[int(action)][2] / length
            candidate = primitive((s.x, s.y, s.yaw), math.atan(self.config.vehicle.wheelbase * curvature),
                                  length, self.config.vehicle.wheelbase, .08)
            distances = np.linalg.norm(candidate[:, :2] - self.goal[:2], axis=1)
            closest = int(np.argmin(distances))
            if distances[closest] < self.config.goal_tolerance * .65 and abs(wrap(candidate[closest, 2] - self.goal[2])) < self.config.goal_yaw_tolerance:
                # The selected action already crosses the parking region. End its reference
                # at the task goal BEFORE rejecting an unused endpoint outside the map.
                # This changes only a planned reference; measured state is never changed.
                target = self.goal
                terminal_prefix = True
                if math.hypot(s.x-self.goal[0], s.y-self.goal[1]) <= self.config.planner.goal_tolerance:
                    self.reference=None;self._reference_selection=None
                    return "parking_brake", {"status":"terminal_brake","planning_time":0.,"terminal_goal_prefix":True}
        if self.reusable_reference(target,speed):
            return 'reference',{'status':'reference_reused','expanded_nodes':0,'planning_time':0.,
                'terminal_goal_prefix':terminal_prefix,'reference_start_time':self.reference.start_time,
                'reuse_scope':'same_selected_pose_and_speed_no_retiming'}
        self.reference = None
        self._reference_selection = None
        cached = self._rule_plan_cache
        self._rule_plan_cache = None
        if cached and cached[0] == s and cached[1] == target:
            plan = cached[2]
        else:
            plan = self.planner.plan((s.x, s.y, s.yaw), target, self.world)
        self.episode["plan_times"].append(plan.planning_time)
        detail = {
            "status": plan.status,
            "expanded_nodes": plan.expanded_nodes,
            "cost": plan.cost,
            "planning_time": plan.planning_time,
            "failure_reason": plan.failure_reason,
            "terminal_goal_prefix": terminal_prefix,
        }
        if plan.status != "success" or len(plan.path) < 2:
            self.episode["plan_failures"] += 1
            return "plan_failed", detail
        path = plan.path
        stop = terminal_prefix
        # A path passing through the final parking tolerance is ended there. This only
        # changes the reference: measured vehicle state is never moved to this point.
        distances = np.linalg.norm(path[:, :2] - self.goal[:2], axis=1)
        closest = int(np.argmin(distances))
        if (
            self.config.parking
            and distances[closest] < self.config.goal_tolerance * 0.65
            and closest > 0
            and abs(wrap(path[closest, 2] - self.goal[2]))
            < self.config.goal_yaw_tolerance
        ):
            path = path[: closest + 1]
            stop = True
        try:
            reference = parameterize(
                path, speed, s.v, self.config.vehicle, s.time, stop
            )
            if not all(
                self.world.free((x, y, yaw), self.config.vehicle)
                for x, y, yaw in zip(reference.x, reference.y, reference.yaw)
            ):
                raise ValueError("resampled_body_collision")
        except ValueError as exc:
            detail["failure_reason"] = str(exc)
            detail["status"] = "trajectory_failed"
            self.episode["plan_failures"] += 1
            return "plan_failed", detail
        visible_world = copy(self.world)
        visible_world.dynamic = [
            o for o in self.world.dynamic if math.hypot(o.x - s.x, o.y - s.y) <= 15
        ]
        if not self.config.shared_shield and not dynamic_safe(
            reference,
            visible_world,
            self.config.vehicle,
            self.config.prediction_seconds,
        ):
            detail["failure_reason"] = "constant_velocity_spacetime_conflict"
            return "dynamic_brake", detail
        self.reference = reference
        self._reference_selection = (target,speed)
        return "reference", detail

    def _event(self, allowed_success=True):
        s, c = self.state, self.config
        pose = (s.x, s.y, s.yaw)
        if (self.collision_history or getattr(self.plant, "collision_history", False)
            or (not getattr(self.world, "sensor_collision_truth", False) and self.world.collision(pose, c.vehicle))):
            self.collision_history = True
            return "collision"
        if self.world.out_of_bounds(pose, c.vehicle):
            return "out_of_bounds"
        arrived = math.hypot(s.x - self.goal[0], s.y - self.goal[1]) <= c.goal_tolerance
        parked = not c.parking or (
            abs(s.v) <= c.goal_speed
            and abs(wrap(s.yaw - self.goal[2])) <= c.goal_yaw_tolerance
        )
        if (
            allowed_success
            and arrived
            and parked
            and self.ever_controlled
            and self.episode["distance"] > 0.1
        ):
            return "success"
        if s.time >= c.deadline - 1e-9:
            return "deadline"
        return None

    def step(self, action):
        decision_started = time.perf_counter()
        if self.done:
            raise RuntimeError("reset is required only after the episode ends")
        c = self.config
        ep = self.episode
        dt = c.control_dt
        decision_initial_speed = self.state.v
        flat = c.method == "B1"
        if flat:
            clipped = flat_action_values(action,c)
            requested = Command(
                float(
                    clipped[0]
                    * (c.vehicle.max_accel if clipped[0] >= 0 else c.vehicle.max_decel)
                ),
                float(clipped[1] * c.vehicle.max_steer),
            )
            if c.flat_longitudinal_mode == 'speed_target_v1':
                # A disclosed 1/s proportional speed servo, no route lookup,
                # planner, reference trajectory, MPC or goal-distance rule.
                # The policy still chooses speed and steering every flat hold.
                target_speed = (float(clipped[0])+1.)*.5*c.vehicle.max_speed
                requested = Command(float(np.clip(target_speed-self.state.v,
                    -c.vehicle.max_decel,c.vehicle.max_accel)),requested.steering)
            self.last_behavior = np.array([*clipped, 0.0, 0.0])
            mode, plan = "flat", None
            self.reference = None
        else:
            if not self.action_space.contains(int(action)):
                raise ValueError("invalid discrete action")
            if (self.config.subgoal_mode.startswith("goal_frame") or self.config.subgoal_mode == "route_frame_terminal") and int(action):
                target, speed = self.decode_action(int(action))
                dx, dy = target[0]-self.state.x, target[1]-self.state.y
                co, si = math.cos(self.state.yaw), math.sin(self.state.yaw)
                self.last_behavior = np.array([co*dx+si*dy, -si*dx+co*dy, wrap(target[2]-self.state.yaw), speed])
            else:
                self.last_behavior = np.array(ACTIONS[int(action)])
            mode, plan = self._prepare_reference(int(action))
        ep["actions"] += 1
        self.total_actions += 1
        high_reward = 0.0
        executed = 0
        event = None
        trace = []
        for j in range(c.substeps):
            cycle_started = time.perf_counter()
            before = self.state
            intervention = False
            if flat:
                command = requested
                status = "direct"
                diagnostics = {}
                self.ever_controlled = True
            elif mode == "reference":
                control = self.controller.step(
                    before, self.reference, self.plant.previous_command
                )
                command, status, diagnostics = (
                    control.command,
                    control.status,
                    control.diagnostics,
                )
                ep["control_times"].append(diagnostics["control_seconds"])
                ep["qp_times"].append(diagnostics["qp_seconds"])
                if status != "solved":
                    ep["mpc_failures"] += 1
                    intervention = True
                else:
                    self.ever_controlled = True
            else:
                command = brake(before, c.vehicle)
                status = mode
                diagnostics = {}
                intervention = mode not in ("wait", "flat", "parking_brake")
            requested_command = command
            shield_intervention = False
            if c.shared_shield:
                from .shield import risk_brake
                command, shield_intervention = risk_brake(before, command, self.world, c.vehicle)
                intervention |= shield_intervention
                if shield_intervention:
                    status = "shared_shield_brake"
            if intervention:
                ep["interventions"] += 1
            # Check start and swept intermediate poses; no position clipping or endpoint snap.
            start_collision = not getattr(self.world, "sensor_collision_truth", False) and self.world.collision(
                (before.x, before.y, before.yaw), c.vehicle
            )
            after = self.plant.step(command, dt)
            swept_collision = False
            swept_bounds = False
            for fraction in (0.25, 0.5, 0.75, 1.0):
                pose = (
                    before.x + fraction * (after.x - before.x),
                    before.y + fraction * (after.y - before.y),
                    before.yaw + fraction * wrap(after.yaw - before.yaw),
                )
                if not getattr(self.world, "sensor_collision_truth", False):
                    swept_collision |= self.world.collision(pose, c.vehicle, dt * fraction)
                swept_bounds |= self.world.out_of_bounds(pose, c.vehicle)
            self.world.advance(dt)
            self.collision_history |= bool(start_collision or swept_collision)
            ep["distance"] += math.hypot(after.x - before.x, after.y - before.y)
            event = self._event(allowed_success=not intervention)
            if event != "collision" and swept_bounds:
                event = "out_of_bounds"
            if event and hasattr(self.plant, "confirm_terminal"):
                self.plant.confirm_terminal()
                if self.plant.collision_history:
                    self.collision_history = True
                    event = "collision"
            reward, parts, physical = self.reward_calculator.calculate(
                before,
                after,
                self.goal,
                self.world.risk(after, c.vehicle),
                event,
                event is not None,
            )
            high_reward += c.gamma_low**j * reward
            ep["discounted_return"] += c.gamma_low ** ep["physics_steps"] * reward
            ep["reward"] += reward
            for key, value in parts.items():
                ep["reward_components"][key] = (
                    ep["reward_components"].get(key, 0.0) + value
                )
            for key in ("jerk", "lateral_acceleration", "risk"):
                ep[key].append(physical[key])
            error = tracking_error(after, self.reference)
            for key, name in (
                ("lateral", "lateral_errors"),
                ("heading", "heading_errors"),
                ("speed", "speed_errors"),
            ):
                if error[key] is not None:
                    ep[name].append(error[key])
            ep["physics_steps"] += 1
            self.total_physics_steps += 1
            executed += 1
            trace.append(
                {
                    "state": asdict(after),
                    "command": asdict(self.plant.previous_command),
                    "requested_command": asdict(requested_command),
                    "shared_shield_intervention": shield_intervention,
                    "backend_frame": getattr(self.plant, "frame", None),
                    "backend_evidence": getattr(self.plant, "evidence", None),
                    "controller_status": status,
                    "diagnostics": diagnostics,
                    "reward": reward,
                    "reward_components": parts,
                    "physical": physical,
                    "tracking": error,
                    "intervention": intervention,
                }
            )
            cycle_seconds = time.perf_counter() - cycle_started
            trace[-1]["control_cycle_seconds"] = cycle_seconds
            ep["control_cycle_times"].append(cycle_seconds)
            if event:
                break
        self.done = event is not None
        if c.execution_feedback != "off":
            self.execution_outcome = np.array([float(mode == "reference"), float(mode == "plan_failed"),
                float(mode == "dynamic_brake"), sum(t["controller_status"] == "failed" for t in trace)/max(1,executed),
                sum(t["intervention"] for t in trace)/max(1,executed),
                (self.state.v-decision_initial_speed)/c.vehicle.max_speed], dtype=np.float32)
        info = {
            "success": event == "success",
            "termination_reason": event,
            "executed_substeps": executed,
            "sim_seconds": executed * dt,
            "plan": plan,
            "mode": mode,
            "trace": trace,
            "physics_steps": self.total_physics_steps,
        }
        info["initial_scenario"] = self.initial_scenario
        if self.rule_diagnostics is not None:
            info["rule_candidate_compute"] = self.rule_diagnostics
            self.rule_diagnostics = None
        observation = self._observation()
        if c.execution_feedback != "off":
            info["execution_feedback_seconds"] = self.feedback_seconds
        info["environment_decision_seconds"] = time.perf_counter() - decision_started
        ep["decision_cycle_times"].append(info["environment_decision_seconds"])
        if self.done:
            info["episode_metrics"] = self.summary(event)
        return observation, float(high_reward), self.done, False, info

    def summary(self, event):
        ep = self.episode
        s = self.state
        remaining = math.hypot(s.x - self.goal[0], s.y - self.goal[1])
        summary = {
            key: value for key, value in ep.items() if not isinstance(value, list)
        }
        summary.update(
            {
                "success": event == "success",
                "termination_reason": event,
                "scenario": self.scenario_name,
                "simulation_seconds": s.time,
                "progress": self.start_distance - remaining,
                "path_efficiency_success_only": (
                    max(0.0, self.start_distance - self.config.goal_tolerance)
                    / ep["distance"]
                    if event == "success" and ep["distance"] > 0
                    else None
                ),
                "travel_time_success_only": s.time if event == "success" else None,
                "intervention_rate": ep["interventions"] / max(1, ep["physics_steps"]),
                "plan_failure_rate": (
                    ep["plan_failures"] / len(ep["plan_times"])
                    if ep["plan_times"]
                    else None
                ),
                "mpc_failure_rate": (
                    ep["mpc_failures"] / len(ep["control_times"])
                    if ep["control_times"]
                    else None
                ),
                "tracking_reference_coverage": len(ep["lateral_errors"])
                / max(1, ep["physics_steps"]),
                "tracking_missing_reason": (
                    None
                    if ep["lateral_errors"]
                    else (
                        "flat_policy_has_no_reference"
                        if self.config.method == "B1"
                        else "no_valid_reference"
                    )
                ),
            }
        )
        for key in (
            "lateral_errors",
            "heading_errors",
            "speed_errors",
            "jerk",
            "lateral_acceleration",
            "risk",
        ):
            summary[key + "_rms"] = (
                float(np.sqrt(np.mean(np.square(ep[key])))) if ep[key] else None
            )
        from .metrics import timing_summary

        for key in ("plan_times", "control_times", "qp_times", "control_cycle_times", "decision_cycle_times"):
            summary[key] = timing_summary(
                ep[key],
                (
                    self.config.control_dt * self.config.substeps
                    if key in ("plan_times", "decision_cycle_times")
                    else self.config.control_dt
                ),
            )
        return summary


def rule_action(env):
    """B3 only: goal-directed candidate scoring, no learned-policy claims."""
    s = env.state
    if (
        math.hypot(s.x - env.goal[0], s.y - env.goal[1])
        < env.config.goal_tolerance * 0.8
    ):
        return 0
    best = (math.inf, 0)
    local_mask=None
    if env.config.local_candidate_filter or env.config.endpoint_candidate_filter:
        if getattr(env,'_local_candidate_mask_state',None) is not env.state:
            env._observation()
        local_mask=env._local_candidate_mask
    if env.config.subgoal_mode.startswith("goal_frame") or env.config.subgoal_mode == "route_frame_terminal":
        from .planning import analytic_connection
        # Strong deterministic comparator: same connector feasibility checks as
        # the planner, no learned action substitution. All cost is charged.
        cached = {}
        continuation = {}
        candidates = []
        for i in range(1, 31):
            if local_mask is not None and not local_mask[i]:
                continue
            target, speed = env.decode_action(i)
            if target not in continuation:
                continuation[target] = (candidate_goal_witness(target,env.goal,env.world,env.config.vehicle)
                    if env.config.execution_feedback.startswith('continuation') else True)
            # Finite preference, not a feasibility mask: if every witness is
            # absent the original ordering and fallback search remain available.
            penalty = 0. if continuation[target] else 5.
            candidates.append((math.dist(target[:2],env.goal[:2])-.1*speed+penalty,i,target,speed))
            if env.config.reference_reuse and env.reusable_reference(target,speed):
                reference=env.reference;keep=reference.start_time+reference.t>=s.time
                points=np.column_stack((reference.x[keep],reference.y[keep]))
                length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
                score=math.dist(target[:2],env.goal[:2])+.05*length-.1*speed+penalty
                if score<best[0]:best=(score,i)
                # Reuse is a verified candidate path, so fallback search for
                # this same target is unnecessary. No selected-action override.
                cached[target]=np.column_stack((reference.x[keep],reference.y[keep],reference.yaw[keep]))
                continue
            if target not in cached:
                cached[target] = analytic_connection((s.x,s.y,s.yaw), target, env.world, env.config.vehicle)
            path = cached[target]
            if path is None:
                continue
            length = np.linalg.norm(np.diff(path[:, :2],axis=0),axis=1).sum()
            score = math.dist(target[:2],env.goal[:2]) + .05*length - .1*speed + penalty
            if score < best[0]:
                best = (score, i)
        # A rejected analytic connector does not prove infeasibility. The strong
        # rule comparator also searches up to three distinct promising targets.
        # PPO never calls this rule; its own selected target gets its own search.
        searched = set()
        searches = []
        for optimistic, i, target, speed in sorted(candidates):
            if optimistic >= best[0] or target in searched or cached[target] is not None:
                continue
            # A blocked route-frame endpoint is rejected by Hybrid A* before
            # expansion. Do not spend one of the three fallback searches on it
            # while a collision-free endpoint remains unexamined.
            if env.config.subgoal_mode=='route_frame_terminal' and not env.world.free(target,env.config.vehicle):
                continue
            searched.add(target)
            plan = env.planner.plan((s.x,s.y,s.yaw),target,env.world)
            searches.append(plan)
            if plan.status == "success" and len(plan.path)>1:
                length = np.linalg.norm(np.diff(plan.path[:,:2],axis=0),axis=1).sum()
                score = optimistic+.05*length
                if score < best[0]:
                    best = (score,i)
                    env._rule_plan_cache = (s,target,plan)
            if len(searched) >= 3:
                break
        env.rule_diagnostics = {"search_calls":len(searches), "search_nodes":sum(p.expanded_nodes for p in searches),
                                "search_seconds":sum(p.planning_time for p in searches)}
        return best[1]
    for i, (_, _, _, speed) in enumerate(ACTIONS[1:], 1):
        goal, _ = env.decode_action(i)
        if not env.world.free(goal, env.config.vehicle, 0.1):
            continue
        # Cheap arc check for the rule baseline. The selected action still uses real planner/MPC.
        x, y, yaw, _ = ACTIONS[i]
        length = math.hypot(x, y)
        steer = math.atan(env.config.vehicle.wheelbase * yaw / max(length, 1e-5))
        samples = primitive(
            (s.x, s.y, s.yaw), steer, length, env.config.vehicle.wheelbase, 0.3
        )
        if not all(env.world.free(p, env.config.vehicle, 0.1) for p in samples):
            continue
        score = (
            math.hypot(goal[0] - env.goal[0], goal[1] - env.goal[1])
            + 2 * abs(wrap(goal[2] - env.goal[2]))
            - 0.1 * speed
        )
        if score < best[0]:
            best = (score, i)
    return best[1]
