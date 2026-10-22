from dataclasses import asdict, dataclass, field
import math


@dataclass
class VehicleConfig:
    wheelbase: float = 2.5
    front: float = 3.4  # rear axle to front bumper, m
    rear: float = 0.8
    width: float = 1.8
    max_steer: float = 0.5  # rad; R_min = wheelbase / tan(max_steer)
    steer_rate: float = 0.6  # rad/s, applies to all architectures
    max_speed: float = 3.0
    max_accel: float = 1.5
    max_decel: float = 2.5
    steer_tau: float = 0.12
    input_delay_steps: int = 0
    wheelbase_scale: float = 1.0  # plant only: real parameter mismatch
    acceleration_scale: float = 1.0

    def __post_init__(self):
        positive = (
            self.wheelbase,
            self.front,
            self.rear,
            self.width,
            self.steer_rate,
            self.max_speed,
            self.max_accel,
            self.max_decel,
            self.wheelbase_scale,
            self.acceleration_scale,
        )
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError(
                "vehicle dimensions and limits must be positive and finite"
            )
        if not 0 < self.max_steer < math.pi / 2 or self.steer_tau < 0:
            raise ValueError("invalid steering limits/lag")
        if (
            not isinstance(self.input_delay_steps, int)
            or not 0 <= self.input_delay_steps <= 5
        ):
            raise ValueError("delay queue supports an integer 0..5 steps")

    @property
    def min_turn_radius(self):
        return self.wheelbase / math.tan(self.max_steer)


@dataclass
class PlannerConfig:
    xy_resolution: float = 0.4
    yaw_resolution: float = 0.15
    primitive_length: float = 0.8
    collision_step: float = 0.12
    goal_tolerance: float = 0.55
    yaw_tolerance: float = 0.20
    max_nodes: int = 1800
    max_seconds: float | None = (
        0.10  # None: deterministic offline node-budget experiment
    )
    steer_cost: float = 0.05
    steer_change_cost: float = 0.08
    grid_resolution: float = 0.5
    heuristic_weight: float = (
        1.5  # weighted Hybrid A*: bounded engineering search, not optimal
    )
    analytic_expansion: bool = False
    vectorized_primitives: bool = False
    grid_postprocess: str = "legacy"

    def __post_init__(self):
        if self.grid_postprocess not in ("legacy", "body_spline_v1"):
            raise ValueError("unknown grid postprocessor")


@dataclass
class RewardConfig:
    success: float = 20.0
    collision: float = -30.0
    out_of_bounds: float = -20.0
    deadline: float = -10.0
    time_weight: float = 0.1
    risk_weight: float = 0.4
    comfort_weight: float = 0.04
    beta: float = 1.0
    distance_scale: float = 40.0
    gamma_per_second: float = 0.99
    progress_weight: float = 0.0  # explicit task reward, not potential-based shaping
    heading_potential_weight: float = 0.0  # optional state-only terminal-pose shaping
    heading_potential_radius: float = 6.0  # metres, smooth proximity decay

    def __post_init__(self):
        if self.distance_scale <= 0 or not 0 < self.gamma_per_second <= 1:
            raise ValueError("invalid potential scale or physical-time discount")
        if not math.isfinite(self.progress_weight) or self.progress_weight < 0:
            raise ValueError("progress weight must be finite and nonnegative")
        if (not math.isfinite(self.heading_potential_weight) or self.heading_potential_weight < 0
                or not math.isfinite(self.heading_potential_radius) or self.heading_potential_radius <= 0):
            raise ValueError("invalid heading-potential weight/radius")


@dataclass
class EnvConfig:
    method: str = "H"
    control_dt: float = 0.05
    decision_dt: float = 0.5
    flat_hold: int = 1
    deadline: float = 30.0
    scenario: str = "mixed"
    split: str = "train"
    obstacle_num: int = 3
    width: float = 34.0
    height: float = 24.0
    goal_tolerance: float = 0.8
    goal_speed: float = 0.25
    goal_yaw_tolerance: float = 0.35
    parking: bool = True
    prediction_seconds: float = 3.0
    mpc_horizon: int = 20
    shared_shield: bool = False
    terminal_goal_prefix: bool = False
    curriculum: bool = False
    training_route_length: float | None = None
    training_route_range: list[float] | None = None
    subgoal_mode: str = "ego_arcs"
    execution_feedback: str = "off"
    candidate_observation: bool = False
    road_observation: bool = False  # full shared 64x64 measured occupancy, new model signature
    empty_pretraining_steps: int = 0
    route_stages: list[dict] = field(default_factory=list)
    route_rehearsal_probability: float = 0.0
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if self.method not in ("H", "B1", "B2", "B3"):
            raise ValueError("method must be H, B1, B2 or B3")
        if self.subgoal_mode not in ("ego_arcs", "goal_frame", "goal_frame_terminal"):
            raise ValueError("Unknown subgoal semantics")
        if self.execution_feedback not in ("off", "enabled", "ablated"):
            raise ValueError("Unknown execution feedback mode")
        if self.candidate_observation and self.execution_feedback == "off":
            raise ValueError("candidate observations require enabled or ablated feedback")
        if not isinstance(self.empty_pretraining_steps, int) or self.empty_pretraining_steps < 0:
            raise ValueError("empty pretraining budget must be a nonnegative integer")
        if self.empty_pretraining_steps and self.curriculum:
            raise ValueError("do not combine task-stage and route-length curricula")
        if self.route_stages:
            if self.curriculum or self.empty_pretraining_steps or self.training_route_length is not None or self.training_route_range is not None:
                raise ValueError("route stages cannot be combined with another curriculum")
            previous = 0
            for stage in self.route_stages:
                if (set(stage) != {"until_physics_steps", "route_m"}
                        or not isinstance(stage["until_physics_steps"], int)
                        or stage["until_physics_steps"] <= previous
                        or not math.isfinite(stage["route_m"])
                        or not 2 <= stage["route_m"] <= self.width-11):
                    raise ValueError("invalid route stage budget or distance")
                previous = stage["until_physics_steps"]
        if (not math.isfinite(self.route_rehearsal_probability)
                or not 0 <= self.route_rehearsal_probability < 1
                or (self.route_rehearsal_probability and not self.route_stages)):
            raise ValueError("route rehearsal requires route stages and probability in [0,1)")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        ratio = self.decision_dt / self.control_dt
        if self.control_dt <= 0 or not math.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError("decision_dt must be an integer multiple of control_dt")
        if self.flat_hold < 1 or not isinstance(self.flat_hold, int):
            raise ValueError("flat_hold must be a positive integer")
        if not 0 <= self.vehicle.input_delay_steps <= 5:
            raise ValueError(
                "input_delay_steps must be 0..5 (observation queue capacity)"
            )
        if self.split not in ("train", "validation", "test"):
            raise ValueError("invalid scenario split")
        if self.deadline <= 0 or self.vehicle.wheelbase_scale <= 0:
            raise ValueError("deadline and wheelbase_scale must be positive")
        if self.training_route_length is not None and not 2 <= self.training_route_length <= self.width - 11:
            raise ValueError("training route must fit the map and be at least 2 m")
        if self.training_route_range is not None:
            if (len(self.training_route_range) != 2 or
                    not 2 <= self.training_route_range[0] < self.training_route_range[1] <= self.width - 11):
                raise ValueError("training route range must be increasing and fit the map")
            if self.training_route_length is not None or self.curriculum:
                raise ValueError("choose one training route distribution")
        if not 0 <= self.obstacle_num <= 18 or self.mpc_horizon < 2:
            raise ValueError(
                "obstacle_num must fit static observation; horizon must be >=2"
            )

    @property
    def substeps(self):
        return (
            self.flat_hold
            if self.method == "B1"
            else round(self.decision_dt / self.control_dt)
        )

    @property
    def gamma_low(self):
        return self.reward.gamma_per_second**self.control_dt

    @property
    def gamma(self):
        return self.gamma_low**self.substeps

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key, typ in (
            ("vehicle", VehicleConfig),
            ("planner", PlannerConfig),
            ("reward", RewardConfig),
        ):
            if isinstance(data.get(key), dict):
                data[key] = typ(**data[key])
        return cls(**data)
