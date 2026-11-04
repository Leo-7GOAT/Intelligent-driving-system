"""Explicit frozen-policy transfer of a steering-slew-aware planning model."""
from .planning import HybridAStar
from dataclasses import replace

VERSION='measured_steering_slew_primitives_v1'
VERSION_BUDGET='measured_steering_slew_32000_memo_v2'


def settings(protocol):
    value=protocol.get('reference_steering_transition')
    if value is None:return None
    valid=(value=={'version':VERSION} or isinstance(value,dict) and type(value.get('max_nodes')) is int
           and value=={'version':VERSION_BUDGET,'max_nodes':32000})
    from .fixed_lowlevel_training import registered
    if (not valid or not (protocol.get('evaluation_only') is True or registered(protocol))
            or protocol.get('methods') not in (['H'],['H0'])):
        raise ValueError('Steering transition prototype requires frozen H/H0 evaluation')
    return value


def prepare_reference(env,action,prepare,options):
    if options is None:return prepare(action)
    owner=env.planner
    target,speed=env.decode_action(action) if int(action) else (None,0.)
    ceiling=max(.1,env.state.v,speed)
    initial=env.state.steering
    budget=options.get('max_nodes',env.config.planner.max_nodes)
    backend=HybridAStar(env.config.vehicle,replace(env.config.planner,max_nodes=budget))
    memo=options['version']==VERSION_BUDGET
    if memo:
        from .planner_memo import ExactSteeringFailureMemo
        if not hasattr(env,'_steering_failure_memo'):env._steering_failure_memo=ExactSteeringFailureMemo(backend)
        env._steering_failure_memo.planner=backend
        backend=env._steering_failure_memo
    class Adapter:
        def plan(self,start,goal,world):
            return backend.plan(start,goal,world,initial_steering=initial,reference_speed=ceiling)
    env.planner=Adapter()
    # A pose-only precomputed plan cannot certify a measured steering transition.
    env._rule_plan_cache=None
    try:mode,detail=prepare(action)
    finally:env.planner=owner
    if detail is not None:
        detail=dict(detail)
        detail['steering_transition']=dict(version=options['version'],initial_steering=initial,speed_ceiling=ceiling,
            analytic_connections=False,exact_failure_memo=memo,max_nodes=budget)
    return mode,detail
