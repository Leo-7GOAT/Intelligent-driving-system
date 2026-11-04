"""Explicit frozen-policy transfer of a steering-slew-aware planning model."""
from .planning import HybridAStar

VERSION='measured_steering_slew_primitives_v1'


def settings(protocol):
    value=protocol.get('reference_steering_transition')
    if value is None:return None
    if (value!={'version':VERSION} or not protocol.get('evaluation_only')
            or protocol.get('methods') not in (['H'],['H0'])):
        raise ValueError('Steering transition prototype requires frozen H/H0 evaluation')
    return value


def prepare_reference(env,action,prepare,options):
    if options is None:return prepare(action)
    owner=env.planner
    target,speed=env.decode_action(action) if int(action) else (None,0.)
    ceiling=max(.1,env.state.v,speed)
    initial=env.state.steering
    backend=HybridAStar(env.config.vehicle,env.config.planner)
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
        detail['steering_transition']=dict(version=VERSION,initial_steering=initial,speed_ceiling=ceiling,
            analytic_connections=False,exact_failure_memo=False)
    return mode,detail
