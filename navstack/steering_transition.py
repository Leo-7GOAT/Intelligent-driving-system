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
    allowed=protocol.get('methods') in (['H'],['H0'])
    if not allowed:
        from .shared_rule_registration import registered as rule_registered
        allowed=rule_registered(protocol)
    if (not valid or not (protocol.get('evaluation_only') is True or registered(protocol))
            or not allowed):
        raise ValueError('Steering transition prototype requires frozen H/H0 evaluation')
    from .reference_compute import enabled
    result=dict(value,cache_success=True) if enabled(protocol) else value
    from .steering_retiming import settings as joint_settings
    joint=joint_settings(protocol)
    return dict(result,joint_speed=joint) if joint else result


def prepare_reference(env,action,prepare,options):
    if options is None:return prepare(action)
    owner=env.planner
    target,speed=env.decode_action(action) if int(action) else (None,0.)
    ceiling=max(.1,env.state.v,speed)
    initial=env.state.steering
    budget=options.get('max_nodes',env.config.planner.max_nodes)
    backend=HybridAStar(env.config.vehicle,replace(env.config.planner,max_nodes=budget))
    memo=options['version']==VERSION_BUDGET
    cache_success=options.get('cache_success',False)
    if memo:
        from .planner_memo import ExactSteeringFailureMemo
        if (not hasattr(env,'_steering_failure_memo')
                or env._steering_failure_memo.cache_success!=cache_success):
            env._steering_failure_memo=ExactSteeringFailureMemo(backend,cache_success=cache_success)
        env._steering_failure_memo.planner=backend
        backend=env._steering_failure_memo
    joint=options.get('joint_speed');attempts=[]
    class Adapter:
        def plan(self,start,goal,world):
            result=backend.plan(start,goal,world,initial_steering=initial,reference_speed=ceiling)
            attempts.append(dict(speed_ceiling=ceiling,status=result.status,expanded_nodes=result.expanded_nodes,
                planning_time=result.planning_time,cache_hit=getattr(result,'cache_hit',False)))
            if joint and result.status in ('no_path','timeout') and ceiling>joint['spatial_speed_ceiling']:
                result=backend.plan(start,goal,world,initial_steering=initial,reference_speed=joint['spatial_speed_ceiling'])
                attempts.append(dict(speed_ceiling=joint['spatial_speed_ceiling'],status=result.status,
                    expanded_nodes=result.expanded_nodes,planning_time=result.planning_time,
                    cache_hit=getattr(result,'cache_hit',False)))
                result=replace(result,planning_time=sum(a['planning_time'] for a in attempts))
            return result
    env.planner=Adapter()
    previous_retimer=getattr(env,'_reference_retimer',None)
    if joint:
        from .steering_retiming import retime
        env._reference_retimer=lambda reference:retime(reference,env.config.vehicle,initial)
    # A pose-only precomputed plan cannot certify a measured steering transition.
    env._rule_plan_cache=None
    try:mode,detail=prepare(action)
    finally:
        env.planner=owner
        if joint:env._reference_retimer=previous_retimer
    if detail is not None:
        detail=dict(detail)
        detail['steering_transition']=dict(version=options['version'],initial_steering=initial,speed_ceiling=ceiling,
            analytic_connections=False,exact_failure_memo=memo,max_nodes=budget)
        if cache_success:detail['steering_transition']['exact_success_cache']=True
        if joint:
            detail['steering_transition']['joint_speed']=joint
            detail['joint_speed_search']=dict(attempts=attempts,
                total_expanded_nodes=sum(a['expanded_nodes'] for a in attempts))
    return mode,detail
