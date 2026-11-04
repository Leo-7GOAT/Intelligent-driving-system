from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from navstack.config import PlannerConfig,VehicleConfig
from navstack.planning import GridAStarPlanner
from navstack.spatial_planner_transfer import PROFILE,settings
from navstack.steering_transition import settings as steering_settings,prepare_reference
from navstack.world import World


def protocol():
    return dict(methods=['H'],evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0},
        recovery={'H':dict(model_sha256='a'*64,normalization_sha256='b'*64)},
        reference_spatial_planner=deepcopy(PROFILE),
        reference_steering_transition=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000),
        reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        reference_road_refinement=dict(version='sampled_native_reference_refinement_v1',max_attempts=3),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'))


@pytest.mark.parametrize('change',[
    {'evaluation_only':False},{'controlled_budget_per_method':True},{'controlled_budget_per_method':1},
    {'method_budgets':{'H':False}},{'methods':['B2']},{'recovery':{}},
    {'reference_joint_speed':None},{'reference_compute_profile':None},
    {'reference_steering_transition':None},
    {'reference_spatial_planner':{**PROFILE,'max_nodes':32000.}},
])
def test_grid_transfer_rejects_learning_implicit_schema_or_incomplete_lowlevel(change):
    with pytest.raises(ValueError):steering_settings({**protocol(),**change})


def test_grid_adapter_executes_grid_and_keeps_shared_measured_steering_retimer(monkeypatch):
    import navstack.steering_transition as module
    def forbidden(*args,**kwargs):raise AssertionError('Grid transfer silently constructed Hybrid A*')
    monkeypatch.setattr(module,'HybridAStar',forbidden)
    p=protocol();assert settings(p)==PROFILE
    options=steering_settings(p);owner=object();retimer=object()
    env=SimpleNamespace(planner=owner,config=SimpleNamespace(vehicle=VehicleConfig(),planner=PlannerConfig()),
        state=SimpleNamespace(v=0.,steering=.1),decode_action=lambda a:((10.,10.,0.),1.),
        _reference_retimer=retimer,_rule_plan_cache='stale')
    def prepare(action):
        result=env.planner.plan((6.,10.,0.),(10.,10.,0.),World(25,25))
        assert callable(env._reference_retimer) and env._rule_plan_cache is None
        assert result.status=='success' and len(result.path)>2
        return 'reference',dict(status=result.status,expanded_nodes=result.expanded_nodes)
    mode,detail=prepare_reference(env,2,prepare,options)
    assert mode=='reference' and env.planner is owner and env._reference_retimer is retimer
    info=detail['steering_transition']
    assert info['spatial_planner']==PROFILE and not info['exact_failure_memo'] and not info['exact_success_cache']
    assert not info['measured_steering_in_spatial_search'] and info['measured_steering_in_shared_retimer']
    assert len(detail['joint_speed_search']['attempts'])==1


def test_grid_budget_failure_does_not_retry_identical_grid_search(monkeypatch):
    from navstack.planning import PlanResult
    calls=[]
    def fail(self,*args,**kwargs):
        calls.append((self.config.max_nodes,self.config.grid_postprocess))
        return PlanResult('timeout',np.empty((0,3)),None,32000,.1,'search_budget')
    monkeypatch.setattr(GridAStarPlanner,'plan',fail)
    owner=object();env=SimpleNamespace(planner=owner,config=SimpleNamespace(vehicle=VehicleConfig(),planner=PlannerConfig()),
        state=SimpleNamespace(v=1.8,steering=0.),decode_action=lambda a:((10,10,0),1.5))
    def prepare(action):
        result=env.planner.plan((6,10,0),(10,10,0),World(25,25))
        return 'plan_failed',dict(status=result.status)
    _,detail=prepare_reference(env,2,prepare,steering_settings(protocol()))
    assert calls==[(32000,'body_spline_v1')]
    assert detail['joint_speed_search']['total_expanded_nodes']==32000 and env.planner is owner


def test_grid_adapter_restores_owner_and_retimer_after_exception():
    owner=object();retimer=object()
    env=SimpleNamespace(planner=owner,config=SimpleNamespace(vehicle=VehicleConfig(),planner=PlannerConfig()),
        state=SimpleNamespace(v=0.,steering=0.),decode_action=lambda a:((10,10,0),1.),_reference_retimer=retimer)
    def fail(action):raise RuntimeError('native road query failed')
    with pytest.raises(RuntimeError):prepare_reference(env,1,fail,steering_settings(protocol()))
    assert env.planner is owner and env._reference_retimer is retimer


def test_overlay_verifier_keeps_hybrid_and_all_unrelated_code(tmp_path):
    from scripts.prepare_shared_native_grid import unchanged_ast
    old=tmp_path/'old.py';new=tmp_path/'new.py'
    old.write_text('class GridAStarPlanner:\n def plan(self): return 1\nclass HybridAStar:\n def plan(self): return 2\n')
    new.write_text('class GridAStarPlanner:\n def plan(self): return 3\nclass HybridAStar:\n def plan(self): return 2\n')
    assert unchanged_ast(old,{'GridAStarPlanner'})==unchanged_ast(new,{'GridAStarPlanner'})
    new.write_text(new.read_text().replace('return 2','return 4'))
    assert unchanged_ast(old,{'GridAStarPlanner'})!=unchanged_ast(new,{'GridAStarPlanner'})
