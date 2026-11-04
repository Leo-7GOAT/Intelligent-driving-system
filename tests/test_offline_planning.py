import numpy as np
from navstack.config import VehicleConfig, PlannerConfig
from navstack.planning import HybridAStar,GridAStarPlanner,grid_search
from navstack.world import World


def test_offline_node_budget_reproducible():
    planner = HybridAStar(
        VehicleConfig(), PlannerConfig(max_seconds=None, max_nodes=1800)
    )
    world = World(40, 30, [(14, 13, 16, 17)])
    a = planner.plan((4, 15, 0), (26, 15, 0), world)
    b = planner.plan((4, 15, 0), (26, 15, 0), world)
    assert a.status == b.status == "success" and a.expanded_nodes == b.expanded_nodes
    np.testing.assert_array_equal(a.path, b.path)


def test_grid_node_budget_stops_at_exact_count_and_preserves_unlimited_path():
    expected=grid_search((0,0),(19,19),set(),20)
    assert expected[2]>3
    for _ in range(2):
        path,cost,count=grid_search((0,0),(19,19),set(),20,max_nodes=3)
        assert not path and np.isinf(cost) and count==3
    assert grid_search((0,0),(19,19),set(),20,max_nodes=expected[2])==expected


def test_grid_planner_reports_budget_before_spline_work(monkeypatch):
    def unexpected(*args,**kwargs):
        raise AssertionError('No postprocessing after the search budget expires')
    monkeypatch.setattr('navstack.grid_smoothing.body_spline',unexpected)
    planner=GridAStarPlanner(VehicleConfig(),PlannerConfig(max_nodes=2,max_seconds=None,
        grid_postprocess='body_spline_v1'))
    for _ in range(2):
        result=planner.plan((4.,10.,0.),(25.,10.,0.),World(32,24))
        assert result.status=='budget' and result.failure_reason=='node_budget'
        assert result.expanded_nodes==2 and result.path.shape==(0,3)
