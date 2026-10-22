import numpy as np
from navstack.config import VehicleConfig, PlannerConfig
from navstack.planning import HybridAStar
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
