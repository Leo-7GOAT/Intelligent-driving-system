import numpy as np
from navstack.world import World
from navstack.config import VehicleConfig


def test_vectorized_envelope_matches_independent_scalar_sat():
    rng = np.random.default_rng(821)
    world = World(34,24,[(10,8,12,16),(20,12,22,20)])
    poses = np.column_stack((rng.uniform(-2,36,3000),rng.uniform(-2,26,3000),rng.uniform(-np.pi,np.pi,3000)))
    for vehicle in (VehicleConfig(), VehicleConfig(front=3.81,rear=.98,width=2.16,wheelbase=3)):
        for margin in (0,.1):
            scalar = np.array([world.free(p,vehicle,margin) for p in poses])
            np.testing.assert_array_equal(world.free_many(poses,vehicle,margin),scalar)
    # Exact touching with an obstacle is collision.
    touching = np.array([[10-3.4,12,0],[20-3.4,15,0]])
    assert not world.free_many(touching,VehicleConfig()).any()
