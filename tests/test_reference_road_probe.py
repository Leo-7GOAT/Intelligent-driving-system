import numpy as np
import math
from scripts.probe_reference_road37 import footprint_points


def test_nominal_points_use_native_rear_axle_and_left_handed_map_coordinates():
    v=dict(rear=1.,front=4.,width=2.)
    points=footprint_points([7.,5.,0.],v,dict(angle=0.,origin=[0.,0.],spawn_z=.6))
    assert len(points)==12
    np.testing.assert_allclose(points[0],[6.,-4.,0.]);np.testing.assert_allclose(points[-1],[11.,-6.,0.])
    rotated=footprint_points([7.,5.,0.],v,dict(angle=90.,origin=[100.,200.],spawn_z=.7))
    np.testing.assert_allclose(rotated[0],[104.,206.,.1]);np.testing.assert_allclose(rotated[-1],[106.,211.,.1])


def test_vehicle_heading_rotates_footprint_before_map_reflection():
    points=footprint_points([7.,5.,math.pi/2],dict(rear=1.,front=4.,width=2.),dict(angle=0.,origin=[0.,0.],spawn_z=.6))
    np.testing.assert_allclose(points[0],[8.,-4.,0.]);np.testing.assert_allclose(points[-1],[6.,-9.,0.])
