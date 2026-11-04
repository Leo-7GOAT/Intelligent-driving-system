import math
import numpy as np
from scripts.preflight_native_static_placements import transformed_body


def test_body_transform_keeps_measurement_shape_and_rotates_about_actor_origin():
    body=[[11.,20.],[13.,20.],[13.,22.],[11.,22.]]
    original=[row[:] for row in body]
    transformed=np.array(transformed_body(body,[10.,20.,0.],[4.,5.,math.pi/2]))
    np.testing.assert_allclose(transformed,[[4.,6.],[4.,8.],[2.,8.],[2.,6.]])
    np.testing.assert_allclose(np.linalg.norm(transformed[:,None]-transformed[None,:],axis=2),
                               np.linalg.norm(np.array(body)[:,None]-np.array(body)[None,:],axis=2))
    assert body==original
