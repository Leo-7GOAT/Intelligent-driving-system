from copy import deepcopy

import pytest

from scripts.diagnose_native_rule_endpoint import pair_tasks


def tasks():
    return {0: {'road_sha256': 'road', 'goal': [10, 0, 0], 'static': [], 'dynamic': [],
                'initial': {'x': 0., 'y': 0., 'yaw': 0., 'v': 0.}}}


def test_pairing_reports_measured_start_differences_without_hiding_them():
    a = tasks()
    b = deepcopy(a)
    b[0]['initial'].update(x=3., y=4., yaw=.2, v=1.)
    result = pair_tasks(a, b)[0]
    assert result['measured_start_position_difference_m'] == 5.
    assert result['measured_start_yaw_difference_rad'] == pytest.approx(.2)
    assert result['measured_start_speed_difference_mps'] == 1.


@pytest.mark.parametrize('mutation', ['missing', 'road', 'goal', 'obstacle'])
def test_pairing_rejects_missing_or_different_tasks(mutation):
    a = tasks()
    b = deepcopy(a)
    if mutation == 'missing':
        b.clear()
    elif mutation == 'road':
        b[0]['road_sha256'] = 'another-road'
    elif mutation == 'goal':
        b[0]['goal'][0] += 1
    else:
        b[0]['static'] = [[1., 2., 3., 4.]]
    with pytest.raises(ValueError, match='inventory|differs'):
        pair_tasks(a, b)
