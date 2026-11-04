from copy import deepcopy
import pytest
from navstack.reference_compute import VERSION,enabled
from navstack.reference_road_refinement import settings as road_settings
from navstack.steering_transition import settings as steering_settings


def protocol():
    return dict(methods=['H'],evaluation_only=True,
        reference_compute_profile={'version':VERSION},
        reference_road_refinement=dict(version='sampled_native_reference_refinement_v1',max_attempts=3),
        reference_steering_transition=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000))


def test_compute_profile_does_not_mutate_registration_and_defaults_remain_off():
    p=protocol();old=deepcopy(p)
    assert enabled(p) and road_settings(p)['stop_repeated_references'] and steering_settings(p)['cache_success']
    assert p==old
    p.pop('reference_compute_profile')
    assert not enabled(p)
    assert 'stop_repeated_references' not in road_settings(p) and 'cache_success' not in steering_settings(p)


@pytest.mark.parametrize('key,value',[
    ('reference_compute_profile',{'version':'unknown'}),('methods',['B1']),
    ('reference_road_refinement',None),('reference_steering_transition',None)])
def test_unregistered_compute_profile_is_rejected(key,value):
    p=protocol();p[key]=value
    with pytest.raises(ValueError):enabled(p)
