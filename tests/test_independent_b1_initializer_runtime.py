import pytest

from scripts.prepare_native_approach_pair import verify_initializer_runtime


def sources(tmp_path):
    roots=[tmp_path/'initial',tmp_path/'runtime']
    names=('candidate_policy','road_candidate_policy','independent_clip_ppo','config','env',
           'measured_demo','imitation','carla_actuation','world','road_grid')
    for root in roots:
        (root/'navstack').mkdir(parents=True)
        for name in names:(root/'navstack'/f'{name}.py').write_text('B1_behavior = 1\n')
    return roots


def test_h0_only_additions_do_not_invalidate_identical_b1_code(tmp_path):
    initial,runtime=sources(tmp_path)
    (initial/'navstack/road_candidate_policy.py').write_text('B1_behavior = 1\nclass SharedEndpointAblatedRoadPolicy:\n    pass\n')
    assert verify_initializer_runtime(initial,runtime)['critical_files']==10


@pytest.mark.parametrize('name',['road_candidate_policy','env','carla_actuation'])
def test_changed_b1_policy_or_physics_never_passes_runtime_comparison(tmp_path,name):
    initial,runtime=sources(tmp_path)
    (initial/'navstack'/f'{name}.py').write_text('B1_behavior = 2\n')
    with pytest.raises(ValueError):verify_initializer_runtime(initial,runtime)
