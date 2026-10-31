import pytest
from scripts.run_native_shared_brake_cohort import b1_parent_map


def state(seeds=(11,22,33,44,55),phase='ALL_FIVE_MATURED_B1_PPO_SEEDS_COMPLETE'):
    return {'phase':phase,'created':[{'seed':s,'batch':'new_'+str(s)} for s in seeds]}


def test_new_version_uses_all_five_new_policies_including_seed11():
    assert b1_parent_map(state(),'all_five_matured_v1','old_11')=={s:'new_'+str(s) for s in (11,22,33,44,55)}


def test_original_version_keeps_its_explicit_independent_seed11():
    old=state((22,33,44,55),'B3_AND_FOUR_INDEPENDENT_B1_PPO_SEEDS_COMPLETE')
    assert b1_parent_map(old,'original_four_plus_seed11','old_11')[11]=='old_11'


@pytest.mark.parametrize('bad',[state((22,33,44,55)),state((11,22,33,44,44)),state(phase='WAITING_FOR_REGISTERED_B2_COHORT')])
def test_partial_duplicate_or_unfinished_cohorts_cannot_activate_safety_probes(bad):
    with pytest.raises(ValueError,match='incomplete'):
        b1_parent_map(bad,'all_five_matured_v1')


def test_unregistered_batch_path_is_rejected():
    bad=state();bad['created'][0]['batch']='../foreign'
    with pytest.raises(ValueError,match='direct-child'):
        b1_parent_map(bad,'all_five_matured_v1')
