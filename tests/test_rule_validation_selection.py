import pytest

from scripts.evaluate_native_rule import prepare


@pytest.mark.parametrize('changes',[{'catalog':'route_frame_terminal'},
    {'planner_nodes':7200},{'continuation_feedback':True},{'speed_policy':'paired_low_speed'}])
def test_paired_validation_never_silently_changes_h_mechanisms(tmp_path,changes):
    with pytest.raises(ValueError,match='no simultaneous mechanism intervention'):
        prepare(tmp_path,'H',development_validation=True,**changes)
    assert not list(tmp_path.iterdir())


def test_validation_selection_rejects_ambiguous_switch(tmp_path):
    with pytest.raises(ValueError,match='explicit boolean'):
        prepare(tmp_path,'H',development_validation='yes')
