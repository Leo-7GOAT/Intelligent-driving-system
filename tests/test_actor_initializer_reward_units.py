import pytest

from scripts.prepare_native_approach_pair import verified_reward_units

DECLARATION={'version':'fixed_physical_reward_units_v1','scale':.05}


def normalizer(kind,**values):
    result=type(kind,(),{})();result.__dict__.update(norm_reward=False,**values)
    return result


def test_ordinary_checkpoint_cannot_inherit_donors_fixed_reward_label():
    with pytest.raises(ValueError,match='cannot claim'):
        verified_reward_units(normalizer('RoadCandidateVecNormalize'),DECLARATION)


def test_unscaled_and_fixed_checkpoint_units_are_kept_separate():
    assert verified_reward_units(normalizer('RoadCandidateVecNormalize'),None) is None
    result=verified_reward_units(normalizer('FixedRewardUnitsNormalize',fixed_reward_scale=.05),DECLARATION)
    assert result==DECLARATION and result is not DECLARATION


@pytest.mark.parametrize('scale',[None,1.,.5])
def test_mislabeled_saved_reward_scale_is_rejected(scale):
    with pytest.raises(ValueError,match='differ from the protocol'):
        verified_reward_units(normalizer('FixedRewardUnitsNormalize',fixed_reward_scale=scale),DECLARATION)
