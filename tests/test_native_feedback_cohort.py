import pytest
from scripts.analyze_native_feedback_cohort import paired_seed_statistics,resolve_batch


def test_equal_seed_pairs_do_not_create_evidence_from_repeated_cases():
    result = paired_seed_statistics([0, 0, 0, 0, 0])
    assert result['independent_training_seeds'] == 5
    assert result['paired_95pct_t_interval'] == [0, 0]
    assert result['exploratory_exact_sign_flip_p_two_sided'] == 1


def test_exact_five_seed_sign_test_keeps_its_discrete_two_sided_resolution():
    result = paired_seed_statistics([.2]*5)
    assert result['paired_mean_difference'] == pytest.approx(.2)
    assert result['exploratory_exact_sign_flip_p_two_sided'] == pytest.approx(2/32)


def test_actual_mixed_seed_effect_cannot_support_positive_interval_lower_bound():
    result = paired_seed_statistics([1/12, 1/12, -2/12, 1/12, 0])
    assert result['paired_mean_difference'] == pytest.approx(1/60)
    assert result['paired_95pct_t_interval'][0] < 0 < result['paired_95pct_t_interval'][1]


@pytest.mark.parametrize('values', [[.1]*4, [float('nan')]*5, [1.1]*5, [[0]*5]*5])
def test_rejects_insufficient_or_invalid_independent_seed_effects(values):
    with pytest.raises(ValueError):
        paired_seed_statistics(values)


def test_restored_public_inputs_can_live_outside_the_source_checkout(tmp_path):
    assert resolve_batch(tmp_path,'runs/registered_batch')==tmp_path/'runs/registered_batch'


@pytest.mark.parametrize('path',['../foreign','/absolute/batch','C:/foreign/batch','C:relative'])
def test_cohort_recipe_cannot_escape_its_declared_restored_input_root(tmp_path,path):
    with pytest.raises(ValueError):resolve_batch(tmp_path,path)
