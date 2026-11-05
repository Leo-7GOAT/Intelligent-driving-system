import pytest
from scripts.build_native_delivery_comparison import five_seed_comparisons


def rows(seeds):
    return [dict(seed=s,method=m,successes=n,episodes=24)
        for s in seeds for m,n in [('H',23),('B1',7),('H0',24),('H_before_new_PPO',24)]]


def test_three_learners_do_not_become_72_independent_replications():
    assert five_seed_comparisons(rows([11,22,33]))=={}


def test_full_cohort_uses_five_seed_differences_and_retains_negative_ablation():
    result=five_seed_comparisons(rows([11,22,33,44,55]))
    assert result['H_minus_B1']['independent_training_seeds']==5
    assert result['H_minus_B1']['paired_mean_difference']==pytest.approx(16/24)
    assert result['H_minus_H0']['paired_mean_difference']==pytest.approx(-1/24)
    assert result['H_minus_H_before_new_PPO']['paired_mean_difference']==pytest.approx(-1/24)


def test_a_missing_control_seed_does_not_inherit_other_comparisons():
    data=[r for r in rows([11,22,33,44,55]) if not(r['seed']==55 and r['method']=='H0')]
    result=five_seed_comparisons(data)
    assert 'H_minus_B1' in result and 'H_minus_H0' not in result
