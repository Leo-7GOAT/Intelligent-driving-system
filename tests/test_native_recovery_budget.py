from scripts.prepare_native_recovery import recovery_targets


def test_additional_course_recovery_keeps_prior_lineage_and_only_finishes_remaining_controls():
    p=dict(methods=['H0'],controlled_budget_per_method=10240,method_budgets={'H0':10240},
           recovery={'H0':{'lineage_controls':20539}})
    assert recovery_targets(p,True)=={'H0':30779}
    assert recovery_targets(p,True)['H0']-(20539+7913)==2327
    assert recovery_targets(p)=={'H0':10240}


def test_explicit_lineage_target_survives_another_interruption():
    p=dict(methods=['H0'],controlled_budget_per_method=10240,method_budgets={'H0':2327},
           recovery={'H0':{'lineage_controls':28452}},lineage_targets={'H0':30779})
    assert recovery_targets(p,True)==recovery_targets(p)=={'H0':30779}
