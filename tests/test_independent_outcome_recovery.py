from copy import deepcopy
from navstack.fixed_lowlevel_training import registered


def test_checkpoint_ancestry_preserves_interrupted_tail_and_cumulative_adam(tmp_path):
    import json
    from scripts.run_independent_outcome_cohort import checkpoint_ancestry
    parent=tmp_path/'recovered_training';checkpoint=parent/'H/checkpoint_final';checkpoint.mkdir(parents=True)
    (parent/'protocol.json').write_text(json.dumps({'recovery':{'H':dict(retained_optimizer_steps=400,unknown_unflushed_parent_tail=True)}}))
    assert checkpoint_ancestry(checkpoint,{'optimizer_steps':570},tmp_path/'cohort')==dict(
        parent_batch='recovered_training',retained_optimizer_steps=970,unknown_unflushed_parent_tail=True)
    fresh=tmp_path/'cohort/initializers/seed22/H'
    assert checkpoint_ancestry(fresh,{'optimizer_steps':0},tmp_path/'cohort')==dict(
        parent_batch='cohort',retained_optimizer_steps=0,unknown_unflushed_parent_tail=False)


def test_saved_cohort_budget_admission_is_exact_and_does_not_change_original_gate():
    from test_fixed_lowlevel_training import protocol
    from navstack.fixed_lowlevel_training import REGISTRATION_COHORT
    p=protocol();p.update(fixed_lowlevel_training=deepcopy(REGISTRATION_COHORT),controlled_budget_per_method=30720,
        method_budgets={'H':30720},reference_joint_speed=dict(version='measured_steering_rate_retiming_v2',spatial_speed_ceiling=.25),
        reference_point_constraints=dict(version='native_counterexample_point_tube_v1'),reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        mpc_reference_effort=dict(version='reference_centered_mpc_effort_v1'),reference_context=dict(version='action_aligned_outcome_context_v1'),
        outcome_context_registration=dict(version='action_aligned_outcome_context_v1',failure_gain_scale=1000.,parameter_states=34,new_supervision=0,failure_feedback_ablated=False))
    assert registered(p)
    p.update(controlled_budget_per_method=18060,method_budgets={'H':18060},recovery={'H':{'lineage_controls':12660}},
        fixed_cohort_recovery=dict(version='saved_outcome_cohort_recovery_v1',original_target=30720,saved_controls=12660,
            parent_protocol_sha256='a'*64,interruption_audit_sha256='b'*64))
    assert registered(p)
    for patch in ({'controlled_budget_per_method':30720},{'method_budgets':{'H':18061}},
                  {'recovery':{'H':{'lineage_controls':12659}}},{'fixed_cohort_recovery':None},
                  {'fixed_cohort_recovery':{**p['fixed_cohort_recovery'],'saved_controls':12660.}}):
        assert not registered({**p,**patch})
