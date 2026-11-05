from copy import deepcopy

import pytest


def test_worker_overlay_changes_only_one_explicit_admission_block():
    from scripts.prepare_contextual_outcome_pilot import worker_overlay, ANCHOR, VALIDATION_BLOCK
    source='before\n'+ANCHOR+'\nafter\n'
    assert worker_overlay(source)==source.replace(ANCHOR,VALIDATION_BLOCK+ANCHOR)
    for invalid in ('',source+source,worker_overlay(source)):
        with pytest.raises(ValueError):worker_overlay(invalid)


def test_endpoint_uses_locked_separated_tasks_and_cannot_change_their_digest():
    from scripts.prepare_contextual_outcome_pilot import canonical_digest
    from scripts.run_fixed_lowlevel_adaptation import endpoint_task_profile
    tasks=dict(routes=[{'training_base':True}],native_case_manifest={'base':True},
               native_validation=dict(manifest=dict(cases=[{'case':i} for i in range(24)])))
    p=dict(contextual_feedback_pilot=dict(validation_tasks=tasks,
        validation_tasks_sha256=canonical_digest(tasks),validation_case_ids=list(range(24))))
    result=endpoint_task_profile(p)
    assert result==tasks and result is not tasks
    changed=deepcopy(p);changed['contextual_feedback_pilot']['validation_tasks']['native_validation']['manifest']['cases'].pop()
    with pytest.raises(ValueError):endpoint_task_profile(changed)
    assert endpoint_task_profile({}) is None


def test_flat_dependency_requires_every_seed_and_distinct_phase():
    from scripts.run_contextual_outcome_pilot import dependency_inventory
    phases=('zero_PPO_basic','TRAIN48_PPO30720','basic','known24')
    p=dict(seeds=[11,22,33,44,55],results=[dict(seed=s,phase=phase) for s in [11,22,33,44,55] for phase in phases])
    assert len(dependency_inventory(p))==20
    p['results'][-1]=p['results'][0]
    with pytest.raises(ValueError):dependency_inventory(p)


def test_dependency_identity_uses_resolved_output_argument_and_numeric_tasks_are_stable(tmp_path):
    from scripts.prepare_contextual_outcome_pilot import identifies_dependency,canonical_digest
    command=['python','run_matched_flat_cohort.py','run','--output',str(tmp_path)]
    assert identifies_dependency(command,tmp_path)
    assert not identifies_dependency(command+['--output',str(tmp_path)],tmp_path)
    assert not identifies_dependency(['python','other.py','--output',str(tmp_path)],tmp_path)
    assert canonical_digest({'x':[1.,0.]})==canonical_digest({'x':[1,0]})
    assert canonical_digest({'x':1.001})!=canonical_digest({'x':1.002})
