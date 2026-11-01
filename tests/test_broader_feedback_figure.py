from copy import deepcopy
import pytest
from scripts.plot_native_feedback_cohort import figure_labels


def broader():
    return {'cohort_kind':'matched_broader_36train_24val_v1','unique_condition_count':24,
        'seeds':[11,22,33,44,55],'results':[{'cases':24} for _ in range(5)]}


def test_broader_plot_uses_actual_condition_count_and_does_not_prejudge_effect():
    title,scope=figure_labels(broader())
    assert title=='All matched seed pairs' and '24 known' in scope
    assert 'not final confirmation' in scope and 'stable benefit is not established' not in scope


@pytest.mark.parametrize('change',['condition_count','seed_missing','case_missing','unknown_kind'])
def test_figure_cannot_hide_partial_or_misidentified_broader_cohort(change):
    cohort=deepcopy(broader())
    if change=='condition_count':cohort['unique_condition_count']=120
    if change=='seed_missing':cohort['seeds'].pop()
    if change=='case_missing':cohort['results'][2]['cases']=23
    if change=='unknown_kind':cohort['cohort_kind']='different'
    with pytest.raises(ValueError):figure_labels(cohort)


def test_existing_published_canonical_title_is_preserved_exactly():
    assert figure_labels({})==('All canonical seed pairs',
        'Feedback ablation: 12 reused development conditions, five trained seed pairs\nExploratory result; stable benefit is not established')
