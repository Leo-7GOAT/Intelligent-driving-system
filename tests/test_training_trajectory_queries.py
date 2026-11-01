import pytest

from scripts.query_native_training_stops import trajectory_indices,reserve_training_episode


def test_late_training_departure_is_covered_without_success_or_failure_selection():
    indices=trajectory_indices(615)
    assert indices[0]==0 and indices[-1]==614
    assert max(b-a for a,b in zip(indices,indices[1:]))<=80
    assert all(0<=index<615 for index in indices)


@pytest.mark.parametrize('length',[1,80,81,1200])
def test_short_and_budget_boundary_episodes_have_unique_in_range_queries(length):
    indices=trajectory_indices(length)
    assert indices==sorted(set(indices)) and indices[-1]==length-1
    assert len(indices)<=16


@pytest.mark.parametrize('length',[0,-1,True,1.5])
def test_no_empty_or_ambiguous_trajectory_length(length):
    with pytest.raises(ValueError):trajectory_indices(length)


def test_three_episode_profile_covers_early_and_late_states_with_fixed_budget():
    indices=trajectory_indices(1200,40)
    assert indices==list(range(0,1200,40))+[1199]
    assert len(indices)==31
    counts={};selected=[]
    # Completion outcome has no input to reservation; each condition is bounded.
    for episode,condition in enumerate([2,1,2,1,2,1,2,1]):
        if reserve_training_episode(counts,condition,3):selected.append(episode)
    assert selected==[0,1,2,3,4,5] and counts=={1:3,2:3}


@pytest.mark.parametrize('stride',[0,True,20,40.0])
def test_unregistered_stride_is_rejected(stride):
    with pytest.raises(ValueError):trajectory_indices(100,stride)
