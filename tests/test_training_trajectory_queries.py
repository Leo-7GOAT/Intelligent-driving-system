import pytest

from scripts.query_native_training_stops import trajectory_indices


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
