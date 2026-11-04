from copy import deepcopy
import pytest
from scripts.prepare_native_placement_inventory import eligible_proposals


def fixture():
    registration=dict(cases=[0,1],proposals=[{'distance_m':10.},{'distance_m':13.}])
    rows=[dict(case_id=c,proposal_index=i,spec=deepcopy(spec),eligible_model_proposal=i==0,
        status='success',hazard_fully_road_supported=True,start_free=True,goal_free=True,
        path_samples_free=True,timed_samples_free=True) for c in (0,1) for i,spec in enumerate(registration['proposals'])]
    return registration,dict(results=rows)


def test_selection_is_independent_of_policy_outcome_and_input_order():
    registration,complete=fixture();expected=eligible_proposals(registration,complete)
    changed=deepcopy(complete)
    for row in changed['results']:row['policy_success']=False
    changed['results'].reverse()
    assert [(r['case_id'],r['proposal_index']) for r in eligible_proposals(registration,changed)]==[(r['case_id'],r['proposal_index']) for r in expected]


@pytest.mark.parametrize('fault',['missing','duplicate','altered_spec','incomplete_witness','truthy_string'])
def test_rejects_unbound_or_partially_selected_proposals(fault):
    registration,complete=fixture();rows=complete['results']
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=deepcopy(rows[0])
    elif fault=='altered_spec':rows[0]['spec']={'distance_m':9.}
    elif fault=='incomplete_witness':rows[0]['timed_samples_free']=False
    else:rows[0]['eligible_model_proposal']='yes'
    with pytest.raises(ValueError):eligible_proposals(registration,complete)
