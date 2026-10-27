import numpy as np
import pytest

from scripts.prepare_native_measured_bc import shifted_holdout_mask


def test_initial_covered_split_keeps_each_case_separate_and_original_immutable():
    groups=np.array([0]*7+[1]*10)
    original=np.array([False,True,True,True,True,False,True]+[False,True,True,True,True]*2)
    before=original.copy()
    unchanged,_=shifted_holdout_mask(groups,original,0)
    np.testing.assert_array_equal(unchanged,original)
    changed,cases=shifted_holdout_mask(groups,original,4)
    np.testing.assert_array_equal(original,before)
    assert np.flatnonzero(~changed).tolist()==[4,11,16]
    assert changed[0] and changed[7]
    assert [(c['train'],c['holdout'],c['initial_is_training']) for c in cases]==[(6,1,True),(8,2,True)]
    changed[0]=False
    assert not np.shares_memory(changed,original)


@pytest.mark.parametrize('phase',[True,1,3,5,None])
def test_unregistered_split_phase_is_rejected(phase):
    with pytest.raises(ValueError):
        shifted_holdout_mask(np.zeros(5,dtype=int),np.array([False,True,True,True,True]),phase)


def test_a_changed_base_mask_cannot_be_relabelled_as_the_original_split():
    with pytest.raises(ValueError):
        shifted_holdout_mask(np.zeros(5,dtype=int),np.array([True,False,True,True,True]),4)
