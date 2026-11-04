from pathlib import Path
import numpy as np
import pytest
from scripts.build_parallel_native_measured_views import merge_arrays


def sources(tmp_path):
    folders=[];cases=[]
    for case_id,size in enumerate((6,9)):
        folder=tmp_path/str(case_id);folder.mkdir();folders.append(folder)
        np.savez_compressed(folder/'arrays.npz',observations=np.full((size,3),case_id,dtype=np.float32),
            actions=np.full((size,2),case_id,dtype=np.float32),returns=np.arange(size,dtype=np.float32),
            train_mask=np.arange(size)%5!=0,groups=np.full(size,case_id,dtype=np.int64))
        cases.append(dict(case_id=case_id,group=case_id,controls=size,termination_reason=('success','collision')[case_id]))
    return folders,cases


def test_parallel_merge_preserves_failed_cases_order_dtype_and_per_case_holdout(tmp_path):
    folders,cases=sources(tmp_path);out=merge_arrays(folders,cases,15)
    np.testing.assert_array_equal(out['groups'],[0]*6+[1]*9)
    np.testing.assert_array_equal(out['train_mask'],np.r_[np.arange(6)%5!=0,np.arange(9)%5!=0])
    assert out['observations'].dtype==np.float32 and out['groups'].dtype==np.int64
    assert out['observations'].shape==(15,3)


def test_partial_reordered_or_duplicate_cases_cannot_become_a_complete_dataset(tmp_path):
    folders,cases=sources(tmp_path)
    with pytest.raises(ValueError):merge_arrays(folders,cases,16)
    with pytest.raises(ValueError):merge_arrays(folders,cases[::-1],15)
    with pytest.raises(ValueError):merge_arrays(folders[::-1],cases,15)
    cases[1]['case_id']=0
    with pytest.raises(ValueError):merge_arrays(folders,cases,15)
