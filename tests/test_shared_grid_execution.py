from copy import deepcopy
import pytest
from scripts.run_shared_native_grid import PHASES,verify_dependency_inventory


def complete():
    return dict(seeds=[11,22,33,44,55],results=[dict(seed=seed,phase=phase) for seed in (11,22,33,44,55) for phase in PHASES])


def test_grid_execution_requires_all_distinct_registered_flat_phases():
    result=complete();assert len(verify_dependency_inventory(result))==20
    for index in range(20):
        missing=deepcopy(result);missing['results'].pop(index)
        with pytest.raises(ValueError):verify_dependency_inventory(missing)
        duplicate=deepcopy(result);duplicate['results'][index]=duplicate['results'][(index+1)%20]
        with pytest.raises(ValueError):verify_dependency_inventory(duplicate)
    with pytest.raises(ValueError):verify_dependency_inventory({**result,'seeds':[11,22,33,44]})
