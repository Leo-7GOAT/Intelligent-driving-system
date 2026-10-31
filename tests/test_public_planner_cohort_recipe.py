from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.reproduce_public_planner_cohort import validate_recipe


def recipe():return json.loads((Path(__file__).resolve().parents[1]/'protocols/public_native_planner_cohort_20261101.json').read_text())


def test_complete_all_five_public_planner_recipe():validate_recipe(recipe())


@pytest.mark.parametrize('fault',['missing','duplicate_seed','foreign_host','unsigned_hash','missing_method','traversal'])
def test_public_planner_reconstruction_rejects_reduced_or_untrusted_inputs(fault):
    r=deepcopy(recipe())
    if fault=='missing':r['capsules'].pop()
    elif fault=='duplicate_seed':r['capsules'][0]['seed']=22
    elif fault=='foreign_host':r['capsules'][0]['manifest_url']=r['capsules'][0]['manifest_url'].replace('github.com','example.com')
    elif fault=='unsigned_hash':r['capsules'][0]['sha256']='0'
    elif fault=='missing_method':r['capsules'][0]['methods']=['B2']
    else:r['capsules'][0]['slot']='../outside'
    with pytest.raises(ValueError):validate_recipe(r)
