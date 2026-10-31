from copy import deepcopy
from pathlib import Path
import json
import pytest
from scripts.reproduce_public_feedback_cohort import validate_recipe


def recipe():
    return json.loads((Path(__file__).resolve().parents[1]/'protocols/public_native_feedback_cohort_20261031.json').read_text())


def test_exact_all_five_public_development_recipe_is_well_formed():
    validate_recipe(recipe())


@pytest.mark.parametrize('change',['missing_capsule','duplicate_batch','foreign_host','missing_seed','unsigned_hash'])
def test_reconstruction_does_not_reduce_inventory_or_load_untrusted_capsules(change):
    r=deepcopy(recipe())
    if change=='missing_capsule':r['capsules'].pop()
    elif change=='duplicate_batch':r['capsules'][1]['batch']=r['capsules'][0]['batch']
    elif change=='foreign_host':r['capsules'][0]['manifest_url']=r['capsules'][0]['manifest_url'].replace('github.com','example.com')
    elif change=='missing_seed':r['seeds'].pop()
    else:r['capsules'][0]['sha256']='untrusted'
    with pytest.raises(ValueError):validate_recipe(r)
