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


def broader_recipe():
    r=recipe();r['cohort_kind']='matched_broader_36train_24val_v1';r['capsules']=r['capsules'][:5]
    for seed,row in zip(r['seeds'],r['capsules']):row.update(seed=seed,methods=['H','H0'])
    return r


def test_broader_recipe_requires_five_complete_paired_capsules_and_external_relative_references(tmp_path):
    validate_recipe(broader_recipe(),tmp_path)


@pytest.mark.parametrize('change',['sixth_capsule','incomplete_arm','wrong_seed','unknown_kind','escape_reference'])
def test_broader_inventory_cannot_reuse_the_legacy_six_capsule_or_twelve_condition_interpretation(change,tmp_path):
    r=broader_recipe()
    if change=='sixth_capsule':r['capsules'].append(deepcopy(r['capsules'][0]))
    if change=='incomplete_arm':r['capsules'][1]['methods']=['H']
    if change=='wrong_seed':r['capsules'][2]['seed']=22
    if change=='unknown_kind':r['cohort_kind']='unknown'
    if change=='escape_reference':r['expected_report']='../foreign.json'
    with pytest.raises(ValueError):validate_recipe(r,tmp_path)
