from copy import deepcopy
import gzip
import json

import pytest


def test_inventory_requires_both_methods_and_each_frequency_and_brake_state():
    from scripts.prepare_current_native_deployment_controls import VARIANTS,variant_label
    from scripts.run_current_native_deployment_controls import job_inventory
    plan={'jobs':[dict(method=m,kind=k,value=v,label=variant_label(m,k,v)) for m in ('H','B1') for k,v in VARIANTS]}
    assert len(job_inventory(plan))==8
    bad=deepcopy(plan);bad['jobs'][-1]=bad['jobs'][0]
    with pytest.raises(ValueError):job_inventory(bad)
    for method,kind,value in [('H','cadence',True),('B1','shared_brake',1),('B2','cadence',2)]:
        with pytest.raises(ValueError):variant_label(method,kind,value)


def test_shared_brake_accounting_preserves_other_interventions_and_requires_every_frame(tmp_path):
    from scripts.run_current_native_deployment_controls import shared_brake_counts
    ticks=[dict(shared_shield_intervention=True,intervention=True,goal_parking_intervention=False),
           dict(shared_shield_intervention=False,intervention=True,goal_parking_intervention=True)]
    path=tmp_path/'trace.gz'
    def write():
        with gzip.open(path,'wt') as stream:stream.write(json.dumps({'info':{'executed_substeps':2,'trace':ticks}})+'\n')
    write();result=shared_brake_counts(path)
    assert result==dict(physics_controls=2,policy_decisions=1,common_brake_controls=1,all_intervention_controls=2,parking_assist_controls=1)
    ticks[0].pop('shared_shield_intervention');write()
    with pytest.raises(ValueError):shared_brake_counts(path)
