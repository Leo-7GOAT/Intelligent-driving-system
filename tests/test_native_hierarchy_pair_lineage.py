import json

import pytest

from scripts.analyze_native_hierarchy_pair import training_record


def frozen_evaluation(tmp_path):
    origin=tmp_path/'parent'/'H';checkpoint=origin/'checkpoint';checkpoint.mkdir(parents=True)
    batch=tmp_path/'evaluation';(batch/'H').mkdir(parents=True)
    for p in (batch/'H/model.zip',checkpoint/'model.zip'):p.write_bytes(b'trained-weights')
    (batch/'protocol.json').write_text(json.dumps({'evaluation_only':True,
        'recovery':{'H':{'parent_batch':'parent','parent_checkpoint':'checkpoint'}}}))
    (batch/'H/summary.json').write_text(json.dumps({'physics_steps':0,'optimizer_steps':0}))
    (origin/'summary.json').write_text(json.dumps({'physics_steps':1280,'optimizer_steps':40,
        'charged_control_steps_lower_bound':2000,'lineage_control_steps':1280}))
    (checkpoint/'metadata.json').write_text(json.dumps({'physics_steps':1280,'optimizer_steps':40}))
    return batch,checkpoint


def test_frozen_segment_records_actual_ancestor_learning_cost(tmp_path):
    batch,_=frozen_evaluation(tmp_path)
    result=training_record(batch,'H')
    assert result['physical_controls']==1280 and result['optimizer_steps']==40
    assert result['charged_controls_lower_bound']==2000


def test_reject_changed_frozen_policy(tmp_path):
    batch,_=frozen_evaluation(tmp_path);(batch/'H/model.zip').write_bytes(b'other-policy')
    with pytest.raises(ValueError,match='differs from original'):training_record(batch,'H')


def test_reject_stale_training_counters(tmp_path):
    batch,checkpoint=frozen_evaluation(tmp_path)
    (checkpoint/'metadata.json').write_text(json.dumps({'physics_steps':1200,'optimizer_steps':39}))
    with pytest.raises(ValueError,match='counters differ'):training_record(batch,'H')
