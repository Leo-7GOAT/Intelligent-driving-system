import json
from types import SimpleNamespace
from scripts import run_fixed_lowlevel_adaptation as course
from scripts.native_road_learning import configure_continuation_normalization


def test_same_condition_endpoint_removes_training_only_normalization_gate(tmp_path,monkeypatch):
    parent=tmp_path/'training';parent.mkdir();(parent/'H/checkpoint').mkdir(parents=True)
    (parent/'source').mkdir();(parent/'calibration').mkdir()
    def write(name,value):(parent/name).write_text(json.dumps(value),encoding='utf-8')
    write('protocol.json',dict(freeze_observation_statistics=True,fixed_lowlevel_training={'version':'training'},
        lineage_targets={'H':5120},methods=['H'],method_budgets={'H':5120},controlled_budget_per_method=5120,
        resource_limits={'wall_seconds':900},recovery={'H':{'retained_optimizer_steps':10}}))
    write('H/summary.json',{'optimizer_steps':170});write('H/latest_checkpoint.json',{'folder':'checkpoint'})
    monkeypatch.setattr(course,'require_finished_parent',lambda p:None)
    monkeypatch.setattr(course,'verified_checkpoint_metadata',lambda p:dict(lineage_control_steps=5396,
        charged_control_steps_lower_bound=6000,model_sha256='model',normalization_sha256='norm'))
    target=tmp_path/'endpoint';course.endpoint(parent,target)
    p=json.loads((target/'protocol.json').read_text(encoding='utf-8'))
    assert p['evaluation_only'] is True and p['method_budgets']=={'H':0}
    assert 'freeze_observation_statistics' not in p and 'fixed_lowlevel_training' not in p
    configure_continuation_normalization(SimpleNamespace(training=True),p,p['recovery']['H'])
