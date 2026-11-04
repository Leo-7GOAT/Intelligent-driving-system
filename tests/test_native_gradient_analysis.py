import json

import pytest

import scripts.analyze_native_gradient_pair as analyzer


@pytest.fixture
def paired_segment(tmp_path,monkeypatch):
    names=['H0_global','H0_independent']
    protocol={'gradient_clipping_experiment':True,'methods':names,
        'jobs':[{'name':name,'method':'H0','gradient_clip_mode':mode} for name,mode in
                zip(names,['global','independent_actor_critic_v1'])],
        'method_budgets':dict.fromkeys(names,20000),'recovery':{}}
    for name in names:
        (tmp_path/name).mkdir()
        (tmp_path/name/'config.json').write_text('{}')
        checkpoint=tmp_path/'resume'/name
        checkpoint.mkdir(parents=True)
        (checkpoint/'model.zip').write_bytes(b'same synthetic parent, never loaded')
        protocol['recovery'][name]={'checkpoint':'resume/'+name,'lineage_controls':100,
            'consumed_controls_lower_bound':110,'retained_optimizer_steps':12,'unknown_unflushed_parent_tail':True}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    def endpoint(batch,name):
        return {'cases':[{'case_id':0,'success':False,'termination_reason':'out_of_bounds',
                         'progress':3. if name.endswith('global') else 1.,'intervention_rate':0.}]}
    monkeypatch.setattr(analyzer,'checked_endpoint',endpoint)
    monkeypatch.setattr(analyzer,'initial_cases',lambda folder:{0:{'road_sha256':'road-a','goal':[8,6,0],
        'static':[],'dynamic':[],'initial':{'x':6,'y':6,'yaw':0,'v':0}}})
    return tmp_path


def test_comparison_keeps_negative_outcomes_and_paired_direction(paired_segment):
    result=analyzer.compare(paired_segment,'H0')
    assert result['mean_progress_difference_m']==-2
    assert result['success_difference_count']==0
    assert all(arm['evaluation_summary']['termination_counts']=={'out_of_bounds':1}
               for arm in result['arms'].values())
    assert result['paired_cases'][0]['measured_start_position_difference_m']==0


def test_different_parent_weights_cannot_be_called_a_clipping_pair(paired_segment):
    (paired_segment/'resume/H0_independent/model.zip').write_bytes(b'another model')
    with pytest.raises(ValueError,match='identical saved learning state'):
        analyzer.compare(paired_segment,'H0')


def test_different_roads_cannot_be_paired_by_case_number(paired_segment,monkeypatch):
    original=analyzer.initial_cases
    def changed(folder):
        result=original(folder)
        if folder.name.endswith('independent'):
            result[0]['road_sha256']='different-road'
        return result
    monkeypatch.setattr(analyzer,'initial_cases',changed)
    with pytest.raises(ValueError,match='road/task identity'):
        analyzer.compare(paired_segment,'H0')
