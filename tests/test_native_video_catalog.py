from copy import deepcopy
import json
import pytest
from scripts.catalog_native_ppo_videos import bind_video,catalog


def inputs():
    video={'case_id':1,'file':'case_01_native_curve_out_of_bounds.mp4','run_id':'H','seed':11,
        'model_sha256':'1'*64,'sha256':'2'*64,'case':{'scenario':'native_curve','route_index':3},
        'success':False,'termination_reason':'out_of_bounds'}
    episode={'case_id':1,'original_parent_case_id':3,'status':'completed','case':deepcopy(video['case']),
        'success':False,'termination_reason':'out_of_bounds'}
    return video,episode


def test_batch_qualified_identity_preserves_failed_original_case_and_legacy_manifest():
    video,episode=inputs();before=deepcopy((video,episode))
    result=bind_video('batch_20261101','H',11,[0,3],video,episode,'1'*64,'3'*64)
    assert result['run_id']=='batch_20261101/H' and result['source_video_manifest_run_id']=='H'
    assert result['original_parent_case_id']==3 and result['case_id_in_recording']==1
    assert result['video_id']=='batch_20261101:H:seed11:parent_case3'
    assert result['normalization_sha256']=='3'*64 and result['success'] is False
    assert (video,episode)==before
    other=bind_video('another_batch','H',11,[0,3],video,episode,'1'*64,'3'*64)
    assert other['video_id']!=result['video_id'] and other['release_asset_name']!=result['release_asset_name']


@pytest.mark.parametrize('bad',['model','seed','original_case','outcome','scene','file'])
def test_mixed_models_scenes_outcomes_or_unsafe_files_cannot_enter_catalog(bad):
    video,episode=inputs()
    if bad=='model':video['model_sha256']='4'*64
    if bad=='seed':video['seed']=22
    if bad=='original_case':episode['original_parent_case_id']=1
    if bad=='outcome':episode['success']=True
    if bad=='scene':episode['case']['route_index']=0
    if bad=='file':video['file']='../foreign.mp4'
    with pytest.raises(ValueError):bind_video('batch_20261101','H',11,[0,3],video,episode,'1'*64,'3'*64)


def test_pending_or_partial_recording_queue_cannot_produce_a_catalog(tmp_path):
    execution=tmp_path/'execution';execution.mkdir();output=tmp_path/'catalog.json'
    plan={'recordings':[{'batch':'one','method':'H'},{'batch':'two','method':'B1'}]}
    (execution/'plan.json').write_text(json.dumps(plan))
    state={'phase':'WAITING_FOR_OWNED_TRAINING_QUEUES_TO_EXIT','completed':[]}
    (execution/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='entire registered'):catalog(execution,output)
    state.update(phase='ALL_REGISTERED_NATIVE_PPO_RECORDINGS_AND_FRAME_VERIFICATIONS_COMPLETE',completed=[plan['recordings'][0]])
    (execution/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='entire registered'):catalog(execution,output)
    assert not output.exists()
