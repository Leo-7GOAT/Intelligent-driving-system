import gzip
import hashlib
import json

import pytest

from scripts.audit_native_policy_inference import recorded_scenes


def save_scene_stream(tmp_path, scene):
    folder=tmp_path/'H0';folder.mkdir()
    infrastructure=tmp_path/'infrastructure';infrastructure.mkdir()
    path=infrastructure/'bridge_method_responses.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        stream.write(json.dumps({'response':{'scene':scene}})+'\n')
    audit={'bridge_response_stream':path.name,
           'bridge_response_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder/'endpoint_raw_audit.json').write_text(json.dumps(audit))
    return folder,path


def scene(digest='validation'):
    return {'width':80,'height':80,'road_grid':{'sha256':digest},'native_layout':{'path':[[0,0],[10,0]]}}


def test_measured_validation_reset_available_without_training_scene_file(tmp_path):
    folder,_=save_scene_stream(tmp_path,scene())
    # A different method's incomplete stream must not contaminate this audit.
    (tmp_path/'infrastructure/other_responses.jsonl.gz').write_bytes(b'truncated')
    assert recorded_scenes(tmp_path,folder)=={'validation':scene()}


def test_reject_changed_measured_scene_stream(tmp_path):
    folder,path=save_scene_stream(tmp_path,scene())
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='differs from raw endpoint audit'):
        recorded_scenes(tmp_path,folder)


def test_reject_same_road_hash_with_conflicting_layout(tmp_path):
    folder,_=save_scene_stream(tmp_path,scene())
    other=scene();other['native_layout']['path']=[[0,0],[0,10]]
    (folder/'scene_0001.json').write_text(json.dumps({'scene':other}))
    with pytest.raises(ValueError,match='Conflicting measured scene'):
        recorded_scenes(tmp_path,folder)
