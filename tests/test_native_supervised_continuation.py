import json
import pytest
from scripts.continue_native_learning import copy_registered_teacher,verified_parent_audit,sha


def test_registered_supervision_survives_continuation_and_tampering_fails(tmp_path):
    import numpy as np
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    data=parent/'measured_demonstrations.npz';np.savez_compressed(data,actions=np.array([[.1,.2]]))
    protocol={'extra_supervision':{'dataset_sha256':sha(data)}}
    copy_registered_teacher(parent,child,protocol)
    assert (child/data.name).read_bytes()==data.read_bytes()
    data.write_bytes(b'changed')
    with pytest.raises(ValueError):copy_registered_teacher(parent,child,protocol)
    data.unlink()
    with pytest.raises(ValueError):copy_registered_teacher(parent,child,protocol)


def test_current_per_method_audit_supports_continuation_without_legacy_plot(tmp_path,monkeypatch):
    import scripts.audit_native_method_endpoint as verifier
    folder=tmp_path/'B1';folder.mkdir()
    raw={'training_control_frames_verified':20480,'evaluation_control_frames_verified':7857}
    (folder/'endpoint_raw_audit.json').write_text(json.dumps(raw))
    calls=[]
    def actual_audit(batch,method):calls.append((batch,method));return dict(raw)
    monkeypatch.setattr(verifier,'audit',actual_audit)
    rebuilt,path=verified_parent_audit(tmp_path,['B1'])
    assert calls==[(tmp_path,'B1')]
    assert rebuilt['methods'][0]['evaluation']['control_frames_verified']==7857
    assert json.loads(path.read_text())==rebuilt
    raw['evaluation_control_frames_verified']=0
    with pytest.raises(ValueError,match='changed'):verified_parent_audit(tmp_path,['B1'])


def test_partial_current_audits_cannot_fall_back_to_older_batch_audit(tmp_path):
    (tmp_path/'H').mkdir();(tmp_path/'H/endpoint_raw_audit.json').write_text('{}')
    with pytest.raises(ValueError,match='Partial'):verified_parent_audit(tmp_path,['H','H0'])
