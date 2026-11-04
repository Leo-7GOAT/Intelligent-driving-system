import base64
import json
import zipfile
import io
import numpy as np

import pytest
import scripts.package_native_continuation as exporter


def test_replay_npz_is_exported_and_scanned_after_decompression():
    assert '.npz' in exporter.SUFFIXES
    safe=io.BytesIO();np.savez_compressed(safe,observations=np.zeros((2,3)),actions=np.array([1,2]))
    exporter.scan_payload('own_success_inputs.npz',safe.getvalue())
    private=io.BytesIO();np.savez_compressed(private,metadata=np.array('ghp_'+'a'*36))
    with pytest.raises(ValueError):exporter.scan_payload('own_success_inputs.npz',private.getvalue())


def test_auxiliary_path_redaction_keeps_embedded_json_metrics():
    # Assemble a synthetic home path without embedding an actual personal path.
    home='C:'+'/'+'Users'+'/'+'fixture'+'/'
    original={'command':'"'+home+'python.exe" task.py',
              'stdout':json.dumps({'path':home+'source','exit_code':0,'successes':0})}
    changes=[]
    public=exporter.portable_metadata(original,changes)
    assert len(changes)==2 and all('original_value_sha256' in row for row in changes)
    assert public['command'].startswith('origin_metadata/')
    embedded=json.loads(public['stdout'])
    assert embedded['exit_code']==0 and embedded['successes']==0
    exporter.scan_payload('auxiliary.json',json.dumps(public).encode())
    encoded={':serialized:':base64.b64encode(home.encode()).decode()}
    with pytest.raises(ValueError,match='Encoded private'):
        exporter.scan_payload('data',json.dumps({'policy':encoded}).encode())


@pytest.fixture
def complete_segment(tmp_path,monkeypatch):
    """Synthetic export fixture, not actual learner/vehicle evidence."""
    batch=tmp_path/'segment'
    for name in ('H0','B1','source','calibration','infrastructure'):
        (batch/name).mkdir(parents=True)
    def put(name,value):
        (batch/name).write_text(json.dumps(value),encoding='utf-8')
    put('protocol.json',{'methods':['H0','B1'],'source_hashes':{}})
    put('summary.json',{'status':'complete','methods':[{'method':'H0'},{'method':'B1'}]})
    put('guard_final.json',{'exit_code':0,'reason':None})
    put('infrastructure/server_stop.json',{'status':'owned_empty_server_stopped'})
    for method in ('H0','B1'):
        put(method+'/endpoint_raw_audit.json',{'total_applied_controls_verified':10})
        put(method+'/evaluation.json',{'episodes':[{'success':False,'termination_reason':'collision'}]})
    monkeypatch.setattr(exporter,'audit',lambda batch,method:{'total_applied_controls_verified':10})
    return batch,put


def test_export_preserves_failure_bytes_and_never_overwrites(complete_segment,tmp_path):
    batch,put=complete_segment
    output=tmp_path/'asset.zip'
    result=exporter.package(batch,output)
    assert result['verified_controls']=={'H0':10,'B1':10}
    with zipfile.ZipFile(output) as archive:
        assert archive.read('segment/B1/evaluation.json')==(batch/'B1/evaluation.json').read_bytes()
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['methods']==['H0','B1']
        assert not manifest['metadata_transformations']
    with pytest.raises(FileExistsError): exporter.package(batch,output)
    put('summary.json',{'status':'complete','methods':[{'method':'H0'}]})
    with pytest.raises(ValueError,match='every registered endpoint'):
        exporter.package(batch,tmp_path/'selected.zip')
    assert not (tmp_path/'selected.zip').exists()


def test_hash_bound_metadata_cannot_be_silently_redacted(complete_segment,tmp_path):
    batch,put=complete_segment
    put('H0/config.json',{'path':'C:'+'/'+'Users'+'/'+'fixture'+'/'+'config'})
    with pytest.raises(ValueError,match='hash-bound'):
        exporter.package(batch,tmp_path/'invalid.zip')
    assert not (tmp_path/'invalid.zip').exists()
    assert (tmp_path/'invalid.zip.partial').exists()


def test_supervised_dataset_bytes_are_exported_and_tampering_is_rejected(complete_segment,tmp_path):
    batch,put=complete_segment
    dataset=batch/'measured_demonstrations.npz'
    np.savez_compressed(dataset,observations=np.zeros((2,3)),actions=np.zeros((2,2)))
    protocol=json.loads((batch/'protocol.json').read_text())
    protocol['extra_supervision']={'dataset_sha256':exporter.sha(dataset.read_bytes())}
    put('protocol.json',protocol)
    output=tmp_path/'supervised.zip'
    exporter.package(batch,output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('segment/measured_demonstrations.npz')==dataset.read_bytes()
    dataset.write_bytes(b'changed data')
    with pytest.raises(ValueError,match='dataset'):
        exporter.package(batch,tmp_path/'changed.zip')
    assert not (tmp_path/'changed.zip').exists()
