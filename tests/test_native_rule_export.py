"""Synthetic archive integrity/privacy checks, not actual CARLA evidence."""
import json
import zipfile
import pytest
from navstack.experiments import write_json
from scripts.audit_native_rule import audit
from scripts.package_native_rule_evidence import package
from test_native_rule_audit import fixture


def batch(path):
    path.mkdir();fixture(path)
    write_json(path/'guard_final.json',{'exit_code':0,'reason':None})
    write_json(path/'infrastructure/server_stop.json',{'status':'owned_empty_server_stopped'})
    write_json(path/'raw_audit.json',audit(path))
    return path


def test_empty_failed_probe_bytes_are_preserved_and_labelled_without_fabricating_json(tmp_path):
    source=batch(tmp_path/'batch');(source/'failed_probe.json').write_bytes(b'')
    output=tmp_path/'evidence.zip';package([source],output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('batch/failed_probe.json')==b''
        manifest=json.loads(archive.read('manifest.json'))
        assert [r['file'] for r in manifest['invalid_auxiliary_json_preserved']]==['batch/failed_probe.json']
        assert manifest['audits']['batch']['termination_counts']=={'deadline':1}
    with pytest.raises(FileExistsError):package([source],output)


def test_nonempty_invalid_metadata_is_rejected_instead_of_bypassing_structured_privacy_scan(tmp_path):
    source=batch(tmp_path/'batch');(source/'failed_probe.json').write_bytes(b'{"unfinished":')
    output=tmp_path/'evidence.zip'
    with pytest.raises(json.JSONDecodeError):package([source],output)
    assert not output.exists()
