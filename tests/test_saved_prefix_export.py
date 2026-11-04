import json
import zipfile

import pytest

import scripts.package_native_saved_prefix as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path, monkeypatch):
    batch = tmp_path / 'original'
    write(batch / 'protocol.json', dict(source_hashes={}))
    write(batch / 'verified_saved_prefix.json', dict(saved_controls=11520))
    write(batch / 'interruption_audit.json', dict(interrupted=True))
    write(batch / 'auxiliary.json', dict(origin='C:/Users/Example/private/path'))
    (batch / 'infrastructure').mkdir()
    (batch / 'B1').mkdir()
    raw = b'exact original opaque control bytes'
    (batch / 'B1/model.zip').write_bytes(b'PK')
    (batch / 'B1/optimization.jsonl').write_bytes(raw)
    monkeypatch.setattr(module, 'audit_prefix', lambda path: dict(saved_controls=11520))
    monkeypatch.setattr(module, 'operational_guard_files', lambda path: set())
    # Model format is outside these export-unit tests; the real audit is separate.
    scan = module.scan_payload
    monkeypatch.setattr(module, 'scan_payload', lambda name, payload: None if name.endswith('model.zip') else scan(name, payload))
    return batch, raw


def test_prefix_export_preserves_raw_bytes_and_marks_no_endpoint(tmp_path, monkeypatch):
    batch, raw = fixture(tmp_path, monkeypatch)
    out = tmp_path / 'prefix.zip'
    result = module.package(batch, out)
    with zipfile.ZipFile(out) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert not manifest['batch_complete'] and manifest['endpoint_results'] == 0
        assert archive.read('original/B1/optimization.jsonl') == raw
        assert archive.read('original/protocol.json') == (batch / 'protocol.json').read_bytes()
        transformed = json.loads(archive.read('original/auxiliary.json'))
        assert transformed['origin'].startswith('origin_metadata/')
        assert 'auxiliary.json' in manifest['metadata_transformations']
    assert result['saved_controls'] == 11520 and result['endpoint_results'] == 0


def test_hash_bound_private_metadata_is_rejected_without_rewriting_source(tmp_path, monkeypatch):
    batch, _ = fixture(tmp_path, monkeypatch)
    write(batch / 'protocol.json', dict(source_hashes={}, private_path='C:/Users/Example/path'))
    before = (batch / 'protocol.json').read_bytes()
    out = tmp_path / 'prefix.zip'
    with pytest.raises(ValueError, match='hash-bound'):
        module.package(batch, out)
    assert not out.exists() and (batch / 'protocol.json').read_bytes() == before


def test_existing_prefix_archive_is_never_overwritten(tmp_path, monkeypatch):
    out = tmp_path / 'existing.zip'
    out.write_bytes(b'original archive')
    monkeypatch.setattr(module, 'audit_prefix', lambda path: pytest.fail('No audit or export on overwrite attempt'))
    with pytest.raises(FileExistsError):
        module.package(tmp_path / 'not_needed', out)
    assert out.read_bytes() == b'original archive'


@pytest.mark.parametrize('fault', ['complete', 'evaluation_present', 'unrelated_error'])
def test_saved_prefix_audit_rejects_completed_or_unrelated_jobs(tmp_path, fault):
    write(tmp_path / 'protocol.json', dict(methods=['B1']))
    write(tmp_path / 'guard_final.json', dict(exit_code=0 if fault == 'complete' else 1, reason='disk_below_12GiB'))
    write(tmp_path / 'failure.json', dict(error="RuntimeError('unrelated')" if fault == 'unrelated_error' else "RuntimeError('Disk below12GiB')"))
    write(tmp_path / 'infrastructure/server_stop.json', dict(status='owned_empty_server_stopped'))
    if fault == 'evaluation_present': write(tmp_path / 'B1/evaluation.json', {})
    with pytest.raises(ValueError, match='TRAIN-only'):
        module.audit_prefix(tmp_path)
