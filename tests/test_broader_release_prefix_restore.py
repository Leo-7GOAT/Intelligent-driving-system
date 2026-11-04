import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.reproduce_broader_b1_release import restore_prefix


def archive(tmp_path, name='prefix/protocol.json', *, corrupt=False, symlink=False, complete=False):
    payload = b'{}'
    manifest = dict(schema='native_saved_training_prefix_v1', batch='prefix',
        batch_complete=complete, endpoint_results=0,
        files={name: dict(bytes=len(payload), sha256=hashlib.sha256(b'changed' if corrupt else payload).hexdigest())})
    path = tmp_path / 'prefix.zip'
    with zipfile.ZipFile(path, 'w') as z:
        item = zipfile.ZipInfo(name)
        if symlink: item.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(item, payload)
        z.writestr('manifest.json', json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), manifest


def test_saved_prefix_restores_exact_payload_without_creating_an_endpoint(tmp_path):
    path, digest, expected = archive(tmp_path)
    output = tmp_path / 'runs'
    batch, manifest = restore_prefix(path, digest, output)
    assert batch == output / 'prefix' and manifest == expected
    assert (batch / 'protocol.json').read_bytes() == b'{}'
    assert not (batch / 'B1/evaluation.json').exists()
    with pytest.raises(FileExistsError): restore_prefix(path, digest, output)


@pytest.mark.parametrize('fault', ['sha', 'payload', 'symlink', 'complete'])
def test_changed_or_misrepresented_prefix_is_rejected_before_extraction(tmp_path, fault):
    path, digest, _ = archive(tmp_path, corrupt=fault == 'payload', symlink=fault == 'symlink', complete=fault == 'complete')
    if fault == 'sha': digest = '0' * 64
    output = tmp_path / 'unstarted'
    with pytest.raises(ValueError): restore_prefix(path, digest, output)
    assert not output.exists()


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'prefix/../outside', 'C:/outside', 'prefix\\bad', 'other/file'])
def test_prefix_paths_cannot_escape_the_original_batch(tmp_path, name):
    path, digest, _ = archive(tmp_path, name)
    output = tmp_path / 'unstarted'
    with pytest.raises(ValueError): restore_prefix(path, digest, output)
    assert not output.exists() and not (tmp_path / 'outside').exists()


def test_course_reconstruction_uses_original_hash_bound_packaged_figure(tmp_path, monkeypatch):
    import scripts.reproduce_broader_b1_release as module
    root = tmp_path / 'repository'
    (root / 'reports').mkdir(parents=True)
    report = root / 'reports/cohort.json'
    report.write_text('{}')
    artifacts = tmp_path / 'archives'
    artifacts.mkdir()
    rows = []
    for seed in (11, 22, 33, 44, 55):
        path = artifacts / f'seed{seed}.zip'
        path.write_bytes(b'placeholder')
        rows.append(dict(seed=seed, batch=f'batch{seed}', archive=path.name, archive_sha256=module.sha(path)))
    recipe = tmp_path / 'recipe.json'
    auditor = artifacts / 'auditors.zip'
    auditor.write_bytes(b'placeholder')
    recipe.write_text(json.dumps(dict(scope='all_five_broader_B1_development_v1', capsules=rows,
        cohort_report='reports/cohort.json', cohort_report_sha256=module.sha(report),
        saved_prefix44=dict(batch='prefix', archive='prefix.zip'),
        audit_sources=dict(archive=auditor.name, archive_sha256=module.sha(auditor)))))
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module, 'restore_auditors', lambda *args: {})
    output = tmp_path / 'rebuild'
    def capture(archive, digest, rebuilt, reference_figure):
        assert reference_figure == output / 'seed11_rebuild/asset/batch11/analysis/native_endpoints.png'
        assert digest == module.sha(archive)
        raise RuntimeError('checked original archived figure reference')
    monkeypatch.setattr(module, 'reproduce', capture)
    with pytest.raises(RuntimeError, match='checked original archived figure'):
        module.run(recipe, output, artifacts)


@pytest.mark.parametrize('corrupt', [False, True])
def test_recorded_auditor_source_identity_is_preserved_or_rejected(tmp_path, corrupt):
    from scripts.reproduce_broader_b1_release import restore_auditors
    name = 'snapshots/original/scripts/analyze_native_broader_b1_course.py'
    payload = b'# Original executed auditor bytes\n'
    digest = hashlib.sha256(payload).hexdigest()
    manifest = dict(schema='native_broader_audit_sources_v1',
        files={name: dict(bytes=len(payload), sha256=digest)},
        seed_sources={str(s): 'snapshots/original' for s in (11, 22, 33, 44, 55)},
        seed_auditor_sha256={str(s): '0' * 64 if corrupt else digest for s in (11, 22, 33, 44, 55)})
    path = tmp_path / 'sources.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, payload)
        z.writestr('manifest.json', json.dumps(manifest))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / 'sources'
    if corrupt:
        with pytest.raises(ValueError): restore_auditors(path, expected, output)
        assert not output.exists()
    else:
        result = restore_auditors(path, expected, output)
        assert result == manifest and (output / name).read_bytes() == payload
