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
