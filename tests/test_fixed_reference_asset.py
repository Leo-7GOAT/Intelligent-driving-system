import hashlib
import json
import zipfile

import pytest

from scripts.fixed_reference_asset import verify


def test_asset_rejects_member_changed_after_manifest(tmp_path):
    archive = tmp_path/'changed.zip'
    manifest = {'version': 'fixed_reference_asset_v1', 'batch': 'batch',
                'files': {'batch/trace.json': {'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}}}
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('manifest.json', json.dumps(manifest))
        stream.writestr('batch/trace.json', '[]')
    with pytest.raises(ValueError, match='differs from manifest'):
        verify(archive, hashlib.sha256(archive.read_bytes()).hexdigest())


@pytest.mark.parametrize('member', ['../outside.json', '/outside.json', 'C:/outside.json', '..\\outside.json'])
def test_asset_rejects_unsafe_extraction_paths(tmp_path, member):
    archive = tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        info = zipfile.ZipInfo('placeholder.json')
        # ZipInfo normally repairs Windows backslashes during construction.
        info.filename = member
        stream.writestr(info, '{}')
    with pytest.raises(ValueError, match='Unsafe archive path'):
        verify(archive, hashlib.sha256(archive.read_bytes()).hexdigest())


def test_asset_rejects_wrong_release_hash_before_parsing(tmp_path):
    archive = tmp_path/'untrusted.zip'
    archive.write_bytes(b'not a ZIP')
    with pytest.raises(ValueError, match='trusted release hash'):
        verify(archive, '0'*64)
