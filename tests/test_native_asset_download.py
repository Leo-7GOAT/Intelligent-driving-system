import io
import json

import pytest

from scripts import download_native_asset as module
from scripts.native_asset_parts import split, digest

BASE = 'https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/test-v1/'


def remote(tmp_path, monkeypatch):
    source = tmp_path/'evidence.zip'
    source.write_bytes(b'whole original model and retained failure evidence'*4)
    manifest = split(source, tmp_path/'remote', 37)
    data = {BASE+p.name: p.read_bytes() for p in manifest.parent.iterdir()}
    requests = []

    def open_url(request, timeout):
        requests.append(request.full_url)
        assert 0 < timeout <= 60
        return io.BytesIO(data[request.full_url])

    monkeypatch.setattr(module, 'urlopen', open_url)
    return source, manifest, data, requests


def test_public_download_verifies_every_part_and_exact_archive_without_overwriting(tmp_path, monkeypatch):
    source, manifest, _, requests = remote(tmp_path, monkeypatch)
    output = tmp_path/'download'
    result = module.download(BASE+manifest.name, output, digest(source))
    assert result['archive_byte_exact'] and result['new_controls'] == result['optimizer_updates'] == 0
    assert (output/source.name).read_bytes() == source.read_bytes()
    assert len(requests) == result['parts']+1
    with pytest.raises(FileExistsError):module.download(BASE+manifest.name, output, digest(source))
    assert len(requests) == result['parts']+1


@pytest.mark.parametrize('fault', ['wrong_trusted_hash', 'traversal', 'bad_order', 'oversized', 'tampered'])
def test_corruption_or_untrusted_names_never_publish_an_archive(tmp_path, monkeypatch, fault):
    source, manifest, data, _ = remote(tmp_path, monkeypatch)
    record = json.loads(data[BASE+manifest.name])
    part = record['parts'][0]
    expected = digest(source)
    if fault == 'wrong_trusted_hash':expected = '0'*64
    elif fault == 'traversal':part['name'] = '../outside'
    elif fault == 'bad_order':part['name'] = 'wrong.part001'
    elif fault == 'oversized':data[BASE+part['name']] += b'oversized'
    else:data[BASE+part['name']] = b'x'*part['bytes']
    data[BASE+manifest.name] = json.dumps(record).encode()
    output = tmp_path/'download'
    with pytest.raises(ValueError):module.download(BASE+manifest.name, output, expected)
    assert not (output/source.name).exists()
    assert source.read_bytes().startswith(b'whole original')
    assert not (tmp_path/'outside').exists()


@pytest.mark.parametrize('url', ['http://github.com/not-https', 'https://example.com/manifest.parts.json',
                                BASE+'evidence.zip.parts.json?token=not-accepted'])
def test_invalid_source_is_rejected_before_any_request(tmp_path, monkeypatch, url):
    def never_request(*args, **kwargs):raise AssertionError('Network must not be reached')
    monkeypatch.setattr(module, 'urlopen', never_request)
    with pytest.raises(ValueError):module.download(url, tmp_path/'download', '0'*64)
    assert not (tmp_path/'download').exists()


def test_insufficient_disk_rejects_before_creating_output(tmp_path, monkeypatch):
    source, manifest, _, requests = remote(tmp_path, monkeypatch)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 0})())
    with pytest.raises(OSError):module.download(BASE+manifest.name, tmp_path/'download', digest(source))
    assert requests == [BASE+manifest.name] and not (tmp_path/'download').exists()


def test_expired_download_budget_never_opens_a_connection(monkeypatch):
    def never_request(*args, **kwargs):raise AssertionError('Network must not be reached')
    monkeypatch.setattr(module, 'urlopen', never_request)
    with pytest.raises(TimeoutError):module.read_bounded(BASE+'evidence.zip.part001', 37, 0)
