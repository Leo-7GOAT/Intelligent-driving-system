import hashlib
import json
import pytest
from scripts.native_asset_parts import split
from scripts.reproduce_public_feedback_cohort import cached_download


def cache(tmp_path):
    source = tmp_path/'evidence.zip';source.write_bytes(b'public evidence payload')
    folder = tmp_path/'public';manifest = split(source, folder, 9)
    (folder/source.name).write_bytes(source.read_bytes())
    record = json.loads(manifest.read_text())
    url = 'https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/test-v1/'+manifest.name
    row = {'manifest_url': url, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    receipt = {'manifest_url': url, 'archive': source.name, 'archive_byte_exact': True,
               'sha256': row['sha256'], 'bytes': len(source.read_bytes()), 'parts': len(record['parts'])}
    (folder/'download_receipt.json').write_text(json.dumps(receipt))
    return row, folder


def test_prior_public_download_is_rehashed_and_provenance_is_preserved(tmp_path):
    row, folder = cache(tmp_path);target = tmp_path/'fresh'
    result = cached_download(row, folder, target)
    assert result['transfer_source'] == 'verified_prior_public_download'
    assert (target/'evidence.zip').read_bytes() == (folder/'evidence.zip').read_bytes()
    assert result['original_download_receipt_sha256'] == hashlib.sha256((folder/'download_receipt.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['tamper', 'false_receipt', 'traversal'])
def test_local_or_corrupt_archive_cannot_impersonate_checked_public_evidence(tmp_path, fault):
    row, folder = cache(tmp_path);target = tmp_path/'fresh'
    if fault == 'tamper':(folder/'evidence.zip').write_bytes(b'corrupt')
    else:
        p = folder/'download_receipt.json';r = json.loads(p.read_text())
        if fault == 'false_receipt':r['archive_byte_exact'] = False
        else:r['archive'] = '../evidence.zip'
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):cached_download(row, folder, target)
    assert not target.exists()


def test_unrelated_cached_capsule_is_not_relabelled(tmp_path):
    row, folder = cache(tmp_path);row['manifest_url'] = row['manifest_url'].replace('test-v1', 'other-v1')
    assert cached_download(row, folder, tmp_path/'fresh') is None
