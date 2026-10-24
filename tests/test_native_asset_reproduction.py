import hashlib
import json
import zipfile

import pytest

from scripts.reproduce_native_asset import reproduce


@pytest.mark.parametrize('failure', ['archive_hash', 'file_hash', 'path_escape'])
def test_reproduction_rejects_untrusted_content_before_extracting_or_loading(tmp_path, failure):
    archive = tmp_path/'release.zip'
    name = '../outside.pkl' if failure=='path_escape' else 'batch/model.pkl'
    payload = b'must never be deserialized'
    manifest = {'files':{name:{'bytes':len(payload), 'sha256':
        '0'*64 if failure=='file_hash' else hashlib.sha256(payload).hexdigest()}}}
    with zipfile.ZipFile(archive,'w') as stream:
        stream.writestr(name,payload)
        stream.writestr('manifest.json',json.dumps(manifest))
    expected = '0'*64 if failure=='archive_hash' else hashlib.sha256(archive.read_bytes()).hexdigest()
    output = tmp_path/'reproduction'
    with pytest.raises(ValueError):
        reproduce(archive,expected,output)
    assert not output.exists()
    assert not (tmp_path/'outside.pkl').exists()
