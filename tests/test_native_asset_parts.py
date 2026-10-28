import json
import zipfile
import pytest
from scripts.native_asset_parts import split,join,digest


def archive(tmp_path):
    path=tmp_path/'evidence.zip'
    with zipfile.ZipFile(path,'w') as stream:stream.writestr('model/manifest.json','retained failures and exact models')
    return path


def test_split_rejoin_preserves_entire_usable_zip_and_existing_output(tmp_path):
    source=archive(tmp_path);manifest=split(source,tmp_path/'parts',17);output=tmp_path/'rebuilt.zip'
    result=join(manifest,output,digest(source))
    assert result['archive_byte_exact'] and output.read_bytes()==source.read_bytes()
    with zipfile.ZipFile(output) as stream:assert stream.read('model/manifest.json')==b'retained failures and exact models'
    with pytest.raises(FileExistsError):join(manifest,output,digest(source))


@pytest.mark.parametrize('fault',['tampered','missing','duplicate','traversal','wrong_total','wrong_trusted_hash'])
def test_corrupt_or_unsafe_parts_never_publish_an_archive(tmp_path,fault):
    source=archive(tmp_path);manifest=split(source,tmp_path/'parts',17);m=json.loads(manifest.read_text())
    first=manifest.parent/m['parts'][0]['name'];expected=digest(source)
    if fault=='tampered':first.write_bytes(b'changed')
    elif fault=='missing':first.unlink()
    elif fault=='duplicate':m['parts'].append(m['parts'][0])
    elif fault=='traversal':m['parts'][0]['name']='../evidence.zip'
    elif fault=='wrong_total':m['bytes']+=1
    else:expected='0'*64
    manifest.write_text(json.dumps(m));output=tmp_path/'rebuilt.zip'
    with pytest.raises((ValueError,FileNotFoundError)):join(manifest,output,expected)
    assert not output.exists()
