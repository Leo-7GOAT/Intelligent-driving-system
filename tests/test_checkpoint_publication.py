from pathlib import Path
import pytest
from scripts.native_road_learning import publish_checkpoint_directory


def staged(tmp_path):
    source=tmp_path/'checkpoint_000010000.partial';source.mkdir()
    (source/'sentinel').write_bytes(b'complete preserved bytes')
    return source,tmp_path/'checkpoint_000010000'


def test_transient_permission_failure_retries_without_rewriting_checkpoint_bytes(tmp_path,monkeypatch):
    import os
    source,target=staged(tmp_path);original=os.rename;calls=[];waits=[]
    def rename(a,b):
        calls.append((a,b))
        if len(calls)<3:raise PermissionError('synthetic transient directory handle')
        return original(a,b)
    monkeypatch.setattr('scripts.native_road_learning.os.rename',rename)
    monkeypatch.setattr('scripts.native_road_learning.time.sleep',waits.append)
    publish_checkpoint_directory(source,target)
    assert len(calls)==3 and waits==[.1,.2]
    assert (target/'sentinel').read_bytes()==b'complete preserved bytes' and not source.exists()


def test_persistent_permission_failure_is_bounded_and_keeps_staging(tmp_path,monkeypatch):
    source,target=staged(tmp_path);calls=[]
    def rename(a,b):calls.append(1);raise PermissionError('synthetic persistent failure')
    monkeypatch.setattr('scripts.native_road_learning.os.rename',rename)
    monkeypatch.setattr('scripts.native_road_learning.time.sleep',lambda seconds:None)
    with pytest.raises(PermissionError):publish_checkpoint_directory(source,target)
    assert len(calls)==5 and source.is_dir() and not target.exists()
    assert (source/'sentinel').read_bytes()==b'complete preserved bytes'


def test_checkpoint_publication_never_overwrites_or_moves_outside_parent(tmp_path):
    source,target=staged(tmp_path);target.mkdir()
    with pytest.raises(FileExistsError):publish_checkpoint_directory(source,target)
    with pytest.raises(ValueError):publish_checkpoint_directory(source,tmp_path/'elsewhere'/target.name)
