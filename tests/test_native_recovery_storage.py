from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import prepare_native_recovery as recovery


def test_explicit_archive_uses_its_own_free_space_instead_of_source_drive(tmp_path, monkeypatch):
    workspace, archive = tmp_path / 'workspace', tmp_path / 'archive'
    workspace.mkdir();archive.mkdir()
    monkeypatch.setattr(recovery, 'ROOT', workspace)
    seen = []
    def resources(path):
        seen.append(Path(path))
        return (True, None) if Path(path) == archive else (False, 'source drive full')
    monkeypatch.setattr(recovery, 'resources_ok', resources)
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20*1024**3 if Path(path)==archive else 1))
    destination = recovery.recovery_destination(archive / 'child')
    assert destination == archive / 'child' and seen == [archive]
    assert not destination.exists()


def test_existing_destination_is_preserved_before_any_resource_or_copy_work(tmp_path, monkeypatch):
    destination = tmp_path / 'existing';destination.mkdir()
    (destination / 'sentinel').write_bytes(b'preserve learned state')
    monkeypatch.setattr(recovery, 'resources_ok', lambda path: pytest.fail('Existing destination must be refused first'))
    with pytest.raises(FileExistsError):
        recovery.recovery_destination(destination)
    assert (destination / 'sentinel').read_bytes() == b'preserve learned state'


def test_selected_artifact_drive_still_enforces_twelve_gib_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'resources_ok', lambda path: (True, None))
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=12*1024**3-1))
    with pytest.raises(RuntimeError, match='below12GiB'):
        recovery.recovery_destination(tmp_path / 'child')
    assert not (tmp_path / 'child').exists()


def test_default_destination_remains_in_workspace_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'resources_ok', lambda path: (True, None))
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20*1024**3))
    assert recovery.recovery_destination().parent == tmp_path / 'runs'
