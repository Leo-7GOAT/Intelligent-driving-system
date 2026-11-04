import os
import subprocess
import sys
from pathlib import Path
import pytest
from navstack.process_lease import ProcessLease


def test_live_owner_excluded_and_clean_reuse(tmp_path):
    path = tmp_path / "owner.lock"
    with ProcessLease(path):
        with pytest.raises((OSError, RuntimeError)):
            ProcessLease(path).acquire()
    assert path.read_bytes() == b""
    with ProcessLease(path):
        pass


def test_process_death_releases_kernel_lock_preserving_evidence(tmp_path):
    path = tmp_path / "owner.lock"
    script = "from navstack.process_lease import ProcessLease; import sys,time; lease=ProcessLease(sys.argv[1]).acquire(); print('ready',flush=True); time.sleep(120)"
    child = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-c", script, str(path)], stdout=subprocess.PIPE, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises((OSError, RuntimeError)):
            ProcessLease(path).acquire()
        child.kill()
        child.wait(timeout=10)
        with ProcessLease(path):
            archives = list(tmp_path.glob("owner.lock.stale.*"))
            assert len(archives) == 1
            assert int(archives[0].read_text()) == child.pid
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_live_legacy_and_unknown_metadata_preserved(tmp_path):
    path = tmp_path / "owner.lock"
    for content in (str(os.getpid()), "unrecognized"):
        path.write_text(content)
        with pytest.raises(RuntimeError):
            ProcessLease(path).acquire()
        assert path.read_text() == content
