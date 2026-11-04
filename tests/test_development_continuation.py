"""Filesystem/ownership tests; fixtures are not training-performance evidence."""
import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures('ample_training_disk')

from navstack.process_lease import ProcessLease
from scripts import continue_development as continuation


@pytest.fixture
def parent(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "resources_ok", lambda root: (True, None))
    batch = tmp_path / "runs" / "parent"
    source = batch / "source"
    source.mkdir(parents=True)
    (source / "frozen.py").write_bytes(b"frozen source\n")
    job = {"id": "B1_seed11", "seed": 11, "env": {"method": "B1"},
           "ppo_options": {"entropy": .001}, "milestones": [100]}
    run = batch / "group" / job["id"]
    checkpoint = run / "checkpoint_100"
    checkpoint.mkdir(parents=True)
    summary = {"physics_steps": 100, "actions": 10, "episodes": 1,
               "optimization_phases": 2, "optimizer_steps": 20,
               "reload_exact": True, "checkpoint": str(checkpoint)}
    for filename in ("model.zip", "normalization.pkl", "resume.pkl"):
        (checkpoint / filename).write_bytes(b"fixture only " + filename.encode())
    continuation.write(checkpoint / "training_summary.json", summary)
    continuation.write(checkpoint / "config.json", {"env": job["env"], "seed": 11})
    continuation.write(run / "completed.json", {"summary": summary})
    continuation.write(run / "latest.json", summary)
    continuation.write(run / "ppo_options.json", job["ppo_options"])
    continuation.write(run / "version.json", {"commit": "fixture"})
    continuation.write(run / "resolved_config.json", {"env": job["env"], "deadline_epoch": 1})
    continuation.write(batch / "protocol.json", {"jobs": [job], "run_group": "group",
        "source_hashes": {"frozen.py": hashlib.sha256((source / "frozen.py").read_bytes()).hexdigest()}})
    continuation.write(batch / "supervisor_final.json", {"children": [{"job": job["id"], "exit_code": 0}]})
    return batch, run, tmp_path


def prepare(parent):
    batch, _, root = parent
    return continuation.prepare(batch, [200, 300], "continuation", "Bounded training-span diagnosis", root=root)


def test_copy_preserves_source_checkpoint_and_cumulative_provenance(parent):
    batch, run, root = parent
    originals = {p: p.read_bytes() for p in batch.rglob("*") if p.is_file()}
    new = prepare(parent)
    protocol = continuation.read(new / "protocol.json")
    assert protocol["jobs"][0]["milestones"] == [200, 300]
    assert protocol["inherited_checkpoints"][0]["inherited_physics_steps"] == 100
    assert protocol["inherited_checkpoints"][0]["inherited_optimizer_steps"] == 20
    copied = new / "group" / "B1_seed11"
    latest = continuation.read(copied / "latest.json")
    assert latest["physics_steps"] == 100
    assert Path(latest["checkpoint"]) == copied / "initial_checkpoint"
    for file in (run / "checkpoint_100").iterdir():
        assert file.read_bytes() == (copied / "initial_checkpoint" / file.name).read_bytes()
    assert (new / "source" / "frozen.py").read_bytes() == (batch / "source" / "frozen.py").read_bytes()
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert continuation.read(copied / "resolved_config.json")["deadline_epoch"] == protocol["deadline_epoch"] - 120
    assert continuation.read(root / "runs" / "continuation.json")["batch"] == str(new)


def test_low_disk_still_refuses_staging_before_creating_child(parent, monkeypatch):
    from types import SimpleNamespace
    batch, _, root = parent
    monkeypatch.setattr(continuation.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**3))
    with pytest.raises(RuntimeError, match='resource guard'):
        prepare(parent)
    assert list((root/'runs').iterdir()) == [batch]


@pytest.mark.parametrize("owner", ["supervisor", "worker"])
def test_reject_live_parent_owner(parent, owner):
    batch, run, root = parent
    lock = batch / "supervisor.lock" if owner == "supervisor" else run / "worker.lock"
    with ProcessLease(lock), pytest.raises((OSError, RuntimeError)):
        prepare(parent)
    assert list((root / "runs").iterdir()) == [batch]


@pytest.mark.parametrize("damage", ["source", "counter", "exit", "missing_resume", "options"])
def test_reject_inconsistent_parent_before_creating_batch(parent, damage):
    batch, run, root = parent
    if damage == "source":
        (batch / "source" / "frozen.py").write_bytes(b"changed")
    elif damage == "counter":
        data = continuation.read(run / "latest.json")
        data["optimizer_steps"] += 1
        continuation.write(run / "latest.json", data)
    elif damage == "exit":
        continuation.write(batch / "supervisor_final.json", {"children": []})
    elif damage == "missing_resume":
        (run / "checkpoint_100" / "resume.pkl").unlink()
    else:
        continuation.write(run / "ppo_options.json", {"entropy": .1})
    with pytest.raises(ValueError):
        prepare(parent)
    assert list((root / "runs").iterdir()) == [batch]


@pytest.mark.parametrize("milestones", [[100], [200, 200], [300, 200], [200.5]])
def test_reject_nonadvancing_or_invalid_budget(parent, milestones):
    batch, _, root = parent
    with pytest.raises(ValueError):
        continuation.prepare(batch, milestones, "continuation", "diagnosis", root=root)
    assert list((root / "runs").iterdir()) == [batch]


@pytest.mark.parametrize("stop", ["diagnostic_window_drain", "user_stop", "disk_below_12GiB"])
def test_incomplete_continuation_only_for_clean_scheduled_window(parent, stop):
    batch, run, root = parent
    (run / "completed.json").unlink()
    (batch / "STOP").write_text(stop, encoding="utf-8")
    protocol = continuation.read(batch / "protocol.json")
    protocol["deadline_epoch"] = 100
    continuation.write(batch / "protocol.json", protocol)
    latest = continuation.read(run / "latest.json")
    latest.update(reason="watchdog_or_user_stop", saved_epoch=100)
    continuation.write(run / "latest.json", latest)
    continuation.write(run / "checkpoint_100" / "training_summary.json", latest)
    with pytest.raises(ValueError, match="scheduled window"):
        prepare(parent)
    if stop == "diagnostic_window_drain":
        new = continuation.prepare(batch, [200], "continuation", "Complete registered budget", root=root,
                                   jobs=["B1_seed11"], allow_window_end=True)
        assert continuation.read(new / "protocol.json")["inherited_checkpoints"][0]["parent_status"] == "scheduled_window_end"
        assert (batch / "STOP").read_text() == stop
    else:
        with pytest.raises(ValueError, match="scheduled window"):
            continuation.prepare(batch, [200], "continuation", "diagnosis", root=root, allow_window_end=True)
