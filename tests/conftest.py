"""Explicit resource fixtures for staging/ownership unit tests, never training."""
from collections import namedtuple
import shutil

import pytest


@pytest.fixture
def ample_training_disk(monkeypatch):
    # These tests create tiny synthetic checkpoints and already mock the other
    # training resource checks. Hosted runner disk capacity is not their input.
    usage = namedtuple('DiskUsage', 'total used free')(64*1024**3, 32*1024**3, 32*1024**3)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage)
