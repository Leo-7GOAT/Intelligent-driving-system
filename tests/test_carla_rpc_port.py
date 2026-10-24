import json
import subprocess
import sys

import pytest

from navstack.carla_bridge import BridgeClient


class _FakeInput:
    def write(self, unused):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class _ExitedBridge:
    def __init__(self):
        self.stdin = _FakeInput()
        self.stdout = ()

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def _batch(tmp_path, protocol):
    (tmp_path / "infrastructure").mkdir()
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    return tmp_path


def test_bridge_uses_frozen_rpc_endpoint(monkeypatch, tmp_path):
    batch = _batch(tmp_path, {
        "carla_rpc_host": "127.0.0.1",
        "carla_rpc_port": 2100,
        "carla_rpc_port_span": [2100, 2101, 2102],
    })
    command = []

    def fake_popen(args, **unused):
        command.extend(args)
        return _ExitedBridge()

    monkeypatch.setenv("NAVSTACK_CARLA_PYTHON", sys.executable)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    client = BridgeClient(batch)
    try:
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("--port") + 1] == "2100"
        assert client.rpc_port == 2100
    finally:
    client.close()

@pytest.mark.parametrize("protocol", [
    {"carla_rpc_port": 65534},
    {"carla_rpc_port": 2100, "carla_rpc_port_span": [2100, 2102]},
    {"carla_rpc_host": "bad host", "carla_rpc_port": 2100},
])
def test_bridge_rejects_invalid_rpc_endpoint_before_spawn(monkeypatch, tmp_path, protocol):
    batch = _batch(tmp_path, protocol)
    monkeypatch.setenv("NAVSTACK_CARLA_PYTHON", sys.executable)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("bridge spawned"))
    with pytest.raises(ValueError, match="CARLA RPC"):
        BridgeClient(batch)
