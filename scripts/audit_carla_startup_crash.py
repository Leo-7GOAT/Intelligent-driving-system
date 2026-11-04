"""Snapshot and symbolicate a CARLA startup crash without modifying the install.

The packaged CARLA build includes a matching PDB.  Keeping the original crash
XML plus resolved function names prevents a generic Windows exit code from
being mistaken for a launcher or GPU failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_properties(context: Path) -> dict[str, str]:
    root = ET.parse(context).getroot()
    node = root.find("RuntimeProperties")
    if node is None:
        raise ValueError("CrashContext has no RuntimeProperties")
    return {child.tag: child.text or "" for child in node}


def symbolicate(symbolizer: Path, executable: Path, stack: str) -> list[dict[str, str]]:
    offsets = re.findall(r"CarlaUE4-Win64-Shipping\s+0x[0-9a-fA-F]+\s+\+\s+([0-9a-fA-F]+)", stack)
    if not offsets:
        raise ValueError("Crash stack contains no CARLA Shipping offsets")
    command = [str(symbolizer), f"--obj={executable}", "--relative-address",
               "--functions=linkage", "--demangle", *[f"0x{x}" for x in offsets]]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    blocks = re.split(r"\r?\n\r?\n", completed.stdout.strip())
    frames = []
    for offset, block in zip(offsets, blocks):
        lines = block.splitlines()
        frames.append({"offset": f"0x{offset}", "function": lines[0] if lines else "",
                       "location": lines[1] if len(lines) > 1 else ""})
    return frames


def port_owners(rpc_port: int) -> list[dict]:
    ports = ",".join(str(rpc_port + offset) for offset in range(3))
    script = (
        f"$rows=@(Get-NetTCPConnection -LocalPort {ports} -ErrorAction SilentlyContinue);"
        "$result=@();foreach($r in $rows){$p=Get-CimInstance Win32_Process "
        "-Filter ('ProcessId='+$r.OwningProcess) -ErrorAction SilentlyContinue;"
        "$result+=[pscustomobject]@{local_address=$r.LocalAddress;local_port=$r.LocalPort;"
        "state=[string]$r.State;owning_process=$r.OwningProcess;process_name=$p.Name;"
        "executable=$p.ExecutablePath;creation_date=[string]$p.CreationDate;command_line=$p.CommandLine}};"
        "$result|ConvertTo-Json -Depth 4 -Compress"
    )
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                               check=True, text=True, capture_output=True)
    payload = completed.stdout.strip()
    if not payload:
        return []
    decoded = json.loads(payload)
    return decoded if isinstance(decoded, list) else [decoded]


def audit(context: Path, executable: Path, symbolizer_path: Path, rpc_port: int,
          output: Path) -> dict:
    context = context.resolve()
    executable = executable.resolve()
    symbolizer_path = symbolizer_path.resolve()
    if output.exists():
        raise FileExistsError("Crash audit output already exists")
    for path in (context, executable, symbolizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    props = runtime_properties(context)
    stack = props.get("PCallStack", "")
    frames = symbolicate(symbolizer_path, executable, stack)
    owners = port_owners(rpc_port)
    rpc_constructor = any("rpc::server" in frame["function"] or
                          "basic_socket_acceptor" in frame["function"] for frame in frames[:8])
    diagnosis = "RPC_BIND_CONFLICT_SUPPORTED" if rpc_constructor and owners else (
        "RPC_CONSTRUCTOR_FAILURE_WITHOUT_CURRENT_PORT_OWNER" if rpc_constructor else
        "UNCLASSIFIED_STARTUP_CRASH")
    output.mkdir(parents=True)
    copied = output / "CrashContext.runtime-xml"
    shutil.copy2(context, copied)
    minidump = context.with_name("UE4Minidump.dmp")
    result = {
        "generated_epoch": time.time(),
        "scope": "Crash snapshot and local-PDB symbolication; diagnosis is evidence-based, not a fabricated CARLA result.",
        "diagnosis": diagnosis,
        "crash_guid": props.get("CrashGUID"),
        "process_id": int(props["ProcessId"]) if props.get("ProcessId", "").isdigit() else props.get("ProcessId"),
        "crash_type": props.get("CrashType"),
        "error_message": props.get("ErrorMessage"),
        "command_line": props.get("CommandLine"),
        "stack_hash": props.get("PCallStackHash"),
        "crash_context_sha256": sha256(copied),
        "minidump_sha256": sha256(minidump) if minidump.is_file() else None,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "symbolizer": str(symbolizer_path),
        "rpc_port_span": [rpc_port, rpc_port + 1, rpc_port + 2],
        "observed_port_owners": owners,
        "symbolized_frames": frames,
        "raw_portable_call_stack": stack,
    }
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-context", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--symbolizer", required=True, type=Path)
    parser.add_argument("--rpc-port", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.crash_context, arguments.executable,
                           arguments.symbolizer, arguments.rpc_port,
                           arguments.output), indent=2))
