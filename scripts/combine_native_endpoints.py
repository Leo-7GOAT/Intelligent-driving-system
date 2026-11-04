"""Validate and continue method endpoints that completed in different batches.

An interrupted paired run can leave H0/H complete while B1 finishes in a
recovery child.  This module records those endpoints without copying results or
pretending that the recovery was free.  All endpoint claims are committed in
one atomic ledger update, avoiding a partially registered multi-parent run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.continue_native_learning import (
    ORCHESTRATION_FILES,
    _allocate_batch,
    _record_abort,
    assert_continuation_pointer_available,
    read,
    sha,
    verified_endpoint,
    write_json_atomic,
)
from scripts.audit_native_method_endpoint import audit as audit_method_endpoint


METHODS = ("H0", "H", "B1")
PROTOCOL_COMPATIBILITY_KEYS = (
    "seed", "controlled_budget_per_method", "rollout_actions", "ppo_epochs",
    "episode_seconds", "road_candidate_policy", "road_geometry_version",
    "deterministic_training_route_evaluation", "reset_ticks_counted_separately",
    "calibration_sha256", "routes", "hypothesis",
)
# These files may change solely to carry a checkpoint, select a free RPC port,
# or supervise a process.  Learning, action, reward, planning and control source
# hashes remain exact across all method endpoints.
INFRASTRUCTURE_SOURCE_KEYS = {
    "navstack/carla_bridge.py",
    "scripts/carla_bridge37.py",
    "scripts/probe_carla37.py",
    "scripts/prepare_native_recovery.py",
    "scripts/continue_native_learning.py",
    "scripts/combine_native_endpoints.py",
    "scripts/watch_existing_pair.py",
    "scripts/start_carla.ps1",
    "scripts/run_carla_owned_foreground.ps1",
    "scripts/stop_owned_carla.ps1",
    "scripts/diagnose_carla_start.ps1",
    "scripts/audit_native_method_endpoint.py",
}


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _value_hash(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> dict[str, str]:
    path, root = path.resolve(), root.resolve()
    try:
        return {"root_relative": path.relative_to(root).as_posix()}
    except ValueError:
        return {"absolute": str(path)}


def _resolve_location(location: dict[str, str], root: Path) -> Path:
    if set(location) == {"root_relative"}:
        return (root / location["root_relative"]).resolve()
    if set(location) == {"absolute"}:
        return Path(location["absolute"]).resolve()
    raise ValueError("Invalid endpoint location")


def _common_config(config: dict) -> dict:
    result = deepcopy(config)
    # Scope describes provenance (original paired run versus interrupted
    # recovery), not environment or learning semantics.  Its exact text and
    # containing config hash remain recorded per endpoint below.
    result.pop("scope", None)
    env = result.get("env")
    signature = result.get("signature")
    if not isinstance(env, dict) or not isinstance(signature, dict):
        raise ValueError("Endpoint config lacks env/signature dictionaries")
    env.pop("method", None)
    env.pop("execution_feedback", None)
    signature.pop("method", None)
    signature.pop("execution_feedback", None)
    signature.pop("action_catalog", None)
    return result


def _method_role(method: str, config: dict) -> None:
    env = config["env"]
    if method == "H0" and (env.get("method") != "H" or env.get("execution_feedback") != "ablated"):
        raise ValueError("H0 endpoint is not the declared feedback ablation")
    if method == "H" and (env.get("method") != "H" or env.get("execution_feedback") != "enabled"):
        raise ValueError("H endpoint is not the enabled-feedback hierarchy")
    if method == "B1" and env.get("method") != "B1":
        raise ValueError("B1 endpoint is not the direct-control PPO")


def _raw_audit(batch: Path, method: str, summary: dict) -> tuple[Path, dict]:
    prefix_path = batch / "interruption_audit.json"
    if not prefix_path.is_file():
        raise ValueError(f"{method} source has no readable-prefix raw audit")
    prefix_audit = read(prefix_path)
    row = prefix_audit.get("methods", {}).get(method)
    if not isinstance(row, dict) or row.get("state") != "complete":
        raise ValueError(f"{method} raw audit does not mark the endpoint complete")
    if row.get("readable_trace_controls") != summary["physics_steps"]:
        raise ValueError(f"{method} raw trace count differs from endpoint")
    if row.get("trace_errors") or row.get("optimization_error"):
        raise ValueError(f"{method} completed endpoint has raw-stream errors")
    if row.get("unknown_unflushed_tail"):
        raise ValueError(f"{method} completed endpoint still has an unflushed local tail")
    last = row.get("last_optimization")
    if not isinstance(last, dict) or any(last.get(name) != summary[name]
                                         for name in ("physics_steps", "optimization_phases", "optimizer_steps")):
        raise ValueError(f"{method} prefix audit optimization counters differ from endpoint")
    path = batch / method / "endpoint_raw_audit.json"
    if not path.is_file():
        raise ValueError(f"{method} has no per-method CARLA frame audit")
    stored = read(path)
    rebuilt = audit_method_endpoint(batch, method)
    if stored != rebuilt:
        raise ValueError(f"{method} per-method CARLA frame audit is stale or changed")
    if (stored.get("training_control_frames_verified") != summary["physics_steps"] or
            stored.get("lineage_controls") != summary["lineage_control_steps"] or
            stored.get("charged_controls_lower_bound") != summary["charged_control_steps_lower_bound"] or
            stored.get("segment_optimizer_steps") != summary["optimizer_steps"]):
        raise ValueError(f"{method} per-method audit accounting differs from endpoint")
    stored = {**stored, "readable_prefix_audit_sha256": sha(prefix_path)}
    return path, stored


def _verify_frozen_source(batch: Path, protocol: dict) -> None:
    hashes = protocol.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Source protocol has no frozen hashes")
    for relative, digest in hashes.items():
        path = batch / "source" / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Frozen source changed: {relative}")
    calibration = batch / "calibration" / "calibration.json"
    if not calibration.is_file() or sha(calibration) != protocol.get("calibration_sha256"):
        raise ValueError("Frozen calibration changed")


def _endpoint_record(method: str, folder: Path, target: int, root: Path) -> tuple[dict, dict, dict]:
    folder = folder.resolve()
    if folder.name != method:
        raise ValueError(f"Source folder {folder} does not match method {method}")
    batch = folder.parent
    protocol = read(batch / "protocol.json")
    if method not in protocol.get("methods", []):
        raise ValueError(f"Protocol does not contain method {method}")
    _verify_frozen_source(batch, protocol)
    checkpoint, summary, metadata = verified_endpoint(folder, target)
    config = read(folder / "config.json")
    _method_role(method, config)
    audit_path, audit_row = _raw_audit(batch, method, summary)
    prior = protocol.get("recovery", {}).get(method, {})
    checkpoint_hashes = {path.name: sha(path) for path in sorted(checkpoint.iterdir()) if path.is_file()}
    record = {
        "method": method,
        "source_batch": _relative_or_absolute(batch, root),
        "source_batch_name": batch.name,
        "source_method_folder": method,
        "protocol_sha256": sha(batch / "protocol.json"),
        "config_sha256": sha(folder / "config.json"),
        "source_config_scope": config.get("scope"),
        "summary_sha256": sha(folder / "summary.json"),
        "evaluation_sha256": sha(folder / "evaluation.json"),
        "raw_audit_sha256": sha(audit_path),
        "raw_audit": {
            "training_control_frames_verified": audit_row["training_control_frames_verified"],
            "evaluation_control_frames_verified": audit_row["evaluation_control_frames_verified"],
            "total_applied_controls_verified": audit_row["total_applied_controls_verified"],
            "bridge_response_sha256": audit_row["bridge_response_sha256"],
            "readable_prefix_audit_sha256": audit_row["readable_prefix_audit_sha256"],
        },
        "checkpoint": checkpoint.name,
        "checkpoint_hashes": checkpoint_hashes,
        "lineage_controls": summary["lineage_control_steps"],
        "charged_controls_lower_bound": summary["charged_control_steps_lower_bound"],
        "optimizer_steps_this_segment": audit_row["segment_optimizer_steps"],
        "retained_optimizer_steps_before_segment": prior.get("retained_optimizer_steps", 0),
        "optimizer_steps_total_lower_bound": audit_row["optimizer_steps_total_lower_bound"],
        "unknown_unflushed_ancestor_tail": audit_row["unknown_unflushed_ancestor_tail"],
        "recovery_chain": list(prior.get("recovery_chain", [])),
        "evaluation": read(folder / "evaluation.json"),
    }
    return record, protocol, config


def build_manifest(sources: dict[str, Path], target: int, output: Path | None = None,
                   root: Path = ROOT) -> dict:
    if tuple(sources) != METHODS and set(sources) != set(METHODS):
        raise ValueError(f"Combined endpoints require exactly {METHODS}")
    if not 20000 < target <= 1_000_000:
        raise ValueError("Combined continuation target outside supported bounds")
    records, protocols, configs = {}, {}, {}
    for method in METHODS:
        records[method], protocols[method], configs[method] = _endpoint_record(
            method, Path(sources[method]), target, Path(root))
    protocol_values = {
        method: {key: protocols[method].get(key) for key in PROTOCOL_COMPATIBILITY_KEYS}
        for method in METHODS
    }
    baseline = protocol_values[METHODS[0]]
    if any(protocol_values[method] != baseline for method in METHODS[1:]):
        raise ValueError("Endpoint protocols differ in training/task semantics")
    common_configs = {method: _common_config(configs[method]) for method in METHODS}
    first_config = common_configs[METHODS[0]]
    if any(common_configs[method] != first_config for method in METHODS[1:]):
        raise ValueError("Endpoint configs differ beyond declared method/feedback/action roles")
    learning_sources = {
        method: {key: value for key, value in protocols[method]["source_hashes"].items()
                 if key not in INFRASTRUCTURE_SOURCE_KEYS}
        for method in METHODS
    }
    first_sources = learning_sources[METHODS[0]]
    if any(learning_sources[method] != first_sources for method in METHODS[1:]):
        raise ValueError("Learning/planning/control source hashes differ across endpoints")
    payload = {
        "schema_version": 1,
        "kind": "native_multi_batch_endpoint_manifest",
        "created_epoch": time.time(),
        "methods": list(METHODS),
        "target_controls": target,
        # B1 is the latest recovery lineage and therefore supplies only runtime
        # orchestration/port plumbing; compatible learning source is proven above.
        "runtime_source_method": "B1",
        "compatibility": {
            "protocol_values": baseline,
            "protocol_values_sha256": _value_hash(baseline),
            "common_config_sha256": _value_hash(first_config),
            "learning_source_hashes": first_sources,
            "learning_source_hashes_sha256": _value_hash(first_sources),
        },
        "sources": records,
        "cost_note": "Each method retains its policy lineage, all charged controls, optimizer history, recovery chain, and unknown interrupted tails.",
    }
    payload["manifest_id"] = _value_hash(payload)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    return payload


def verify_manifest(path: Path, root: Path = ROOT) -> dict:
    manifest = read(path)
    identifier = manifest.pop("manifest_id", None)
    if identifier != _value_hash(manifest):
        raise ValueError("Combined endpoint manifest content hash mismatch")
    manifest["manifest_id"] = identifier
    if manifest.get("schema_version") != 1 or manifest.get("methods") != list(METHODS):
        raise ValueError("Unsupported combined endpoint manifest")
    sources = {}
    for method in METHODS:
        record = manifest["sources"][method]
        batch = _resolve_location(record["source_batch"], Path(root))
        sources[method] = batch / record["source_method_folder"]
    rebuilt = build_manifest(sources, manifest["target_controls"], root=Path(root))
    for key in ("methods", "target_controls", "runtime_source_method", "compatibility", "sources", "cost_note"):
        if rebuilt[key] != manifest[key]:
            raise ValueError(f"Combined endpoint manifest no longer matches sources: {key}")
    return manifest


def _claim_key(record: dict) -> str:
    identity = {
        "source_batch": record["source_batch"], "method": record["method"],
        "checkpoint": record["checkpoint"],
    }
    return _value_hash(identity)


def _claim_endpoints_atomic(runs: Path, manifest: dict, batch: Path) -> dict:
    path = runs / "native_endpoint_claims.json"
    ledger = read(path) if path.is_file() else {"schema_version": 1, "claims": {}}
    if ledger.get("schema_version") != 1 or not isinstance(ledger.get("claims"), dict):
        raise ValueError("Endpoint claim ledger is unreadable")
    keys = {method: _claim_key(manifest["sources"][method]) for method in METHODS}
    conflicts = {method: ledger["claims"][key] for method, key in keys.items() if key in ledger["claims"]}
    if conflicts:
        raise FileExistsError("Endpoint already has a continuation claim: " + _canonical(conflicts))
    updated = deepcopy(ledger)
    now = time.time()
    for method, key in keys.items():
        record = manifest["sources"][method]
        updated["claims"][key] = {
            "method": method, "source_batch": record["source_batch"],
            "checkpoint": record["checkpoint"], "manifest_id": manifest["manifest_id"],
            "checkpoint_hashes": record["checkpoint_hashes"],
            "child_batch": batch.name, "target_controls": manifest["target_controls"],
            "epoch": now,
        }
    write_json_atomic(path, updated)
    return {"ledger": str(path), "ledger_sha256": sha(path), "claim_keys": keys}


def prepare_from_manifest(manifest_path: Path, wall_seconds: int = 7200,
                          root: Path = ROOT) -> Path:
    root = Path(root).resolve()
    if not 900 <= wall_seconds <= 21600:
        raise ValueError("Continuation wall budget outside supported bounds")
    healthy, reason = resources_ok(root)
    if not healthy or shutil.disk_usage(root).free < 12 * 1024 ** 3:
        raise RuntimeError("Resource guard: " + str(reason))
    runs = root / "runs"
    with ProcessLease(runs / "native_continuation.prepare.lock"):
        assert_continuation_pointer_available(runs)
        manifest = verify_manifest(Path(manifest_path), root)
        runtime_record = manifest["sources"][manifest["runtime_source_method"]]
        runtime_parent = _resolve_location(runtime_record["source_batch"], root)
        template = read(runtime_parent / "protocol.json")
        batch = _allocate_batch(runs)
        try:
            (batch / "infrastructure").mkdir()
            shutil.copytree(runtime_parent / "source", batch / "source",
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(runtime_parent / "calibration", batch / "calibration")
            (batch / "source" / "scripts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "navstack/independent_clip_ppo.py", batch / "source/navstack/independent_clip_ppo.py")
            for name in (*ORCHESTRATION_FILES, "combine_native_endpoints.py",
                         "audit_native_method_endpoint.py"):
                shutil.copy2(root / "scripts" / name, batch / "source" / "scripts" / name)
            source_hashes = {
                path.relative_to(batch / "source").as_posix(): sha(path)
                for path in (batch / "source").rglob("*")
                if path.is_file() and path.suffix in (".py", ".ps1")
            }
            recovery, budgets, parent_records = {}, {}, {}
            target = manifest["target_controls"]
            for method in METHODS:
                record = manifest["sources"][method]
                source_batch = _resolve_location(record["source_batch"], root)
                checkpoint = source_batch / method / record["checkpoint"]
                destination = batch / "resume" / method
                shutil.copytree(checkpoint, destination)
                copied = {path.name: sha(path) for path in destination.iterdir() if path.is_file()}
                if copied != record["checkpoint_hashes"]:
                    raise ValueError(f"Copied {method} checkpoint differs from manifest")
                recovery[method] = {
                    "checkpoint": f"resume/{method}",
                    "parent_batch": record["source_batch_name"],
                    "parent_checkpoint": record["checkpoint"],
                    "lineage_controls": record["lineage_controls"],
                    "consumed_controls_lower_bound": record["charged_controls_lower_bound"],
                    "retained_optimizer_steps": record["optimizer_steps_total_lower_bound"],
                    "model_sha256": record["checkpoint_hashes"]["model.zip"],
                    "normalization_sha256": record["checkpoint_hashes"]["normalization.pkl"],
                    "unknown_unflushed_parent_tail": record["unknown_unflushed_ancestor_tail"],
                    "recovery_chain": record["recovery_chain"] + [record["source_batch_name"]],
                }
                budgets[method] = target - record["lineage_controls"]
                parent_records[method] = record
            now = time.time()
            template.pop("parent_batch", None)
            template.update(
                created_epoch=now, deadline_epoch=now + wall_seconds,
                methods=list(METHODS), source_hashes=source_hashes,
                controlled_budget_per_method=target, method_budgets=budgets,
                recovery=recovery, parent_batches={m: manifest["sources"][m]["source_batch_name"] for m in METHODS},
                parent_records=parent_records, combined_endpoint_manifest_id=manifest["manifest_id"],
                combined_endpoint_manifest_sha256=sha(manifest_path),
                scope="Multi-batch completed-policy native continuation; same seed/task/learning source, new measured episodes, inherited costs and interrupted tails retained; not independent replication",
                continuation_reason=f"Continue compatible H0/H/B1 endpoints to cumulative {target} controls without changing learning semantics",
                disk_free_bytes=shutil.disk_usage(root).free,
            )
            template["resource_limits"]["wall_seconds"] = wall_seconds
            write_json(batch / "protocol.json", template)
            write_json_atomic(batch / "combined_endpoint_manifest.json", manifest)
            write_json_atomic(batch / "prepared.json", {
                "status": "PREPARED_UNREGISTERED", "epoch": time.time(),
                "child_batch": batch.name, "protocol_sha256": sha(batch / "protocol.json"),
                "manifest_id": manifest["manifest_id"], "methods": list(METHODS),
            })
        except BaseException as exc:
            _record_abort(batch, "multi_batch_manifest", "staging", exc)
            raise
        try:
            claims = _claim_endpoints_atomic(runs, manifest, batch)
        except BaseException as exc:
            _record_abort(batch, "multi_batch_manifest", "endpoint_registration", exc)
            raise
        try:
            write_json_atomic(runs / "native_continuation.json", {"batch": str(batch)})
        except BaseException as exc:
            write_json_atomic(batch / "pointer_publication_failure.json", {
                "status": "REGISTERED_CHILD_POINTER_NOT_PUBLISHED", "epoch": time.time(),
                "claims": claims, "error_type": type(exc).__name__, "error": str(exc),
            })
            raise
        write_json_atomic(batch / "registered.json", {
            "status": "REGISTERED_READY", "epoch": time.time(), **claims,
            "pointer_sha256": sha(runs / "native_continuation.json"),
        })
        print(batch, flush=True)
        return batch


def _parse_sources(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Source must be METHOD=METHOD_FOLDER")
        method, path = value.split("=", 1)
        if method in result:
            raise ValueError("Duplicate source method: " + method)
        result[method] = Path(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", action="append", required=True)
    build.add_argument("--target", type=int, default=100000)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--wall-seconds", type=int, default=7200)
    arguments = parser.parse_args()
    if arguments.command == "build":
        result = build_manifest(_parse_sources(arguments.source), arguments.target, arguments.output)
    elif arguments.command == "verify":
        result = verify_manifest(arguments.manifest)
    else:
        result = {"batch": str(prepare_from_manifest(arguments.manifest, arguments.wall_seconds))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
