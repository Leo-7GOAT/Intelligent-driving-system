"""Verify one completed native-policy endpoint against its CARLA bridge frames.

This verifier is method-local so a complete H/H0 stream remains auditable even
when a later method in the same parent batch has a truncated gzip tail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prefix(path: Path):
    rows, error = [], None
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
    except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return rows, error


def _verify_ticks(ticks, actual, acknowledgements, owners, used, previous, episode_ids, actuator_observation="off"):
    count = 0
    for tick in ticks:
        evidence = tick["backend_evidence"]
        key = (evidence["episode_id"], evidence["frame"])
        if key in used:
            raise ValueError("CARLA control frame counted more than once")
        if key not in actual:
            raise ValueError("Recorded policy frame is absent from bridge responses")
        measured = actual[key]
        if ('native_dynamic_actors' in measured)!=('native_dynamic_actors' in evidence):
            raise ValueError('Native dynamic actor measurements missing or fabricated in trace')
        if actuator_observation != "off":
            from navstack.carla_actuation import actuator_state_values
            actuator_state_values(measured)
            actuator_state_values(evidence)
            for name in ("actuator_observation", "speed_setpoint", "actuation_state"):
                if evidence[name] != measured[name]:
                    raise ValueError("Actuator evidence differs from measured bridge frame")
        final_ack = None
        if evidence["request_id"] != measured.get("request_id"):
            if evidence["request_id"] != measured.get("request_id") + 1:
                raise ValueError("Trace request does not match applied or terminal-fence acknowledgement")
            final_ack = acknowledgements.get((evidence["episode_id"], evidence["request_id"]))
            if (not final_ack or final_ack.get("frame") != evidence["frame"] or
                    final_ack.get("terminal_sensor_drain_seconds") is None):
                raise ValueError("Terminal evidence lacks its no-tick bridge acknowledgement")
        if tick["backend_frame"] != evidence["frame"]:
            raise ValueError("Trace backend frame differs from evidence frame")
        if tick["state"] != measured.get("state"):
            raise ValueError("Trace state differs from measured CARLA state")
        if tick["command"] != measured.get("command"):
            raise ValueError("Trace command differs from acknowledged CARLA command")
        for name, value in evidence.items():
            if name == "request_id" and final_ack is not None:
                continue
            if name == "terminal_sensor_drain_seconds" and final_ack is not None:
                if final_ack.get(name) != value:
                    raise ValueError("Terminal sensor drain differs from bridge acknowledgement")
                continue
            if name not in measured or measured[name] != value:
                raise ValueError(f"Trace backend evidence differs from bridge response: {name}")
        if evidence["imu_frame"] != evidence["frame"]:
            raise ValueError("IMU heartbeat does not match CARLA frame")
        episode = evidence["episode_id"]
        if episode in previous:
            old_frame, old_time = previous[episode]
            if evidence["frame"] != old_frame + 1:
                raise ValueError("CARLA policy frames are not contiguous")
            if abs(tick["state"]["time"] - old_time - 0.05) >= 1e-6:
                raise ValueError("Measured CARLA clock did not advance by 0.05 seconds")
        previous[episode] = evidence["frame"], tick["state"]["time"]
        episode_ids.add(episode)
        used.add(key)
        count += 1
    return count


def audit(batch: Path, method: str) -> dict:
    batch = Path(batch).resolve()
    folder = batch / method
    protocol = read(batch / "protocol.json")
    actuator_observation = read(folder / "config.json")["env"].get("actuator_observation", "off")
    if method not in protocol.get("methods", []):
        raise ValueError("Method is absent from the batch protocol")

    actual, acknowledgements, owners, streams, resets = {}, {}, {}, {}, {}
    reset_positions,applied_positions={},{}
    for path in sorted((batch / "infrastructure").glob("*_responses.jsonl.gz")):
        rows, error = read_prefix(path)
        keys = set()
        for position,wrapper in enumerate(rows):
            response = wrapper.get("response", {})
            episode = response.get("episode_id")
            request_id = response.get("request_id")
            if episode is not None and request_id is not None:
                acknowledgement_key = (episode, request_id)
                if acknowledgement_key in acknowledgements and acknowledgements[acknowledgement_key] != response:
                    raise ValueError("Conflicting bridge acknowledgement for one request")
                acknowledgements[acknowledgement_key] = response
            if episode is not None and "reset_ticks" in response:
                if episode in resets and resets[episode] != response["reset_ticks"]:
                    raise ValueError("Conflicting reset tick evidence")
                resets[episode] = response["reset_ticks"]
                reset_positions.setdefault(episode,(path.name,position))
            if "vehicle_control" not in response:
                continue
            key = (episode, response["frame"])
            if key in actual:
                raise ValueError("Duplicate applied CARLA frame across bridge streams")
            actual[key] = response
            owners[key] = path.name
            applied_positions[key]=(path.name,position)
            keys.add(key)
        streams[path.name] = {
            "sha256": sha(path), "complete_rows": len(rows),
            "read_error": error, "applied_keys": keys,
        }
    if not actual:
        raise ValueError("Batch has no applied CARLA bridge responses")

    used, previous, training_episodes = set(), {}, set()
    training_frames = 0
    transition_hashes = {}
    terminations = Counter()
    last_training_terminal=None
    for path in sorted(folder.glob("transitions*.gz")):
        rows, error = read_prefix(path)
        if error:
            raise ValueError(f"Method transition stream is truncated: {path.name}: {error}")
        transition_hashes[path.name] = sha(path)
        for action in rows:
            last_training_terminal=action.get('termination_reason')
            if action.get("episode_id") is None:
                raise ValueError("Transition lacks episode identity")
            training_frames += _verify_ticks(action["trace"], actual, acknowledgements, owners, used,
                                             previous, training_episodes, actuator_observation)
            # action.episode_id is the recorder's local integer trajectory
            # index; backend_evidence.episode_id is the bridge UUID.  They are
            # intentionally different identity domains and must not be equated.
            if action.get("termination_reason"):
                terminations[action["termination_reason"]] += 1
    if not transition_hashes:
        raise ValueError("Method has no transition stream")

    summary = read(folder / "summary.json")
    if training_frames != summary["physics_steps"]:
        raise ValueError("Verified training frames differ from summary")
    optimization_rows, optimization_error = read_prefix(folder / "optimization.jsonl")
    evaluation_only=bool(protocol.get('evaluation_only'))
    if evaluation_only:
        prior=protocol.get('recovery',{}).get(method)
        if (optimization_error or optimization_rows or not prior or
                any(summary[k]!=0 for k in ('physics_steps','actions','optimization_phases','optimizer_steps'))):
            raise ValueError('Evaluation-only phase contains training or lacks source provenance')
        origin=batch/prior['checkpoint']
        for name in ('model.zip','normalization.pkl'):
            if sha(folder/name)!=sha(origin/name):
                raise ValueError('Evaluation-only phase changed saved learning bytes')
        if read(folder/'config.json')['env']!=read(origin/'config.json')['env']:
            raise ValueError('Evaluation-only phase changed the task/controller')
    elif optimization_error or not optimization_rows:
        raise ValueError("Optimization log is missing or unreadable")
    if not evaluation_only:
        last_optimization = optimization_rows[-1]
        for name in ("physics_steps", "optimization_phases", "optimizer_steps"):
            if last_optimization[name] != summary[name]:
                raise ValueError(f"Last optimization record differs from summary: {name}")

    latest = read(folder / "latest_checkpoint.json")
    checkpoint = folder / latest["folder"]
    metadata = read(checkpoint / "metadata.json")
    for name in ("physics_steps", "lineage_control_steps",
                 "charged_control_steps_lower_bound", "optimizer_steps"):
        if metadata[name] != summary[name]:
            raise ValueError(f"Final checkpoint differs from summary: {name}")
    for filename, key in (("model.zip", "model_sha256"),
                          ("normalization.pkl", "normalization_sha256"),
                          ("resume_state.pkl", "resume_state_sha256")):
        if sha(checkpoint / filename) != metadata[key]:
            raise ValueError(f"Final checkpoint hash mismatch: {filename}")

    prior = protocol.get("recovery", {}).get(method, {})
    inherited_lineage = prior.get("lineage_controls", 0)
    inherited_cost = prior.get("consumed_controls_lower_bound", 0)
    inherited_optimizer = prior.get("retained_optimizer_steps", 0)
    if summary["lineage_control_steps"] != inherited_lineage + training_frames:
        raise ValueError("Policy lineage accounting is inconsistent")
    if summary["charged_control_steps_lower_bound"] != inherited_cost + training_frames:
        raise ValueError("Charged control accounting is inconsistent")
    if summary["inherited_controls_not_in_checkpoint"] != inherited_cost - inherited_lineage:
        raise ValueError("Lost-parent control accounting is inconsistent")

    evaluation = read(folder / "evaluation.json")
    for filename, key in (("model.zip", "model_sha256"),
                          ("normalization.pkl", "normalization_sha256")):
        if sha(folder / filename) != evaluation[key]:
            raise ValueError(f"Evaluated endpoint hash mismatch: {filename}")
    evaluation_path = folder / "evaluation_trace.jsonl.gz"
    evaluation_rows, evaluation_error = read_prefix(evaluation_path)
    if evaluation_error:
        raise ValueError("Evaluation trace is truncated: " + evaluation_error)
    evaluation_cases, evaluation_episodes = {}, set()
    evaluation_frames = 0
    for row in evaluation_rows:
        info = row["info"]
        case = evaluation_cases.setdefault(row["case_id"],
                                           {"frames": 0, "actions": 0, "terminal": None})
        if case["terminal"] is not None:
            raise ValueError("Evaluation contains controls after termination")
        count = _verify_ticks(info["trace"], actual, acknowledgements, owners, used, previous,
                              evaluation_episodes, actuator_observation)
        if actuator_observation != "off":
            verify_initial_actuator(info['initial_scenario'], info['trace'][0]['backend_evidence']['episode_id'], acknowledgements)
        case["frames"] += count
        case["actions"] += 1
        evaluation_frames += count
        if info.get("termination_reason"):
            case["terminal"] = info["episode_metrics"]
    expected_cases = {episode["case_id"]: episode for episode in evaluation["episodes"]}
    if protocol.get('native_case_manifest'):
        from navstack.native_cases import registered_native_cases, verify_registered_native_begin
        inventory = set(range(len(registered_native_cases(protocol,'evaluation'))))
        if protocol.get('native_validation') and evaluation.get('partition')!='validation':
            raise ValueError('Native validation evaluation lacks its explicit partition binding')
        if set(expected_cases) != inventory:
            raise ValueError('Evaluation requires every registered development condition')
        verified_case_resets = set()
        case_episode_ids = {}
        for row in evaluation_rows:
            for tick in row['info']['trace']:
                episode = tick['backend_evidence']['episode_id']
                if case_episode_ids.setdefault(row['case_id'],episode) != episode:
                    raise ValueError('One registered evaluation condition contains multiple episodes')
                key = (row['case_id'],episode)
                if key not in verified_case_resets:
                    begin = next((r for r in acknowledgements.values() if r.get('episode_id') == episode and 'reset_ticks' in r),None)
                    if begin is None: raise ValueError('Missing registered native reset acknowledgement')
                    verify_registered_native_begin(begin,protocol,row['case_id'],partition='evaluation')
                    verified_case_resets.add(key)
    elif evaluation_only and (len(protocol.get('routes',[]))!=4 or set(expected_cases)!=set(range(4))):
        raise ValueError('Evaluation-only recovery requires all four registered cases')
    if set(evaluation_cases) != set(expected_cases):
        raise ValueError("Evaluation trace cases differ from evaluation summary")
    for case_id, episode in expected_cases.items():
        case = evaluation_cases[case_id]
        expected_terminal = {key: value for key, value in episode.items() if key != "case_id"}
        if case["terminal"] != expected_terminal:
            raise ValueError("Evaluation terminal metrics differ from trace")
        if (case["frames"], case["actions"]) != (episode["physics_steps"], episode["actions"]):
            raise ValueError("Evaluation frame/action counts differ from summary")

    matched_streams = {owners[key] for key in used}
    if len(matched_streams) != 1:
        raise ValueError("One method must map to exactly one bridge response stream")
    stream_name = next(iter(matched_streams))
    stream = streams[stream_name]
    if stream["read_error"]:
        raise ValueError("Method bridge response stream is truncated: " + stream["read_error"])
    if stream["applied_keys"] != used:
        raise ValueError("Method bridge stream has unaccounted applied controls")
    # DummyVecEnv immediately resets a terminal last rollout transition. The
    # new episode may receive no action before frozen evaluation starts. Charge
    # its actual reset ticks, but accept only this single, ordered terminal tail.
    idle_resets={e for e,(owner,_) in reset_positions.items() if owner==stream_name}-training_episodes-evaluation_episodes
    if idle_resets:
        last_train=max((applied_positions[k][1] for k in used if k[0] in training_episodes),default=-1)
        first_eval=min(reset_positions[e][1] for e in evaluation_episodes)
        if (evaluation_only or len(idle_resets)!=1 or not last_training_terminal
                or any(not last_train<reset_positions[e][1]<first_eval for e in idle_resets)):
            raise ValueError('Unaccounted reset is not one terminal training auto-reset before evaluation')
    if actuator_observation != "off":
        from navstack.carla_actuation import actuator_state_values
        for episode in training_episodes | evaluation_episodes | idle_resets:
            begin = next((r for r in acknowledgements.values() if r.get('episode_id') == episode and 'reset_ticks' in r), None)
            if begin is None:
                raise ValueError("Missing actuator reset acknowledgement")
            actuator_state_values(begin)
    training_reset_ticks = sum(resets.get(episode, 0) for episode in training_episodes | idle_resets)
    if protocol.get('native_hazard') or protocol.get('native_case_manifest'):
        from navstack.native_cases import verify_registered_native_begin
        begins={r['episode_id']:r for r in acknowledgements.values() if 'reset_ticks' in r}
        registered={}
        for episode in training_episodes | evaluation_episodes | idle_resets:
            begin=begins.get(episode)
            if begin is None:raise ValueError('Missing native hazard reset acknowledgement')
            verify_registered_native_begin(begin,protocol,partition='evaluation' if episode in evaluation_episodes else 'training')
            registered[episode]=[(a['actor_id'],a['actor_type']) for a in begin.get('native_dynamic_actors',[])]
        for (episode,_),response in actual.items():
            if episode in registered and [(a['actor_id'],a['actor_type']) for a in response.get('native_dynamic_actors',[])]!=registered[episode]:
                raise ValueError('Native hazard actor identity changes within measured episode')
    evaluation_reset_ticks = sum(resets.get(episode, 0) for episode in evaluation_episodes)
    if summary["world_ticks"] != training_frames + training_reset_ticks:
        raise ValueError("Training world-tick accounting differs from bridge resets")
    if evaluation["world_ticks"] != evaluation_frames + evaluation_reset_ticks:
        raise ValueError("Evaluation world-tick accounting differs from bridge resets")

    return {
        "schema_version": 1,
        "scope": "Per-method raw CARLA verification; every training/evaluation control is bound to one measured bridge frame.",
        "batch": batch.name,
        "method": method,
        "protocol_sha256": sha(batch / "protocol.json"),
        "config_sha256": sha(folder / "config.json"),
        "summary_sha256": sha(folder / "summary.json"),
        "evaluation_sha256": sha(folder / "evaluation.json"),
        "evaluation_trace_sha256": sha(evaluation_path),
        "optimization_sha256": sha(folder / "optimization.jsonl"),
        "transition_hashes": transition_hashes,
        "bridge_response_stream": stream_name,
        "bridge_response_sha256": stream["sha256"],
        "bridge_complete_rows": stream["complete_rows"],
        "training_control_frames_verified": training_frames,
        "evaluation_control_frames_verified": evaluation_frames,
        "total_applied_controls_verified": len(used),
        "training_world_ticks_verified": summary["world_ticks"],
        "evaluation_world_ticks_verified": evaluation["world_ticks"],
        "training_termination_reasons": dict(terminations),
        "checkpoint": checkpoint.name,
        "checkpoint_metadata_sha256": sha(checkpoint / "metadata.json"),
        "lineage_controls": summary["lineage_control_steps"],
        "charged_controls_lower_bound": summary["charged_control_steps_lower_bound"],
        "segment_optimizer_steps": summary["optimizer_steps"],
        "optimizer_steps_total_lower_bound": inherited_optimizer + summary["optimizer_steps"],
        "unknown_unflushed_ancestor_tail": bool(prior.get("unknown_unflushed_parent_tail", False)),
        **({'training_idle_reset_episodes_verified':sorted(idle_resets),
            'training_idle_reset_ticks_verified':sum(resets[e] for e in idle_resets)} if idle_resets else {}),
    }


def verify_initial_actuator(initial, episode, acknowledgements):
    from navstack.carla_actuation import actuator_state_values
    begin = next((r for r in acknowledgements.values() if r.get('episode_id') == episode and 'reset_ticks' in r), None)
    if begin is None or actuator_state_values(initial.get('actuator_evidence', {})) != actuator_state_values(begin):
        raise ValueError("Initial actuator observation differs from reset acknowledgement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--method", required=True, help="Method or registered experiment job name")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.batch, arguments.method)
    output = arguments.output or arguments.batch / arguments.method / "endpoint_raw_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
