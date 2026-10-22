"""Aggregate logged training-session elapsed times, excluding queue and evaluation waits."""


def session_training_seconds(rows):
    sessions = []
    for row in rows:
        elapsed = row.get("session_wall_seconds")
        if elapsed is None:
            continue
        if "epoch" not in row:
            raise ValueError(
                "Cannot recover session boundaries without epoch timestamps"
            )
        started = row["epoch"] - elapsed
        if not sessions or abs(started - sessions[-1]["started_epoch"]) > 0.1:
            sessions.append(
                {"started_epoch": started, "logged_elapsed_seconds": elapsed}
            )
        else:
            sessions[-1]["logged_elapsed_seconds"] = max(
                sessions[-1]["logged_elapsed_seconds"], elapsed
            )
    return {
        "seconds": sum(s["logged_elapsed_seconds"] for s in sessions),
        "sessions": sessions,
        "scope": "Time through each session's latest logged optimizer update, including in-session checkpointing; excludes queue waits, model loading before training, later final save and evaluation.",
    }
