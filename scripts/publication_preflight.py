"""Read-only checks of Git publication candidates; findings never echo secrets."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    listing = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted(set(listing.decode("utf-8").split("\0")) - {""})
    patterns = {
        "concrete_windows_home": re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]"),
        "concrete_unix_home": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
        "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
    }
    findings = []
    checked = 0
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        checked += 1
        if path.stat().st_size > 5 * 1024**2:
            findings.append({"file": name, "kind": "larger_than_5MiB"})
        if any(part in {".venv", ".carla-python37", "__pycache__"} for part in Path(name).parts):
            findings.append({"file": name, "kind": "runtime_or_cache"})
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for kind, pattern in patterns.items():
            if pattern.search(content):
                findings.append({"file": name, "kind": kind})
    report = {"scope": "Git tracked and nonignored untracked files; heuristic scan, not a security proof",
              "files_checked": checked, "findings": findings,
              "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files_checked": checked, "findings": findings}))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
