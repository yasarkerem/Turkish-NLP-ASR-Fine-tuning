"""Run logging: every training/eval/benchmark run writes one JSON under runs/.

The JSON records the exact config, CLI arguments, environment, timings and
final metrics — or the error, when a run fails. RESULTS.md may only cite
numbers that exist in one of these files.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return None


def _environment() -> dict:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


class RunLog:
    """Context-manager style logger for a single run."""

    def __init__(self, runs_dir: Path, kind: str, name: str, config: dict):
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{kind}_{name}"
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self.payload: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": kind,
            "name": name,
            "argv": sys.argv,
            "config": config,
            "environment": _environment(),
            "git_commit": _git_commit(),
            "started_utc": stamp,
            "status": "running",
            "metrics": {},
            "artifacts": {},
        }
        self.flush()

    @property
    def json_path(self) -> Path:
        return self.dir / "run.json"

    def flush(self) -> None:
        self.payload["wall_seconds"] = round(time.monotonic() - self._t0, 2)
        with open(self.json_path, "w") as f:
            json.dump(self.payload, f, indent=2, ensure_ascii=False, default=str)

    def log_metrics(self, **metrics: Any) -> None:
        self.payload["metrics"].update(metrics)
        self.flush()

    def log_artifact(self, key: str, path: str | Path) -> None:
        self.payload["artifacts"][key] = str(path)
        self.flush()

    def finish(self, status: str = "completed") -> None:
        self.payload["status"] = status
        self.flush()

    def fail(self, exc: BaseException) -> None:
        self.payload["status"] = "failed"
        self.payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.flush()
