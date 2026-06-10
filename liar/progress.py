"""Durable progress files per the long-running-job framework."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_progress(
    out_dir: str | Path,
    activity: str,
    completed: int,
    expected: int,
    current: dict | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "activity": activity,
        "completed": completed,
        "expected": expected,
        "progress_percent": round(100.0 * completed / expected, 2) if expected else 0.0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current": current or {},
    }
    tmp = out_dir / "progress.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_dir / "progress.json")


def count_jsonl(path: str | Path) -> int:
    """Rows in a JSONL file, tolerating a corrupt final line."""
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except json.JSONDecodeError:
                break
    return n
