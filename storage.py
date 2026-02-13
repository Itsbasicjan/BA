from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_LOG_PATH = "storage.jsonl"


def append_event(event: Dict[str, Any], path: str = DEFAULT_LOG_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(path: str = DEFAULT_LOG_PATH, limit: int = 200) -> List[Dict[str, Any]]:
    if limit < 1:
        return []

    target = Path(path)
    if not target.exists():
        return []

    events: List[Dict[str, Any]] = []
    lines = target.read_text(encoding="utf-8").splitlines()
    for line in lines[-limit:]:
        payload = line.strip()
        if not payload:
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            events.append(decoded)
    return events
