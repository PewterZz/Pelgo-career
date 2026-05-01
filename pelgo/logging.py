"""Structured JSON-line logger for Pelgo agent runs.

Each log call emits one JSON object to stderr:
  {"ts": <iso>, "level": "info|warn|error", "event": "...", ...fields}

Usage:
  from pelgo.logging import log
  log("tool_call", job_id=job_id, tool="extract_jd_requirements", status="success", latency_ms=340)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


def log(event: str, level: str = "info", **fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(record, default=str), file=sys.stderr, flush=True)
