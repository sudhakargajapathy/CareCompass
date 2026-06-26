"""Audit logging utilities for security-sensitive events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from .config import get_config

logger = logging.getLogger(__name__)


def log_audit_event(
    event_type: str,
    user: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Write a structured audit event to the audit log.

    Args:
        event_type: Short identifier for the event (e.g., login_success)
        user: Username associated with the event
        success: Whether the action succeeded
        details: Additional structured details for the event
    """
    config = get_config()
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_type": event_type,
        "user": user or "anonymous",
        "success": success,
        "details": details or {},
    }

    log_path = config.AUDIT_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.error("Failed to write audit log: %s", exc)
