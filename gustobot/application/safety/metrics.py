"""JSONL metrics writer for safety defense events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from gustobot.config import settings

from .schemas import SafetyMetricEvent


class SafetyMetricsWriter:
    """Append safety events to a local JSONL file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or settings.SAFETY_METRICS_PATH)

    def write(self, event: SafetyMetricEvent) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - metrics must not break chat
            logger.warning("Failed to write safety metric event: {}", exc)
