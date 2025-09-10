from __future__ import annotations
import logging
import os
from typing import Optional


def _default_level() -> int:
    lvl = os.getenv("SEEG_LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


class _SimpleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Attach structured extras if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            return f"{base} | extra={record.extra}"
        return base


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name or "seeg")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = os.getenv("SEEG_LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(_SimpleFormatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(_default_level())
        logger.propagate = False
    return logger
