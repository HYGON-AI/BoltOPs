"""Logging helpers for bolt_ops."""

from __future__ import annotations

import logging
import os


def get_logger(name: str = "bolt_ops") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("BOLT_OPS_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
