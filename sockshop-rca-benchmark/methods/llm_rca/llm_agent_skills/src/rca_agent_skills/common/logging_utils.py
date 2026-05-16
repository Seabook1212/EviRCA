from __future__ import annotations

import json
import logging
import os
from typing import Any


def configure_logging(level: str | None = None) -> None:
    log_level = (level or os.environ.get("RCA_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_json(logger: logging.Logger, prefix: str, payload: Any, level: int = logging.INFO) -> None:
    logger.log(level, "%s%s", prefix, json.dumps(payload, ensure_ascii=False, default=str, indent=2))
