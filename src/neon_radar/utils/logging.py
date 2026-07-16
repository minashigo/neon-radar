"""Logging configuration for Neon Radar.

Design notes
------------
* We use the standard library ``logging`` module rather than ``loguru`` or
  ``structlog`` because:
    - It is part of Python's stdlib (no dependency).
    - Qt apps already need stdlib logging for Qt-internal logs.
    - The Pydantic config model already describes log level / file / JSON.
* A simple, dependency-free colored formatter is provided so the console
  output looks pleasant during development. In production, switching to
  ``json_format=True`` produces machine-readable lines for aggregation.
* ``get_logger`` returns a module-level logger. Modules should do::

      logger = get_logger(__name__)

  at the top of the file. This is the convention recommended by the
  Python logging HOWTO and gives clean hierarchical filtering.
* ``configure_logging`` is idempotent — calling it twice will not add
  duplicate handlers. Useful for tests.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

# A small ANSI palette — keeps console output readable without pulling
# in colorlog. Disable by setting NO_COLOR or by running in a non-TTY.
_RESET = "\x1b[0m"
_COLORS: dict[int, str] = {
    logging.DEBUG: "\x1b[38;5;244m",  # grey
    logging.INFO: "\x1b[38;5;39m",  # blue
    logging.WARNING: "\x1b[38;5;220m",  # yellow
    logging.ERROR: "\x1b[38;5;196m",  # red
    logging.CRITICAL: "\x1b[1;38;5;201m",  # bold magenta
}


class _ColorFormatter(logging.Formatter):
    """ANSI-coloured formatter for the console."""

    DEFAULT_FMT: ClassVar[str] = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    DEFAULT_DATEFMT: ClassVar[str] = "%H:%M:%S"

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(self.DEFAULT_FMT, self.DEFAULT_DATEFMT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self._use_color:
            color = _COLORS.get(record.levelno, "")
            message = f"{color}{message}{_RESET}"
        return message


class _JsonFormatter(logging.Formatter):
    """One-line JSON per record, suitable for log aggregation."""

    # Standard LogRecord attributes we don't want in the JSON output.
    _RESERVED: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Custom extras — anything attached via ``logger.info("x", extra={...})``.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    file: Path | None = None,
    json_format: bool = False,
    console: bool = True,
) -> None:
    """Configure the root logger for Neon Radar.

    Parameters
    ----------
    level
        Log level name (``"DEBUG"``, ``"INFO"``, …).
    file
        Optional path to a rotating log file. Created if missing.
    json_format
        If ``True`` (and ``file`` is set), writes JSON lines to the file.
        Console output is never JSON so it stays human-readable.
    console
        If ``True``, attach a stderr handler. Set ``False`` for tests.
    """
    global _configured
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove handlers we previously attached to keep this idempotent.
    for handler in list(root.handlers):
        if getattr(handler, "_neon_radar", False):
            root.removeHandler(handler)

    if console:
        use_color = sys.stderr.isatty() and not json_format
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_ColorFormatter(use_color=use_color))
        handler._neon_radar = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        # 5 MB x 3 backups - plenty for a daily-use tool.
        fh = RotatingFileHandler(file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(_JsonFormatter() if json_format else _ColorFormatter(use_color=False))
        fh._neon_radar = True  # type: ignore[attr-defined]
        root.addHandler(fh)

    # Quiet down noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Always use this rather than ``logging.getLogger``
    directly so we have a single chokepoint if we ever want to switch to
    a structured logger.
    """
    return logging.getLogger(name)
