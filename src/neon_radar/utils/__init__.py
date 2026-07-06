"""Shared utilities.

Anything that is genuinely cross-cutting (logging setup, async helpers,
formatting) lives here. This package must not import from any other
Neon Radar layer — it sits at the bottom of the dependency graph.
"""

from neon_radar.utils.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
