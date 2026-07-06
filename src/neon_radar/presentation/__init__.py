"""Presentation layer.

PySide6 / Qt UI code and the command-line interface. The CLI
(:mod:`neon_radar.presentation.cli`) is the first usable form of
the product. The Qt UI (Stage 6) is the second.

This is the outermost layer; it depends on everything below it but
nothing above it does.
"""

from neon_radar.presentation.main_window import MainWindow
from neon_radar.presentation.theme.neon_palette import NeonPalette
from neon_radar.presentation.widgets.chart_widget import ChartWidget
from neon_radar.presentation.widgets.detail_panel import DetailPanel
from neon_radar.presentation.widgets.ranking_table import RankingRow, RankingTable

__all__ = [
    "ChartWidget",
    "DetailPanel",
    "MainWindow",
    "NeonPalette",
    "RankingRow",
    "RankingTable",
]
