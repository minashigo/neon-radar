"""Neon dark theme for the Qt UI.

The theme follows the project's visual identity:
* dark slate background for low eye-strain during long sessions
* neon green / red / yellow as accent colours matching the bias states
* subtle blue for selected rows and focus indicators

The palette is applied through :class:`QPalette` for built-in
widgets and through ``setStyleSheet`` for the few custom styles we
need (``QTableWidget`` row colouring, header, status bar).
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


class NeonPalette:
    """Static colour constants used across the UI."""

    BG_DARK = "#1a1d23"
    BG_PANEL = "#22262e"
    BG_ROW_ALT = "#1e2127"
    BG_ROW_SELECTED = "#2a3142"
    BG_HEADER = "#2d3340"
    TEXT_PRIMARY = "#e6e8ec"
    TEXT_DIM = "#8a8f99"
    TEXT_BORDER = "#3a3f4a"

    ACCENT_BULLISH = "#00e676"  # neon green
    ACCENT_BEARISH = "#ff1744"  # neon red
    ACCENT_NEUTRAL = "#ffd600"  # neon yellow
    ACCENT_SELECTED = "#4fc3f7"  # neon blue

    @classmethod
    def build_palette(cls) -> QPalette:
        """Build a :class:`QPalette` for the application.

        Applied via ``QApplication.setPalette``.
        """
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(cls.BG_DARK))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(cls.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Base, QColor(cls.BG_PANEL))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(cls.BG_ROW_ALT))
        palette.setColor(QPalette.ColorRole.Text, QColor(cls.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Button, QColor(cls.BG_PANEL))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(cls.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(cls.ACCENT_SELECTED))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(cls.BG_DARK))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(cls.TEXT_DIM))
        return palette

    @classmethod
    def stylesheet(cls) -> str:
        """Application-wide stylesheet for custom widget styling."""
        return f"""
        QMainWindow, QWidget {{
            background-color: {cls.BG_DARK};
            color: {cls.TEXT_PRIMARY};
        }}

        QStatusBar {{
            background-color: {cls.BG_PANEL};
            color: {cls.TEXT_DIM};
            border-top: 1px solid {cls.TEXT_BORDER};
        }}

        QHeaderView::section {{
            background-color: {cls.BG_HEADER};
            color: {cls.TEXT_PRIMARY};
            padding: 6px 8px;
            border: none;
            border-right: 1px solid {cls.TEXT_BORDER};
            border-bottom: 1px solid {cls.TEXT_BORDER};
            font-weight: bold;
        }}

        QTableWidget {{
            background-color: {cls.BG_PANEL};
            gridline-color: {cls.TEXT_BORDER};
            selection-background-color: {cls.BG_ROW_SELECTED};
            selection-color: {cls.TEXT_PRIMARY};
        }}

        QTableWidget::item {{
            padding: 8px;
        }}

        QSplitter::handle {{
            background-color: {cls.TEXT_BORDER};
        }}

        QLabel#score_value_bullish {{
            color: {cls.ACCENT_BULLISH};
            font-size: 22px;
            font-weight: bold;
        }}
        QLabel#score_value_bearish {{
            color: {cls.ACCENT_BEARISH};
            font-size: 22px;
            font-weight: bold;
        }}
        QLabel#score_value_neutral {{
            color: {cls.ACCENT_NEUTRAL};
            font-size: 22px;
            font-weight: bold;
        }}
        """
