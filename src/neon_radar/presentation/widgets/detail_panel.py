"""Detail panel — secondary view shown below the ranking table.

Shows the currently-selected symbol's score, confidence, and per-factor
breakdown. Acts as a "drill-down" before opening the chart (which is
out of scope for Stage 6A).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from neon_radar.domain.enums import Bias
from neon_radar.presentation.theme.neon_palette import NeonPalette

if TYPE_CHECKING:
    from neon_radar.domain.scoring.value_objects import (
        AnalysisResult,
        FactorBreakdown,
    )
    pass


class DetailPanel(QFrame):
    """Compact score breakdown for the selected symbol."""

    #: Emitted when the user clicks "View Chart" (wired in Stage 6B).
    view_chart_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("detailPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_ui()
        self.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_result(self, symbol: str, result: AnalysisResult) -> None:
        """Populate the panel with the latest analysis for ``symbol``."""
        self._title.setText(f"SELECTED: {symbol}")
        score = result.score
        bias = score.bias
        self._score_value.setText(f"{score.value:+.2f}")
        self._score_value.setObjectName(
            "score_value_" + bias.value.lower()
        )
        # Re-apply style by forcing stylesheet refresh.
        self._score_value.style().unpolish(self._score_value)
        self._score_value.style().polish(self._score_value)
        self._conf_value.setText(f"{score.confidence:.2f}")
        self._bias_value.setText(self._bias_label(bias))
        self._bias_value.setStyleSheet(
            f"color: {self._bias_color(bias)}; font-weight: bold;"
        )
        self._signals_value.setText(str(len(result.signals)))
        self._view_chart_btn.setEnabled(True)

        # Breakdown table.
        rows = list(result.breakdown())
        self._breakdown.setRowCount(len(rows))
        for i, b in enumerate(rows):
            self._set_breakdown_cell(i, 0, b.factor)
            self._set_breakdown_cell(
                i, 1, f"{b.contribution:+.2f}",
                color=self._contrib_color(b),
                bold=True,
                align=Qt.AlignmentFlag.AlignRight,
            )
            self._set_breakdown_cell(i, 2, f"{b.value:+.2f}", align=Qt.AlignmentFlag.AlignRight)
            self._set_breakdown_cell(i, 3, f"{b.weight:.2f}", align=Qt.AlignmentFlag.AlignRight)
            self._set_breakdown_cell(i, 4, f"{b.confidence:.2f}", align=Qt.AlignmentFlag.AlignRight)
            self._set_breakdown_cell(i, 5, b.description)

    def clear(self) -> None:
        """Reset to empty state."""
        self._title.setText("SELECTED: —")
        self._score_value.setText("—")
        self._score_value.setObjectName("score_value_neutral")
        self._score_value.style().unpolish(self._score_value)
        self._score_value.style().polish(self._score_value)
        self._conf_value.setText("—")
        self._bias_value.setText("—")
        self._bias_value.setStyleSheet("")
        self._signals_value.setText("—")
        self._breakdown.setRowCount(0)
        self._view_chart_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        # Title + score row.
        top_row = QHBoxLayout()
        self._title = QLabel("SELECTED: —")
        title_font = self._title.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        self._title.setFont(title_font)

        self._view_chart_btn = QPushButton("View Chart →")
        self._view_chart_btn.setEnabled(False)
        self._view_chart_btn.clicked.connect(self.view_chart_requested)

        top_row.addWidget(self._title)
        top_row.addStretch()
        top_row.addWidget(self._view_chart_btn)
        layout.addLayout(top_row)

        # Score / conf / bias / signals summary.
        summary_grid = QGridLayout()
        summary_grid.setHorizontalSpacing(24)

        score_lbl = self._make_label("SCORE", dim=True)
        conf_lbl = self._make_label("CONF", dim=True)
        bias_lbl = self._make_label("BIAS", dim=True)
        signals_lbl = self._make_label("SIGNALS", dim=True)

        self._score_value = self._make_value_label()
        self._conf_value = self._make_value_label()
        self._bias_value = self._make_value_label()
        self._signals_value = self._make_value_label()

        summary_grid.addWidget(score_lbl, 0, 0)
        summary_grid.addWidget(self._score_value, 1, 0)
        summary_grid.addWidget(conf_lbl, 0, 1)
        summary_grid.addWidget(self._conf_value, 1, 1)
        summary_grid.addWidget(bias_lbl, 0, 2)
        summary_grid.addWidget(self._bias_value, 1, 2)
        summary_grid.addWidget(signals_lbl, 0, 3)
        summary_grid.addWidget(self._signals_value, 1, 3)

        layout.addLayout(summary_grid)

        # Breakdown table.
        self._breakdown = QTableWidget(0, 6)
        self._breakdown.setHorizontalHeaderLabels(
            ["Factor", "Contrib", "Value", "Weight", "Conf", "Description"]
        )
        bd_header = self._breakdown.horizontalHeader()
        bd_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        bd_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        bd_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        bd_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        bd_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        bd_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._breakdown.verticalHeader().setVisible(False)
        self._breakdown.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._breakdown.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._breakdown.setShowGrid(False)
        layout.addWidget(self._breakdown, stretch=1)

    @staticmethod
    def _make_label(text: str, *, dim: bool = False) -> QLabel:
        lbl = QLabel(text)
        if dim:
            lbl.setStyleSheet(f"color: {NeonPalette.TEXT_DIM}; font-size: 10px;")
        return lbl

    @staticmethod
    def _make_value_label() -> QLabel:
        lbl = QLabel("—")
        font = lbl.font()
        font.setPointSize(font.pointSize() + 2)
        lbl.setFont(font)
        return lbl

    def _set_breakdown_cell(
        self,
        row: int,
        col: int,
        text: str,
        *,
        color: str | None = None,
        bold: bool = False,
        align: Qt.AlignmentFlag | None = None,
    ) -> None:
        item = QTableWidgetItem(text)
        if color:
            from PySide6.QtGui import QBrush, QColor

            item.setForeground(QBrush(QColor(color)))
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        if align:
            item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        self._breakdown.setItem(row, col, item)

    @staticmethod
    def _bias_label(bias: Bias) -> str:
        if bias is Bias.BULLISH:
            return "BULLISH ▲"
        if bias is Bias.BEARISH:
            return "BEARISH ▼"
        return "NEUTRAL →"

    @staticmethod
    def _bias_color(bias: Bias) -> str:
        if bias is Bias.BULLISH:
            return NeonPalette.ACCENT_BULLISH
        if bias is Bias.BEARISH:
            return NeonPalette.ACCENT_BEARISH
        return NeonPalette.ACCENT_NEUTRAL

    @staticmethod
    def _contrib_color(breakdown: FactorBreakdown) -> str:
        if breakdown.is_bullish:
            return NeonPalette.ACCENT_BULLISH
        if breakdown.is_bearish:
            return NeonPalette.ACCENT_BEARISH
        return NeonPalette.TEXT_DIM
