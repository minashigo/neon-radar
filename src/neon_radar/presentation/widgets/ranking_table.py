"""Ranking table — primary view of Neon Radar.

Shows all configured symbols ranked by their Score, with bias,
confidence, and the contributing factor arrows.

This is the widget the user spends most time looking at. It must
stay scannable: at a glance, the user should see "which coins are
interesting right now" without having to read every column.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from neon_radar.domain.enums import Bias
from neon_radar.presentation.theme.neon_palette import NeonPalette


@dataclass(frozen=True)
class RankingRow:
    """One row of the ranking table — derived from an :class:`AnalysisResult`."""

    symbol: str
    score: float
    confidence: float
    bias: Bias
    factor_arrows: str  # short summary like "trend↑ mom↑ vol↑"


class RankingTable(QTableWidget):
    """Sortable ranking table — primary view of the Radar."""

    #: Emitted when the user selects a row.
    symbol_selected = Signal(str)

    COLUMNS = ("#", "Symbol", "Score", "Conf", "Bias", "Factors")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSortingEnabled(False)  # we sort manually by score

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self._symbol_column: dict[str, int] = {}
        self.currentItemChanged.connect(self._on_current_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_rows(self, rows: list[RankingRow]) -> None:
        """Replace all rows with the supplied list, sorted by score desc."""
        rows_sorted = sorted(rows, key=lambda r: r.score, reverse=True)
        self.setRowCount(len(rows_sorted))
        self._symbol_column.clear()

        for rank, row in enumerate(rows_sorted, start=1):
            self._set_cell(rank - 1, 0, str(rank), align=Qt.AlignmentFlag.AlignCenter)
            self._set_cell(rank - 1, 1, row.symbol)
            self._set_cell(
                rank - 1, 2,
                f"{row.score:+.2f}",
                color=self._score_color(row.score),
                bold=True,
                align=Qt.AlignmentFlag.AlignRight,
            )
            self._set_cell(
                rank - 1, 3,
                f"{row.confidence:.2f}",
                color=NeonPalette.TEXT_PRIMARY,
                align=Qt.AlignmentFlag.AlignRight,
            )
            self._set_cell(
                rank - 1, 4,
                self._bias_label(row.bias),
                color=self._bias_color(row.bias),
                bold=True,
            )
            self._set_cell(rank - 1, 5, row.factor_arrows)
            self._symbol_column[row.symbol] = rank - 1

        # Default selection: top row.
        if self.rowCount() > 0 and self.currentRow() < 0:
            self.selectRow(0)

    def select_symbol(self, symbol: str) -> None:
        """Select the row corresponding to ``symbol``. No-op if absent."""
        row = self._symbol_column.get(symbol)
        if row is not None:
            self.selectRow(row)

    def selected_symbol(self) -> str | None:
        """Return the currently selected symbol, or ``None``."""
        row = self.currentRow()
        if row < 0:
            return None
        item = self.item(row, 1)  # symbol column
        return item.text() if item else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_cell(
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
            item.setForeground(QBrush(QColor(color)))
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        if align:
            item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        self.setItem(row, col, item)

    @staticmethod
    def _score_color(score: float) -> str:
        if score > 0.2:
            return NeonPalette.ACCENT_BULLISH
        if score < -0.2:
            return NeonPalette.ACCENT_BEARISH
        return NeonPalette.ACCENT_NEUTRAL

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

    def _on_current_changed(self, current, _previous) -> None:
        if current is None:
            return
        symbol_item = self.item(current.row(), 1)
        if symbol_item is not None:
            self.symbol_selected.emit(symbol_item.text())
