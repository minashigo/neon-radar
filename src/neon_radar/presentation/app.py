"""Qt UI entry point.

Run with::

    python -m neon_radar.presentation.app

or::

    neon-radar-app
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from neon_radar.config.loader import ConfigLoader
from neon_radar.config.scoring_loader import load_rules
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.presentation.main_window import MainWindow
from neon_radar.presentation.theme.neon_palette import NeonPalette


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    config_path = Path(args.config)
    scoring_path = Path(args.scoring)

    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    if not scoring_path.is_file():
        print(f"Scoring rules not found: {scoring_path}", file=sys.stderr)
        return 2

    config = ConfigLoader(config_path).load()
    rules = load_rules(scoring_path)
    # Need ScoringRulesConfig for min_confidence in MainWindow.
    import json

    from neon_radar.config.loader import _strip_meta

    raw = json.loads(scoring_path.read_text(encoding="utf-8"))
    scoring_cfg_dict = _strip_meta(raw)
    scoring_cfg = ScoringRulesConfig.model_validate(scoring_cfg_dict)

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Neon Radar")
    app.setOrganizationName("Neon Radar")
    app.setPalette(NeonPalette.build_palette())
    app.setStyleSheet(NeonPalette.stylesheet())

    window = MainWindow(
        config=config,
        scoring_config=scoring_cfg,
        rules=tuple(rules),
        refresh_seconds=config.refresh.interval_seconds,
    )
    window.show()
    return app.exec()


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="neon-radar-app",
        description="Neon Radar — Binance Futures scoring UI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the application config (default: config.json)",
    )
    parser.add_argument(
        "--scoring",
        type=Path,
        default=Path("scoring_rules.json"),
        help="Path to the scoring rules config (default: scoring_rules.json)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
