"""Exhibit the remaining cross-asset OHLC ordering limitation; no live orders."""

from __future__ import annotations

import json
from pathlib import Path

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import OHLCV, Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.risk import PortfolioRiskPolicy, RiskDecision
from neon_radar.domain.trading.setup import FinalTradeSetup


def scenario(order):
    btc, eth = Symbol("BTCUSDT"), Symbol("ETHUSDT")
    portfolio = PortfolioEngine(10000)
    portfolio.open_position(
        OpenPosition(
            symbol=btc,
            direction=Bias.BULLISH,
            entry_price=100,
            quantity=10,
            position_size=1000,
            stop_loss=50,
            take_profit=300,
            opened_at=1,
        )
    )
    execution = PaperExecutionEngine(
        portfolio,
        risk_manager=RiskManager(
            PortfolioRiskPolicy(base_risk_per_trade_pct=0.01),
        ),
    )
    setup = FinalTradeSetup(
        symbol=eth,
        direction=Bias.BULLISH,
        entry=100,
        stop_loss=90,
        take_profit=120,
        confidence=0.8,
        score=0.8,
        risk_decision=RiskDecision(True),
        position_size=10,
        quote_size=1000,
        risk_amount=100,
        expected_reward=200,
    )
    assert execution.execute_setup(setup, 1)
    candles = {
        btc: OHLCV(open_time=1000, open=100, high=101, low=89, close=90, volume=1, close_time=1999),
        eth: OHLCV(
            open_time=1000, open=100, high=101, low=99, close=100, volume=1, close_time=1999
        ),
    }
    for symbol in order:
        execution.process_market_tick(symbol, candles[symbol])
    return {
        "equity": portfolio.state.account.total_capital,
        "eth_filled": any(p.symbol == eth for p in portfolio.state.positions),
    }


def main():
    result = {
        "btc_then_eth": scenario((Symbol("BTCUSDT"), Symbol("ETHUSDT"))),
        "eth_then_btc": scenario((Symbol("ETHUSDT"), Symbol("BTCUSDT"))),
    }
    # The same timestamps/prices/equity have different fills solely from iteration order.
    assert result["btc_then_eth"]["equity"] == result["eth_then_btc"]["equity"] == 9900
    assert not result["btc_then_eth"]["eth_filled"] and result["eth_then_btc"]["eth_filled"]
    (Path(__file__).resolve().parent / "order_bias.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
