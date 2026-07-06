"""Tests for FundingRate and OpenInterest."""

from __future__ import annotations

import pytest

from neon_radar.domain.funding import FundingRate, OpenInterest


class TestFundingRate:
    def test_basic(self) -> None:
        fr = FundingRate(symbol="BTCUSDT", rate=0.0001)
        assert fr.symbol == "BTCUSDT"
        assert fr.rate == pytest.approx(0.0001)
        assert fr.is_positive is True
        assert fr.mark_price is None

    def test_negative_rate(self) -> None:
        fr = FundingRate(symbol="BTCUSDT", rate=-0.0002)
        assert fr.is_positive is False

    def test_zero_rate(self) -> None:
        fr = FundingRate(symbol="BTCUSDT", rate=0.0)
        assert fr.is_positive is False

    def test_string_symbol_normalised(self) -> None:
        fr = FundingRate(symbol="btcusdt", rate=0.0)
        assert fr.symbol == "BTCUSDT"

    def test_annualised_pct(self) -> None:
        # 0.01% per 8h -> approx 10.95% APR.
        fr = FundingRate(symbol="BTCUSDT", rate=0.0001)
        assert fr.annualized_pct == pytest.approx(10.95, rel=1e-3)

    def test_negative_annualised(self) -> None:
        fr = FundingRate(symbol="BTCUSDT", rate=-0.0001)
        assert fr.annualized_pct == pytest.approx(-10.95, rel=1e-3)

    def test_immutable(self) -> None:
        fr = FundingRate(symbol="BTCUSDT", rate=0.0001)
        with pytest.raises((AttributeError, Exception)):
            fr.rate = 0.5  # type: ignore[misc]


class TestOpenInterest:
    def test_basic(self) -> None:
        oi = OpenInterest(symbol="BTCUSDT", value=50_000.5, value_quote=1_500_000_000.0)
        assert oi.symbol == "BTCUSDT"
        assert oi.value == 50_000.5
        assert oi.value_quote == 1_500_000_000.0

    def test_only_base(self) -> None:
        oi = OpenInterest(symbol="BTCUSDT", value=1000.0)
        assert oi.value_quote is None

    def test_rejects_negative_value(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OpenInterest(symbol="BTCUSDT", value=-1.0)

    def test_rejects_negative_quote(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OpenInterest(symbol="BTCUSDT", value=1.0, value_quote=-1.0)

    def test_string_symbol_normalised(self) -> None:
        oi = OpenInterest(symbol="btcusdt", value=1.0)
        assert oi.symbol == "BTCUSDT"
