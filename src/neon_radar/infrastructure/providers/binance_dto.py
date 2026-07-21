"""Raw Data Transfer Objects (DTOs) for Binance Microstructure API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class BinancePremiumIndexDTO:
    """Matches Binance /fapi/v1/premiumIndex."""
    symbol: str
    markPrice: str
    indexPrice: str
    estimatedSettlePrice: str
    lastFundingRate: str
    nextFundingTime: int
    interestRate: str
    time: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinancePremiumIndexDTO:
        return cls(
            symbol=data["symbol"],
            markPrice=data["markPrice"],
            indexPrice=data["indexPrice"],
            estimatedSettlePrice=data["estimatedSettlePrice"],
            lastFundingRate=data["lastFundingRate"],
            nextFundingTime=int(data["nextFundingTime"]),
            interestRate=data["interestRate"],
            time=int(data["time"]),
        )


@dataclass(slots=True, frozen=True)
class BinanceOpenInterestDTO:
    """Matches Binance /fapi/v1/openInterest."""
    symbol: str
    openInterest: str
    time: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinanceOpenInterestDTO:
        return cls(
            symbol=data["symbol"],
            openInterest=data["openInterest"],
            time=int(data["time"]),
        )


@dataclass(slots=True, frozen=True)
class BinanceLongShortRatioDTO:
    """Matches Binance /futures/data/globalLongShortAccountRatio."""
    longShortRatio: str
    longAccount: str
    shortAccount: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinanceLongShortRatioDTO:
        return cls(
            longShortRatio=data["longShortRatio"],
            longAccount=data["longAccount"],
            shortAccount=data["shortAccount"],
            timestamp=int(data["timestamp"]),
        )


@dataclass(slots=True, frozen=True)
class BinanceTakerVolumeDTO:
    """Matches Binance /futures/data/takerlongshortRatio."""
    buySellRatio: str
    buyVol: str
    sellVol: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinanceTakerVolumeDTO:
        return cls(
            buySellRatio=data["buySellRatio"],
            buyVol=data["buyVol"],
            sellVol=data["sellVol"],
            timestamp=int(data["timestamp"]),
        )
