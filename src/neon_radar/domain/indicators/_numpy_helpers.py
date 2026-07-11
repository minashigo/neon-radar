"""Internal numpy utilities shared by all built-in indicators.

This module is **private** (underscore prefix) and not part of the
public domain API. It exists purely to avoid duplicating EMA / SMA /
Wilder / rolling-std math across indicator files.

Design notes
------------
* Every helper returns a ``np.ndarray`` with ``NaN`` for indices where
  the result is undefined (warm-up period). The first valid index is
  always ``period - 1``.
* SMA uses the cumulative-sum trick for O(n) total work instead of
  O(n*period) naive loop.
* Rolling std uses the ``E[X^2] - E[X]^2`` identity for the same
  reason. A ``max(variance, 0)`` clamps floating-point noise that
  could otherwise yield a negative variance for near-constant series.
* EMA and Wilder's smoothing are both O(n) but use a small Python
  loop. For our series length (≤ 1500 candles) this is well under
  1 ms and avoids a numba/Cython dependency.
"""

from __future__ import annotations

import numpy as np


def sma_array(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Returns an array of the same length as ``values``. The first
    ``period - 1`` elements are ``NaN``; from ``period - 1`` onward
    the value is the mean of the trailing ``period`` inputs.

    Leading ``NaN`` values in the input are tolerated: the first
    contiguous run of NaN is dropped before the rolling computation
    starts. This matters for MACD, whose signal line is an EMA of an
    array whose first ``slow - 1`` values are NaN.
    """
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n < period:
        return out

    valid = ~np.isnan(values)
    if not valid.any():
        return out
    first_valid = int(np.argmax(valid))
    trimmed = values[first_valid:]
    n_trimmed = trimmed.size
    if n_trimmed < period:
        return out

    cumsum = np.cumsum(trimmed, dtype=float)
    out[first_valid + period - 1] = cumsum[period - 1] / period
    if n_trimmed > period:
        tail_start = first_valid + period
        out[tail_start:] = (cumsum[period:] - cumsum[: n_trimmed - period]) / period
    return out


def ema_array(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average, SMA-seeded.

    The first valid value is the SMA of the first ``period`` non-NaN
    inputs; subsequent values use
    ``EMA[i] = alpha * values[i] + (1 - alpha) * EMA[i-1]`` with
    ``alpha = 2 / (period + 1)``.

    Leading ``NaN`` values are dropped before the seed is computed
    (same tolerance as :func:`sma_array`).
    """
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n < period:
        return out

    valid = ~np.isnan(values)
    if not valid.any():
        return out
    first_valid = int(np.argmax(valid))
    trimmed = values[first_valid:]
    n_trimmed = trimmed.size
    if n_trimmed < period:
        return out

    alpha = 2.0 / (period + 1)
    seed_idx = first_valid + period - 1
    out[seed_idx] = float(np.mean(trimmed[:period]))
    for i in range(seed_idx + 1, n):
        v = values[i]
        if np.isnan(v):
            # Pass-through NaN: output is NaN again. Subsequent
            # non-NaN values will resume the recursive EMA from the
            # most recent valid output (handled below by the "trim"
            # logic at the start of the next non-NaN region).
            out[i] = np.nan
        else:
            # If the previous output was NaN, restart the recursion
            # with this value as the seed.
            if np.isnan(out[i - 1]):
                out[i] = v
            else:
                out[i] = alpha * v + (1.0 - alpha) * out[i - 1]
    return out


def wilder_array(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing — used by RSI and ATR.

    Equivalent to EMA with ``alpha = 1/period``. Recurrence:
    ``out[i] = out[i-1] + (values[i] - out[i-1]) / period``.

    Leading ``NaN`` values are tolerated (same convention as
    :func:`ema_array`).
    """
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n < period:
        return out

    valid = ~np.isnan(values)
    if not valid.any():
        return out
    first_valid = int(np.argmax(valid))
    trimmed = values[first_valid:]
    n_trimmed = trimmed.size
    if n_trimmed < period:
        return out

    seed_idx = first_valid + period - 1
    out[seed_idx] = float(np.mean(trimmed[:period]))
    for i in range(seed_idx + 1, n):
        v = values[i]
        if np.isnan(v):
            out[i] = np.nan
        elif np.isnan(out[i - 1]):
            out[i] = v
        else:
            out[i] = out[i - 1] + (v - out[i - 1]) / period
    return out


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling **population** standard deviation.

    Uses the identity ``Var(X) = E[X^2] - E[X]^2`` for O(n) cost.
    Negative variances from floating-point noise are clamped to zero.
    """
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n < period:
        return out
    cumsum = np.cumsum(values, dtype=float)
    cumsum_sq = np.cumsum(values * values, dtype=float)
    sums = np.empty(n - period + 1, dtype=float)
    sums[0] = cumsum[period - 1]
    sums[1:] = cumsum[period:] - cumsum[: n - period]
    sums_sq = np.empty(n - period + 1, dtype=float)
    sums_sq[0] = cumsum_sq[period - 1]
    sums_sq[1:] = cumsum_sq[period:] - cumsum_sq[: n - period]
    means = sums / period
    variance = sums_sq / period - means * means
    variance = np.maximum(variance, 0.0)
    out[period - 1:] = np.sqrt(variance)
    return out


def roc_array(values: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change (ROC).

    Formula: (current - past) / past * 100
    The first `period` elements are NaN.
    """
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n <= period:
        return out

    past = values[:-period]
    current = values[period:]

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (current - past) / past * 100.0

    out[period:] = result
    return out
