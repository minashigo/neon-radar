"""Score aggregation — turn a list of :class:`Signal` into a :class:`Score`.

Decoupled design
----------------
After Stage 5, Score and Confidence are **independent** axes:

* ``value`` (and ``long_score``, ``short_score``) are derived from
  ``signal.value x signal.weight`` only.
* ``confidence`` is the weighted average of ``signal.confidence``,
  computed independently.

Why decouple?
^^^^^^^^^^^^^
The previous formula ``contribution = value * confidence * weight``
mixed direction strength with certainty. A high-value signal with low
confidence contributed the same as a low-value signal with high
confidence, hiding two distinct messages. With decoupled axes the user
can see "bullish but low confidence" vs "weak bullish but high
agreement" and filter accordingly.

The ``min_confidence`` filter prevents an extremely-low-confidence
signal from contributing its possibly-stale value to the score.
Signals below the threshold are dropped before aggregation.
"""

from __future__ import annotations

from neon_radar.domain.scoring.value_objects import Score, Signal


def aggregate(
    signals: tuple[Signal, ...],
    *,
    min_confidence: float = 0.0,
) -> Score:
    """Combine ``signals`` into a single :class:`Score`.

    Parameters
    ----------
    signals
        Per-factor contributions from the rules engine.
    min_confidence
        Signals below this confidence are dropped before aggregation.
        Use this to ignore noisy votes (e.g. ``min_confidence=0.3``
        excludes volatility-filter "low vol" signals that may mislead).

    Returns
    -------
    Score
        ``value`` reflects direction and magnitude, weighted by rule
        importance. ``confidence`` is the weighted average of signal
        confidences, computed independently.
    """
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

    filtered = tuple(s for s in signals if s.confidence >= min_confidence)
    n = len(filtered)

    if n == 0:
        return Score(
            value=0.0,
            confidence=0.0,
            long_score=0.0,
            short_score=0.0,
            contributing_signals=0,
        )

    # We separate denominators: directional normalization uses only
    # signals with non-zero `value`, while confidence remains the
    # weighted average across all (filtered) signals. This preserves
    # the semantic that confidence-only modifiers don't dilute
    # directional magnitude; they only affect `Score.confidence`.
    total_weight_all = 0.0
    directional_weight = 0.0
    long_sum = 0.0
    short_sum = 0.0
    confidence_sum = 0.0

    for sig in filtered:
        w = sig.weight
        total_weight_all += w
        contribution = sig.value * w  # NO confidence multiplier here
        if sig.value > 0:
            long_sum += contribution
            directional_weight += w
        elif sig.value < 0:
            short_sum += -contribution  # store as positive magnitude
            directional_weight += w
        confidence_sum += w * sig.confidence

    # If there is no overall weight (all weights zero) return zeros.
    if total_weight_all == 0.0:
        return Score(
            value=0.0,
            confidence=0.0,
            long_score=0.0,
            short_score=0.0,
            contributing_signals=n,
        )

    # Compute confidence as the weighted average across all filtered signals
    confidence = confidence_sum / total_weight_all

    # Normalize directional scores using only the directional weight.
    if directional_weight == 0.0:
        long_score = 0.0
        short_score = 0.0
        value = 0.0
    else:
        long_score = long_sum / directional_weight
        short_score = short_sum / directional_weight
        value = long_score - short_score  # [-1, +1]

    return Score(
        value=value,
        confidence=confidence,
        long_score=long_score,
        short_score=short_score,
        contributing_signals=n,
    )
