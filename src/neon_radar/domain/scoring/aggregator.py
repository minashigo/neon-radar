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
    confluence_bonus: float = 0.20,
    confluence_penalty: float = 0.15,
    max_confidence_boost: float = 0.40,
) -> Score:
    """Combine ``signals`` into a single :class:`Score`.

    Parameters
    ----------
    signals
        Per-factor contributions from the rules engine.
    min_confidence
        Signals below this confidence are dropped before aggregation.
    confluence_bonus
        Added to confidence for each confirming category.
    confluence_penalty
        Subtracted from confidence for each conflicting category.
    max_confidence_boost
        Maximum allowed increase to confidence from confluence.

    Returns
    -------
    Score
        ``value`` reflects direction and magnitude, weighted by rule
        importance. ``confidence`` is the weighted average of signal
        confidences, plus confluence modifiers.
    """
    from collections import defaultdict
    from neon_radar.domain.scoring.value_objects import ConfluenceState

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
            confluence_state=ConfluenceState.NEUTRAL,
            confluence_details=("No signals",),
        )

    total_weight_all = 0.0
    directional_weight = 0.0
    long_sum = 0.0
    short_sum = 0.0
    confidence_sum = 0.0

    cat_signals = defaultdict(list)

    for sig in filtered:
        w = sig.weight
        total_weight_all += w
        contribution = sig.value * w
        if sig.value > 0:
            long_sum += contribution
            directional_weight += w
        elif sig.value < 0:
            short_sum += -contribution
            directional_weight += w
        confidence_sum += w * sig.confidence
        cat_signals[sig.category].append(sig)

    if total_weight_all == 0.0:
        return Score(
            value=0.0,
            confidence=0.0,
            long_score=0.0,
            short_score=0.0,
            contributing_signals=n,
            confluence_state=ConfluenceState.NEUTRAL,
            confluence_details=("Total weight is zero",),
        )

    base_confidence = confidence_sum / total_weight_all

    if directional_weight == 0.0:
        long_score = 0.0
        short_score = 0.0
        value = 0.0
        overall_dir = 0
    else:
        long_score = long_sum / directional_weight
        short_score = short_sum / directional_weight
        value = long_score - short_score
        overall_dir = 1 if value > 0 else (-1 if value < 0 else 0)

    # --- Confluence Logic ---
    cat_dirs: dict[str, int] = {}
    for cat, c_signals in cat_signals.items():
        cat_val = sum(s.value * s.weight for s in c_signals)
        if cat_val > 0.001:
            cat_dirs[cat] = 1
        elif cat_val < -0.001:
            cat_dirs[cat] = -1

    active_cats = list(cat_dirs.keys())
    state = ConfluenceState.UNALIGNED
    details: list[str] = []
    final_confidence = base_confidence

    if overall_dir == 0 or len(active_cats) < 2:
        details.append("Not enough categories for confluence")
    else:
        confirming = [c.value for c, d in cat_dirs.items() if d == overall_dir]
        conflicting = [c.value for c, d in cat_dirs.items() if d == -overall_dir]

        if conflicting:
            state = ConfluenceState.CONFLICTING
            penalty = confluence_penalty * len(conflicting)
            final_confidence = max(0.0, base_confidence - penalty)
            details.append(f"Primary: {', '.join(confirming) if confirming else 'None'}")
            details.append(f"Conflicting: {', '.join(conflicting)}")
            details.append(f"Penalty applied: -{penalty:.2f}")
        elif len(confirming) > 1:
            state = ConfluenceState.CONFIRMED
            bonus = min(max_confidence_boost, confluence_bonus * (len(confirming) - 1))
            final_confidence = min(1.0, base_confidence + bonus)
            details.append(f"Confirmed by: {', '.join(confirming)}")
            details.append(f"Bonus applied: +{bonus:.2f}")
        else:
            details.append("Unaligned categories")

    return Score(
        value=value,
        confidence=final_confidence,
        long_score=long_score,
        short_score=short_score,
        contributing_signals=n,
        confluence_state=state,
        confluence_details=tuple(details),
    )
