"""Pipeline for processing intelligence signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neon_radar.domain.market_intelligence.models import IntelligenceSignal, PipelineContext


@runtime_checkable
class PipelineStep(Protocol):
    """Protocol for a step in the intelligence signal pipeline."""

    async def process(
        self, context: PipelineContext, signals: Sequence[IntelligenceSignal]
    ) -> tuple[IntelligenceSignal, ...]:
        """Process a batch of signals and return the modified/filtered signals."""
        ...


class SignalPipeline:
    """Executes a series of processing steps on intelligence signals."""

    def __init__(self, steps: Sequence[PipelineStep]) -> None:
        """Initialize the pipeline with a specific sequence of steps."""
        self._steps = tuple(steps)

    async def execute(
        self, context: PipelineContext, signals: Sequence[IntelligenceSignal]
    ) -> tuple[IntelligenceSignal, ...]:
        """Run signals through all pipeline steps sequentially."""
        current_signals = tuple(signals)
        for step in self._steps:
            if not current_signals:
                break
            current_signals = await step.process(context, current_signals)
        return current_signals
