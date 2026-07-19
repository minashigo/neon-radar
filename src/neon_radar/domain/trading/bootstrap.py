"""Domain models for Bootstrap Validation."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BootstrapMetricDistribution:
    """Summary statistics and full distribution for a single bootstrapped metric."""

    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    ci_lower_95: float
    ci_upper_95: float
    values: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class BootstrapReport:
    """The result of a Block Bootstrap analysis."""

    iterations: int
    block_size: int
    metrics: dict[str, BootstrapMetricDistribution]
