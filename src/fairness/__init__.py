"""Fairness metrics and counterfactual analysis"""

from .real import (
    FlipRateResult,
    BootstrapResult,
    compute_counterfactual_flip_rate,
    compute_counterfactual_flip_rate_ci,
)

__all__ = [
    "FlipRateResult",
    "BootstrapResult",
    "compute_counterfactual_flip_rate",
    "compute_counterfactual_flip_rate_ci",
]
