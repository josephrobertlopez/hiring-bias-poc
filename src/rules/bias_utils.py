"""Shared bias calculation utilities."""
from typing import List, Dict, Any
import numpy as np


def compute_disparity_index(hiring_rates: List[float], threshold: float = 0.8) -> Dict[str, Any]:
    """Compute disparity index with standardized logic.

    Disparity Index (DI) measures fairness by comparing the lowest to highest
    hiring rate across demographic groups. A DI of 1.0 means perfect parity.
    A DI of 0.8 is the standard 4/5 rule threshold in employment law.

    Args:
        hiring_rates: List of hiring rates per group (0.0-1.0)
        threshold: Bias detection threshold (default 0.8 per 4/5 rule)

    Returns:
        Dict with:
            - disparity_index: Computed DI (0-1 range)
            - bias_detected: Bool indicating if DI < threshold
            - min_rate: Minimum hiring rate
            - max_rate: Maximum hiring rate
    """
    if not hiring_rates or len(hiring_rates) < 2:
        return {
            "disparity_index": 1.0,
            "bias_detected": False,
            "min_rate": 0.0,
            "max_rate": 0.0
        }

    min_rate = min(hiring_rates)
    max_rate = max(hiring_rates)

    if max_rate == 0:
        di = 1.0  # All rates zero = no bias
    else:
        di = min_rate / max_rate

    return {
        "disparity_index": float(di),
        "bias_detected": di < threshold,
        "min_rate": float(min_rate),
        "max_rate": float(max_rate)
    }
