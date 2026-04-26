"""Real counterfactual flip rate computation for fairness measurement"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FlipRateResult:
    """Result of counterfactual flip rate computation"""
    flip_rate: float
    flip_count: int
    total_records: int
    flips_per_record: np.ndarray
    y_prob_original: np.ndarray
    y_prob_swapped: np.ndarray
    skipped_records: int
    mean_flip_magnitude: float


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result"""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    n_bootstrap: int


def compute_counterfactual_flip_rate(
    y_prob: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5,
    random_seed: int = 42
) -> FlipRateResult:
    """
    Compute group shift estimate via conservative group mean difference attribution.

    IMPORTANT: This is NOT true counterfactual analysis. This is a conservative
    estimate that attributes 50% of observed group mean difference to the protected
    attribute itself. True counterfactual would require causal model training.

    For each group, computes the observed mean prediction difference and applies
    a 50% conservative fraction as the estimated group effect. Swaps this effect
    to create counterfactual predictions and counts "flips" where |ΔP| > threshold.

    Args:
        y_prob: Predicted probabilities (1D array)
        protected_attr: Protected attribute values (binary 0/1, may contain NaN)
        threshold: Magnitude threshold for flip detection (default 0.5)
        random_seed: Random seed for reproducibility

    Returns:
        FlipRateResult with flip_rate, per-record flip status, and metadata
    """
    np.random.seed(random_seed)

    # Convert to numpy arrays
    y_prob = np.asarray(y_prob, dtype=float)
    protected_attr = np.asarray(protected_attr, dtype=float)

    n_total = len(y_prob)

    # Handle missing values in protected_attr
    valid_mask = ~np.isnan(protected_attr)
    skipped_records = np.sum(~valid_mask)

    # Work only with valid records
    y_prob_valid = y_prob[valid_mask]
    protected_attr_valid = protected_attr[valid_mask]
    n_valid = len(y_prob_valid)

    if n_valid == 0:
        # All records skipped
        return FlipRateResult(
            flip_rate=0.0,
            flip_count=0,
            total_records=n_total,
            flips_per_record=np.zeros(n_valid, dtype=bool),
            y_prob_original=np.array([], dtype=float),
            y_prob_swapped=np.array([], dtype=float),
            skipped_records=skipped_records,
            mean_flip_magnitude=0.0
        )

    # Check if single group (all same value)
    unique_groups = np.unique(protected_attr_valid)
    if len(unique_groups) <= 1:
        # Single group, no swap possible
        return FlipRateResult(
            flip_rate=0.0,
            flip_count=0,
            total_records=n_total,
            flips_per_record=np.zeros(n_valid, dtype=bool),
            y_prob_original=y_prob_valid.copy(),
            y_prob_swapped=y_prob_valid.copy(),
            skipped_records=skipped_records,
            mean_flip_magnitude=0.0
        )

    # Create counterfactual predictions by swapping protected attribute
    # Estimate the effect of protected attribute on predictions
    group_0_mask = protected_attr_valid == 0
    group_1_mask = protected_attr_valid == 1

    group_0_mean = np.mean(y_prob_valid[group_0_mask]) if np.sum(group_0_mask) > 0 else 0.5
    group_1_mean = np.mean(y_prob_valid[group_1_mask]) if np.sum(group_1_mask) > 0 else 0.5

    # Estimate group effect (bias due to protected attribute)
    # Use a fraction of the observed difference to be conservative
    raw_effect = group_1_mean - group_0_mean
    group_effect = raw_effect * 0.5  # Conservative: attribute accounts for 50% of mean difference

    # Create swapped probabilities
    y_prob_swapped = np.copy(y_prob_valid)
    y_prob_swapped[group_0_mask] = np.clip(y_prob_valid[group_0_mask] + group_effect, 0.0, 1.0)
    y_prob_swapped[group_1_mask] = np.clip(y_prob_valid[group_1_mask] - group_effect, 0.0, 1.0)

    # Compute flip magnitudes
    flip_magnitudes = np.abs(y_prob_swapped - y_prob_valid)

    # Detect flips
    flips = flip_magnitudes > threshold
    flip_count = np.sum(flips)

    # Compute statistics
    flip_rate = flip_count / n_valid if n_valid > 0 else 0.0
    mean_flip_magnitude = np.mean(flip_magnitudes[flips]) if flip_count > 0 else 0.0

    return FlipRateResult(
        flip_rate=float(flip_rate),
        flip_count=int(flip_count),
        total_records=n_total,
        flips_per_record=flips,
        y_prob_original=y_prob_valid.copy(),
        y_prob_swapped=y_prob_swapped.copy(),
        skipped_records=int(skipped_records),
        mean_flip_magnitude=float(mean_flip_magnitude)
    )


def compute_counterfactual_flip_rate_ci(
    y_prob: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> BootstrapResult:
    """
    Bootstrap confidence interval for group shift estimate (NOT true counterfactual).

    See compute_counterfactual_flip_rate() for limitations: this is a conservative
    50% attribution estimate, not causal inference.

    Args:
        y_prob: Predicted probabilities
        protected_attr: Protected attribute values
        threshold: Flip detection threshold
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence: Confidence level for CI (default 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with point estimate and CI bounds
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_prob)

    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        y_prob_boot = y_prob[indices]
        protected_attr_boot = protected_attr[indices]

        # Compute flip rate on bootstrap sample
        result = compute_counterfactual_flip_rate(
            y_prob=y_prob_boot,
            protected_attr=protected_attr_boot,
            threshold=threshold,
            random_seed=random_seed + len(bootstrap_estimates)
        )
        bootstrap_estimates.append(result.flip_rate)

    bootstrap_estimates = np.array(bootstrap_estimates)
    point_estimate = np.mean(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return BootstrapResult(
        point_estimate=float(point_estimate),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        n_bootstrap=n_bootstrap
    )
