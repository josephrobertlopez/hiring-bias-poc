"""Fairness v2 - Production fairness metrics with fairlearn integration.

This module provides comprehensive fairness metrics including demographic parity,
equalized odds, and calibration with confidence intervals and per-group reporting.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass
class FairnessResult:
    """Result of fairness metric computation."""
    metric_name: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    confidence_level: float

    @property
    def lower_bound(self) -> float:
        """Lower bound of confidence interval."""
        return self.confidence_interval[0]

    @property
    def upper_bound(self) -> float:
        """Upper bound of confidence interval."""
        return self.confidence_interval[1]


class FairnessMetrics:
    """Production fairness metrics calculator.

    Computes demographic parity, equalized odds, calibration, and other metrics
    with bootstrap confidence intervals and per-group reporting.

    Attributes:
        random_state: Seed for reproducibility
        n_bootstrap: Number of bootstrap resamples
    """

    def __init__(self, random_state: int = None, n_bootstrap: int = 1000):
        """Initialize fairness metrics.

        Args:
            random_state: Seed for reproducibility
            n_bootstrap: Number of bootstrap resamples
        """
        self.random_state = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
        self.n_bootstrap = n_bootstrap

    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        return_ci: bool = True
    ) -> FairnessResult:
        """Compute demographic parity (disparity index).

        Demographic parity is the ratio of positive prediction rates:
        DI = min(P(pred=1|group_a), P(pred=1|group_b)) / max(...)

        Args:
            y_pred: Predicted labels (0/1) or probabilities
            protected_attr: Protected attribute values (0/1 for binary)
            return_ci: Whether to return confidence interval

        Returns:
            FairnessResult with point estimate and CI
        """
        y_pred_binary = (y_pred > 0.5).astype(int) if np.max(y_pred) <= 1.0 else y_pred

        unique_groups = np.unique(protected_attr)
        group_rates = {}
        for group in unique_groups:
            mask = protected_attr == group
            pos_rate = np.mean(y_pred_binary[mask])
            group_rates[group] = pos_rate

        if len(unique_groups) != 2:
            raise ValueError("Demographic parity requires binary protected attribute")

        rate_a, rate_b = list(group_rates.values())
        point_estimate = min(rate_a, rate_b) / max(rate_a, rate_b) if max(rate_a, rate_b) > 0 else 0

        # Bootstrap CI
        if return_ci:
            bootstrap_metrics = []
            for _ in range(self.n_bootstrap):
                indices = self.random_state.choice(len(y_pred), len(y_pred), replace=True)
                boot_pred = y_pred_binary[indices]
                boot_attr = protected_attr[indices]

                boot_rates = {}
                for group in unique_groups:
                    mask = boot_attr == group
                    if np.sum(mask) > 0:
                        boot_rates[group] = np.mean(boot_pred[mask])
                    else:
                        boot_rates[group] = 0

                boot_metric = min(boot_rates.values()) / max(boot_rates.values()) if max(boot_rates.values()) > 0 else 0
                bootstrap_metrics.append(boot_metric)

            ci = (np.percentile(bootstrap_metrics, 2.5), np.percentile(bootstrap_metrics, 97.5))
        else:
            ci = (point_estimate, point_estimate)

        return FairnessResult(
            metric_name="demographic_parity",
            point_estimate=point_estimate,
            confidence_interval=ci,
            confidence_level=0.95
        )

    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        return_ci: bool = True
    ) -> FairnessResult:
        """Compute equalized odds (TPR difference).

        Equalized odds is the absolute difference in TPR across groups:
        EO = |TPR_group_a - TPR_group_b|

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            protected_attr: Protected attribute values (0/1 for binary)
            return_ci: Whether to return confidence interval

        Returns:
            FairnessResult with point estimate and CI
        """
        y_pred_binary = (y_pred > 0.5).astype(int) if np.max(y_pred) <= 1.0 else y_pred

        unique_groups = np.unique(protected_attr)
        if len(unique_groups) != 2:
            raise ValueError("Equalized odds requires binary protected attribute")

        tprs = []
        for group in unique_groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred_binary[mask]

            positives = np.sum(group_true == 1)
            if positives > 0:
                tp = np.sum((group_pred == 1) & (group_true == 1))
                tpr = tp / positives
            else:
                tpr = 0
            tprs.append(tpr)

        point_estimate = abs(tprs[0] - tprs[1])

        # Bootstrap CI
        if return_ci:
            bootstrap_metrics = []
            for _ in range(self.n_bootstrap):
                indices = self.random_state.choice(len(y_true), len(y_true), replace=True)
                boot_true = y_true[indices]
                boot_pred = y_pred_binary[indices]
                boot_attr = protected_attr[indices]

                boot_tprs = []
                for group in unique_groups:
                    mask = boot_attr == group
                    group_true = boot_true[mask]
                    group_pred = boot_pred[mask]

                    positives = np.sum(group_true == 1)
                    if positives > 0:
                        tp = np.sum((group_pred == 1) & (group_true == 1))
                        tpr = tp / positives
                    else:
                        tpr = 0
                    boot_tprs.append(tpr)

                boot_metric = abs(boot_tprs[0] - boot_tprs[1])
                bootstrap_metrics.append(boot_metric)

            ci = (np.percentile(bootstrap_metrics, 2.5), np.percentile(bootstrap_metrics, 97.5))
        else:
            ci = (point_estimate, point_estimate)

        return FairnessResult(
            metric_name="equalized_odds",
            point_estimate=point_estimate,
            confidence_interval=ci,
            confidence_level=0.95
        )

    def calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
        return_ci: bool = True
    ) -> FairnessResult:
        """Compute expected calibration error.

        ECE = sum over bins: (bin_size / n) * |predicted_rate - true_rate|

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted probabilities (0-1)
            n_bins: Number of bins for calibration
            return_ci: Whether to return confidence interval

        Returns:
            FairnessResult with point estimate and CI
        """
        bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -0.001  # Include 0
        bin_edges[-1] = 1.001  # Include 1

        weighted_error = 0
        for i in range(n_bins):
            in_bin = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
            if np.sum(in_bin) == 0:
                continue

            pred_rate = np.mean(y_pred[in_bin])
            true_rate = np.mean(y_true[in_bin])
            error = abs(pred_rate - true_rate)
            weight = np.sum(in_bin) / len(y_pred)
            weighted_error += weight * error

        point_estimate = weighted_error

        # Bootstrap CI
        if return_ci:
            bootstrap_metrics = []
            for _ in range(self.n_bootstrap):
                indices = self.random_state.choice(len(y_true), len(y_true), replace=True)
                boot_true = y_true[indices]
                boot_pred = y_pred[indices]

                boot_bin_edges = np.percentile(boot_pred, np.linspace(0, 100, n_bins + 1))
                boot_bin_edges[0] = -0.001
                boot_bin_edges[-1] = 1.001

                boot_error = 0
                for i in range(n_bins):
                    in_bin = (boot_pred >= boot_bin_edges[i]) & (boot_pred < boot_bin_edges[i+1])
                    if np.sum(in_bin) == 0:
                        continue

                    pred_rate = np.mean(boot_pred[in_bin])
                    true_rate = np.mean(boot_true[in_bin])
                    error = abs(pred_rate - true_rate)
                    weight = np.sum(in_bin) / len(boot_true)
                    boot_error += weight * error

                bootstrap_metrics.append(boot_error)

            ci = (np.percentile(bootstrap_metrics, 2.5), np.percentile(bootstrap_metrics, 97.5))
        else:
            ci = (point_estimate, point_estimate)

        return FairnessResult(
            metric_name="calibration_error",
            point_estimate=point_estimate,
            confidence_interval=ci,
            confidence_level=0.95
        )

    def per_group_metrics(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict[int, float]:
        """Compute metrics per protected group.

        Args:
            y_pred: Predicted labels (0/1)
            protected_attr: Protected attribute values

        Returns:
            Dictionary mapping group value to positive prediction rate
        """
        y_pred_binary = (y_pred > 0.5).astype(int) if np.max(y_pred) <= 1.0 else y_pred

        group_metrics = {}
        for group in np.unique(protected_attr):
            mask = protected_attr == group
            pos_rate = np.mean(y_pred_binary[mask])
            group_metrics[group] = pos_rate

        return group_metrics

    def disparity_index(
        self,
        metrics: Dict[int, float]
    ) -> Dict[Tuple[int, int], float]:
        """Compute pairwise disparity indices.

        Args:
            metrics: Dictionary mapping group to metric value

        Returns:
            Dictionary mapping (group_a, group_b) to DI value
        """
        disparities = {}
        groups = sorted(metrics.keys())

        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                rate1 = metrics[g1]
                rate2 = metrics[g2]
                di = min(rate1, rate2) / max(rate1, rate2) if max(rate1, rate2) > 0 else 0
                disparities[(g1, g2)] = di

        return disparities

