"""Fairness metrics for hiring bias detection and CI gates.

Implements key fairness metrics with proper CI gate thresholds:
- Disparate Impact (4/5 rule): DI ≥ 0.8
- Equalized Odds Gap: gap ≤ 0.1
- Calibration ECE: ECE ≤ 0.05
- Per-group AUC analysis
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings


@dataclass
class FairnessGateResult:
    """Result of fairness gate evaluation."""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    group_breakdown: Dict[str, float]
    details: Dict[str, Any]


@dataclass
class GroupMetrics:
    """Per-group fairness metrics."""
    group_name: str
    group_value: Any
    sample_size: int
    positive_rate: float  # P(Y=1|A=group)
    selection_rate: float  # P(Ŷ=1|A=group)
    tpr: float  # True Positive Rate
    fpr: float  # False Positive Rate
    auc: float  # Area Under Curve
    calibration_error: float  # Expected Calibration Error


class FairnessMetricsCalculator:
    """Calculate fairness metrics for hiring bias detection."""

    def __init__(self):
        """Initialize fairness metrics calculator."""
        pass

    def calculate_disparate_impact(self,
                                 y_pred: np.ndarray,
                                 sensitive_attr: np.ndarray) -> FairnessGateResult:
        """Calculate Disparate Impact using 4/5 rule.

        Args:
            y_pred: Binary predictions (0/1)
            sensitive_attr: Sensitive attribute values

        Returns:
            FairnessGateResult with DI calculation
        """
        # Get unique groups
        groups = np.unique(sensitive_attr)
        selection_rates = {}

        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() > 0:
                selection_rate = y_pred[group_mask].mean()
                selection_rates[str(group)] = selection_rate
            else:
                selection_rates[str(group)] = 0.0

        # Calculate DI as min_rate / max_rate
        rates = list(selection_rates.values())
        if len(rates) == 0 or max(rates) == 0:
            # Edge case: return 1.0 when max_rate=0 (fix from prompt)
            di_value = 1.0
        else:
            di_value = min(rates) / max(rates)

        # 4/5 rule: DI should be ≥ 0.8
        passed = di_value >= 0.8

        return FairnessGateResult(
            metric_name="disparate_impact",
            value=di_value,
            threshold=0.8,
            passed=passed,
            group_breakdown=selection_rates,
            details={
                "min_selection_rate": min(rates) if rates else 0.0,
                "max_selection_rate": max(rates) if rates else 0.0,
                "group_counts": {str(g): (sensitive_attr == g).sum() for g in groups}
            }
        )

    def calculate_equalized_odds_gap(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   sensitive_attr: np.ndarray) -> FairnessGateResult:
        """Calculate Equalized Odds Gap.

        Equalized odds requires TPR and FPR to be similar across groups.
        Gap = max(|TPR_i - TPR_j|, |FPR_i - FPR_j|) for all groups i, j.

        Args:
            y_true: True binary labels (0/1)
            y_pred: Binary predictions (0/1)
            sensitive_attr: Sensitive attribute values

        Returns:
            FairnessGateResult with equalized odds gap
        """
        groups = np.unique(sensitive_attr)
        group_metrics = {}

        # Calculate TPR and FPR for each group
        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() == 0:
                continue

            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate confusion matrix
            if len(np.unique(y_true_group)) < 2 or len(np.unique(y_pred_group)) < 2:
                # Handle edge case: only one class present
                tpr = 0.0 if (y_true_group == 1).sum() == 0 else float(np.mean(y_pred_group[y_true_group == 1]))
                fpr = 0.0 if (y_true_group == 0).sum() == 0 else float(np.mean(y_pred_group[y_true_group == 0]))
            else:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            group_metrics[str(group)] = {
                "tpr": tpr,
                "fpr": fpr,
                "sample_size": group_mask.sum()
            }

        # Calculate maximum gaps
        tprs = [metrics["tpr"] for metrics in group_metrics.values()]
        fprs = [metrics["fpr"] for metrics in group_metrics.values()]

        if len(tprs) < 2:
            # Need at least 2 groups to calculate gap
            eo_gap = 0.0
        else:
            tpr_gap = max(tprs) - min(tprs)
            fpr_gap = max(fprs) - min(fprs)
            eo_gap = max(tpr_gap, fpr_gap)

        # Threshold: gap ≤ 0.1
        passed = eo_gap <= 0.1

        return FairnessGateResult(
            metric_name="equalized_odds_gap",
            value=eo_gap,
            threshold=0.1,
            passed=passed,
            group_breakdown={g: metrics["tpr"] for g, metrics in group_metrics.items()},
            details={
                "tpr_by_group": {g: metrics["tpr"] for g, metrics in group_metrics.items()},
                "fpr_by_group": {g: metrics["fpr"] for g, metrics in group_metrics.items()},
                "tpr_gap": max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
                "fpr_gap": max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0
            }
        )

    def calculate_calibration_ece_by_group(self,
                                         y_true: np.ndarray,
                                         y_prob: np.ndarray,
                                         sensitive_attr: np.ndarray,
                                         n_bins: int = 10) -> FairnessGateResult:
        """Calculate Expected Calibration Error by demographic group.

        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities [0, 1]
            sensitive_attr: Sensitive attribute values
            n_bins: Number of bins for calibration calculation

        Returns:
            FairnessGateResult with per-group ECE
        """
        groups = np.unique(sensitive_attr)
        group_eces = {}

        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() == 0:
                continue

            y_true_group = y_true[group_mask]
            y_prob_group = y_prob[group_mask]

            # Calculate ECE for this group
            ece = self._calculate_ece(y_prob_group, y_true_group, n_bins)
            group_eces[str(group)] = ece

        # Overall ECE (worst group)
        if group_eces:
            worst_ece = max(group_eces.values())
        else:
            worst_ece = 0.0

        # Threshold: ECE ≤ 0.05
        passed = worst_ece <= 0.05

        return FairnessGateResult(
            metric_name="calibration_ece",
            value=worst_ece,
            threshold=0.05,
            passed=passed,
            group_breakdown=group_eces,
            details={
                "per_group_ece": group_eces,
                "worst_group": max(group_eces.keys(), key=lambda k: group_eces[k]) if group_eces else None
            }
        )

    def calculate_per_group_auc(self,
                               y_true: np.ndarray,
                               y_prob: np.ndarray,
                               sensitive_attr: np.ndarray) -> FairnessGateResult:
        """Calculate AUC by demographic group.

        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities [0, 1]
            sensitive_attr: Sensitive attribute values

        Returns:
            FairnessGateResult with per-group AUC analysis
        """
        groups = np.unique(sensitive_attr)
        group_aucs = {}

        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() == 0:
                continue

            y_true_group = y_true[group_mask]
            y_prob_group = y_prob[group_mask]

            # Calculate AUC if we have both classes
            if len(np.unique(y_true_group)) >= 2:
                try:
                    auc = roc_auc_score(y_true_group, y_prob_group)
                    group_aucs[str(group)] = auc
                except ValueError:
                    # Handle edge cases
                    group_aucs[str(group)] = 0.5
            else:
                # Only one class present
                group_aucs[str(group)] = 0.5

        # Calculate AUC gap (difference between best and worst groups)
        if len(group_aucs) >= 2:
            auc_gap = max(group_aucs.values()) - min(group_aucs.values())
            min_auc = min(group_aucs.values())
        else:
            auc_gap = 0.0
            min_auc = list(group_aucs.values())[0] if group_aucs else 0.5

        # Custom threshold: min AUC ≥ 0.7 and gap ≤ 0.1
        passed = min_auc >= 0.7 and auc_gap <= 0.1

        return FairnessGateResult(
            metric_name="per_group_auc",
            value=min_auc,
            threshold=0.7,
            passed=passed,
            group_breakdown=group_aucs,
            details={
                "auc_by_group": group_aucs,
                "auc_gap": auc_gap,
                "min_auc": min_auc,
                "max_auc": max(group_aucs.values()) if group_aucs else 0.5
            }
        )

    def get_group_statistics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray,
                           sensitive_attr: np.ndarray) -> List[GroupMetrics]:
        """Get comprehensive statistics for each demographic group.

        Args:
            y_true: True binary labels
            y_pred: Binary predictions
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute values

        Returns:
            List of GroupMetrics for each group
        """
        groups = np.unique(sensitive_attr)
        group_stats = []

        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() == 0:
                continue

            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            y_prob_group = y_prob[group_mask]

            # Basic rates
            positive_rate = y_true_group.mean()
            selection_rate = y_pred_group.mean()

            # TPR and FPR
            if len(np.unique(y_true_group)) >= 2:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                # AUC
                try:
                    auc = roc_auc_score(y_true_group, y_prob_group)
                except ValueError:
                    auc = 0.5
            else:
                tpr = 0.0
                fpr = 0.0
                auc = 0.5

            # Calibration error
            ece = self._calculate_ece(y_prob_group, y_true_group, n_bins=10)

            group_stats.append(GroupMetrics(
                group_name=str(group),
                group_value=group,
                sample_size=group_mask.sum(),
                positive_rate=positive_rate,
                selection_rate=selection_rate,
                tpr=tpr,
                fpr=fpr,
                auc=auc,
                calibration_error=ece
            ))

        return group_stats

    def _calculate_ece(self, y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error.

        Args:
            y_prob: Predicted probabilities [0, 1]
            y_true: True binary labels {0, 1}
            n_bins: Number of bins

        Returns:
            ECE value (lower is better)
        """
        if len(y_prob) == 0:
            return 0.0

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def run_all_fairness_gates(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray,
                              sensitive_attrs: Dict[str, np.ndarray]) -> Dict[str, List[FairnessGateResult]]:
        """Run all fairness gates for multiple sensitive attributes.

        Args:
            y_true: True binary labels
            y_pred: Binary predictions
            y_prob: Predicted probabilities
            sensitive_attrs: Dictionary of {attr_name: attr_values}

        Returns:
            Dictionary of {attr_name: [gate_results]}
        """
        all_results = {}

        for attr_name, attr_values in sensitive_attrs.items():
            attr_results = []

            # Disparate Impact
            di_result = self.calculate_disparate_impact(y_pred, attr_values)
            attr_results.append(di_result)

            # Equalized Odds Gap
            eo_result = self.calculate_equalized_odds_gap(y_true, y_pred, attr_values)
            attr_results.append(eo_result)

            # Calibration ECE
            ece_result = self.calculate_calibration_ece_by_group(y_true, y_prob, attr_values)
            attr_results.append(ece_result)

            # Per-group AUC
            auc_result = self.calculate_per_group_auc(y_true, y_prob, attr_values)
            attr_results.append(auc_result)

            all_results[attr_name] = attr_results

        return all_results

    def check_all_gates_passed(self, gate_results: Dict[str, List[FairnessGateResult]]) -> bool:
        """Check if all fairness gates passed.

        Args:
            gate_results: Results from run_all_fairness_gates

        Returns:
            True if all gates passed, False otherwise
        """
        for attr_results in gate_results.values():
            for result in attr_results:
                if not result.passed:
                    return False
        return True