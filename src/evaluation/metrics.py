"""
6-Metric Measurement Harness for Hiring Bias Detection

Implements the empirical discipline framework for evaluating resume scoring approaches.
All methods must pass these falsification criteria:
- Disparate impact < 0.8 → rejected (EEOC standard)
- Flip rate > 0.1 → direct bias evidence → rejected
- No explainable output → not audit-friendly → rejected
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, calibration_curve


@dataclass
class FairnessResults:
    """Results from 6-metric measurement harness"""
    auc_overall: float
    auc_per_group: Dict[str, float]
    disparate_impact_ratio: float
    equalized_odds_gap: Tuple[float, float]  # (tpr_gap, fpr_gap)
    calibration_per_group: Dict[str, float]
    counterfactual_flip_rate: float
    explanation_faithfulness: float

    @property
    def passes_eeoc(self) -> bool:
        """EEOC 4/5ths rule compliance"""
        return self.disparate_impact_ratio >= 0.8

    @property
    def passes_flip_test(self) -> bool:
        """Counterfactual stability test"""
        return self.counterfactual_flip_rate <= 0.1

    @property
    def is_explainable(self) -> bool:
        """Audit-friendly explanation requirement"""
        return self.explanation_faithfulness >= 0.9

    @property
    def passes_all_tests(self) -> bool:
        """Full empirical discipline compliance"""
        return self.passes_eeoc and self.passes_flip_test and self.is_explainable


class FairnessHarness:
    """6-metric measurement harness for empirical bias detection"""

    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes

    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                y_prob: np.ndarray,
                sensitive_attrs: pd.DataFrame,
                feature_importance: np.ndarray = None) -> FairnessResults:
        """
        Run complete 6-metric evaluation

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            sensitive_attrs: Protected attributes dataframe
            feature_importance: Feature importance scores for explainability

        Returns:
            FairnessResults with all 6 metrics
        """

        # 1. AUC overall + per group
        auc_overall = roc_auc_score(y_true, y_prob)
        auc_per_group = self._compute_auc_per_group(y_true, y_prob, sensitive_attrs)

        # 2. Disparate impact ratio
        disparate_impact = self._compute_disparate_impact(y_pred, sensitive_attrs)

        # 3. Equalized odds gap
        eq_odds_gap = self._compute_equalized_odds_gap(y_true, y_pred, sensitive_attrs)

        # 4. Calibration per group
        calibration_per_group = self._compute_calibration_per_group(y_true, y_prob, sensitive_attrs)

        # 5. Counterfactual flip rate
        flip_rate = self._compute_flip_rate(y_prob, sensitive_attrs)

        # 6. Explanation faithfulness
        explanation_fidelity = self._compute_explanation_faithfulness(feature_importance) if feature_importance is not None else 0.0

        return FairnessResults(
            auc_overall=auc_overall,
            auc_per_group=auc_per_group,
            disparate_impact_ratio=disparate_impact,
            equalized_odds_gap=eq_odds_gap,
            calibration_per_group=calibration_per_group,
            counterfactual_flip_rate=flip_rate,
            explanation_faithfulness=explanation_fidelity
        )

    def _compute_auc_per_group(self, y_true, y_prob, sensitive_attrs):
        """AUC computed separately for each protected group"""
        auc_by_group = {}
        for attr in self.sensitive_attributes:
            for group in sensitive_attrs[attr].unique():
                mask = sensitive_attrs[attr] == group
                if len(np.unique(y_true[mask])) > 1:  # Need both classes
                    auc_by_group[f"{attr}_{group}"] = roc_auc_score(y_true[mask], y_prob[mask])
        return auc_by_group

    def _compute_disparate_impact(self, y_pred, sensitive_attrs):
        """EEOC 4/5ths rule: min_group_rate / max_group_rate >= 0.8"""
        selection_rates = []
        for attr in self.sensitive_attributes:
            for group in sensitive_attrs[attr].unique():
                mask = sensitive_attrs[attr] == group
                rate = np.mean(y_pred[mask])
                selection_rates.append(rate)

        if len(selection_rates) == 0:
            return 1.0
        return min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 1.0

    def _compute_equalized_odds_gap(self, y_true, y_pred, sensitive_attrs):
        """TPR and FPR parity across groups"""
        tprs, fprs = [], []

        for attr in self.sensitive_attributes:
            for group in sensitive_attrs[attr].unique():
                mask = sensitive_attrs[attr] == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                if len(y_true_group) > 0:
                    tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
                    fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
                    tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
                    fn = np.sum((y_true_group == 1) & (y_pred_group == 0))

                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    tprs.append(tpr)
                    fprs.append(fpr)

        tpr_gap = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0
        fpr_gap = max(fprs) - min(fprs) if len(fprs) > 1 else 0.0

        return (tpr_gap, fpr_gap)

    def _compute_calibration_per_group(self, y_true, y_prob, sensitive_attrs):
        """Score reliability across demographic groups"""
        calibration_by_group = {}

        for attr in self.sensitive_attributes:
            for group in sensitive_attrs[attr].unique():
                mask = sensitive_attrs[attr] == group
                if len(y_true[mask]) > 10:  # Need sufficient samples
                    try:
                        fraction_pos, mean_pred = calibration_curve(
                            y_true[mask], y_prob[mask], n_bins=5
                        )
                        # Expected calibration error
                        ece = np.mean(np.abs(fraction_pos - mean_pred))
                        calibration_by_group[f"{attr}_{group}"] = ece
                    except:
                        calibration_by_group[f"{attr}_{group}"] = float('inf')

        return calibration_by_group

    def _compute_flip_rate(self, y_prob, sensitive_attrs):
        """Prediction stability under demographic attribute swaps"""
        # This is a placeholder - full implementation would require
        # counterfactual generation by swapping protected attributes
        # while keeping all other features constant

        # For now, compute variance in predictions across groups
        # as a proxy for demographic dependence
        flip_rates = []

        for attr in self.sensitive_attributes:
            groups = sensitive_attrs[attr].unique()
            if len(groups) > 1:
                group_means = []
                for group in groups:
                    mask = sensitive_attrs[attr] == group
                    group_means.append(np.mean(y_prob[mask]))

                # Variance in group means as proxy for flip sensitivity
                flip_rate = np.var(group_means)
                flip_rates.append(flip_rate)

        return np.mean(flip_rates) if flip_rates else 0.0

    def _compute_explanation_faithfulness(self, feature_importance):
        """How well can we explain individual decisions"""
        if feature_importance is None:
            return 0.0

        # Gini coefficient of feature importance -
        # highly concentrated importance = more explainable
        sorted_importance = np.sort(feature_importance)[::-1]
        n = len(sorted_importance)
        cumsum = np.cumsum(sorted_importance)

        # Gini coefficient
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_importance)) / (n * np.sum(sorted_importance)) - (n + 1) / n

        return gini  # Higher Gini = more concentrated = more explainable