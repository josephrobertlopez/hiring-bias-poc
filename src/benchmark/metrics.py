"""Bias metrics computation for benchmark tasks"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import roc_auc_score


def compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    protected_attr: np.ndarray,
    protected_attr_name: str = 'gender',
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute bias metrics for a task using a simple baseline classifier

    Args:
        X: Feature matrix
        y: Labels
        protected_attr: Protected attribute values
        protected_attr_name: Name of protected attribute
        random_seed: Random seed for reproducibility

    Returns:
        Dict with metrics: auc, disparate_impact, flip_rate, explanation_coverage
    """
    np.random.seed(random_seed)

    # Simple baseline: use feature mean as predictor
    feature_importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    y_prob = (X @ feature_importance) / feature_importance.sum()
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. AUC Score
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0.5

    # 2. Disparate Impact (EEOC 4/5ths rule)
    di = compute_disparate_impact(y_pred, protected_attr)

    # 3. Flip Rate (demographic stability)
    flip_rate = compute_flip_rate(y_prob, protected_attr)

    # 4. Explanation Coverage (Gini coefficient of feature importance)
    explanation_coverage = compute_explanation_coverage(feature_importance)

    return {
        'auc': float(auc),
        'disparate_impact': float(di),
        'flip_rate': float(flip_rate),
        'explanation_coverage': float(explanation_coverage),
        'feature_importance': feature_importance.tolist(),
        'protected_attr_name': protected_attr_name
    }


def compute_disparate_impact(
    y_pred: np.ndarray,
    protected_attr: np.ndarray
) -> float:
    """
    Compute disparate impact ratio (EEOC 4/5ths rule)
    Returns min(selection_rates) / max(selection_rates)

    Higher is better (1.0 = no bias, 0.0 = maximum bias)
    """
    groups = np.unique(protected_attr)
    selection_rates = []

    for group in groups:
        mask = protected_attr == group
        if np.sum(mask) > 0:
            rate = np.mean(y_pred[mask])
            selection_rates.append(rate)

    if len(selection_rates) <= 1:
        return 1.0

    # Handle division by zero
    min_rate = np.min(selection_rates)
    max_rate = np.max(selection_rates)

    if max_rate == 0:
        return 1.0 if min_rate == 0 else 0.0

    return min_rate / max_rate


def compute_flip_rate(
    y_prob: np.ndarray,
    protected_attr: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute flip rate: variance of predicted probability across groups
    as proxy for demographic dependence

    Lower is better (0.0 = stable, 1.0 = highly unstable)
    """
    groups = np.unique(protected_attr)

    if len(groups) <= 1:
        return 0.0

    group_means = []
    for group in groups:
        mask = protected_attr == group
        if np.sum(mask) > 0:
            group_means.append(np.mean(y_prob[mask]))

    # Normalize variance
    flip = np.var(group_means)
    return float(np.clip(flip, 0.0, 1.0))


def compute_explanation_coverage(feature_importance: np.ndarray) -> float:
    """
    Compute explanation coverage using Gini coefficient
    Higher Gini = more concentrated importance = more explainable

    Range: [0, 1] where higher is better
    """
    if len(feature_importance) == 0:
        return 0.0

    # Handle negative importances
    abs_importance = np.abs(feature_importance)
    total = np.sum(abs_importance)

    if total == 0:
        return 0.0

    # Gini coefficient
    sorted_imp = np.sort(abs_importance)[::-1]
    n = len(sorted_imp)

    # Cumulative Gini
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_imp)) / (n * total) - (n + 1) / n

    # Normalize to [0, 1]
    return float(np.clip(gini, 0.0, 1.0))
