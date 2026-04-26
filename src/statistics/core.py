import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass
from scipy.stats import ttest_rel, norm
from sklearn.metrics import roc_auc_score

@dataclass
class BootstrapResult:
    statistic: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_bootstrap: int

    @property
    def point_estimate(self) -> float:
        """Alias for statistic (point estimate)"""
        return self.statistic

    @property
    def lower_bound(self) -> float:
        """Lower CI bound"""
        return self.confidence_interval[0]

    @property
    def upper_bound(self) -> float:
        """Upper CI bound"""
        return self.confidence_interval[1]

@dataclass
class StatisticalTestResult:
    statistic: float
    p_value: float
    effect_size: float
    test_name: str

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cohen's d effect size between two samples"""
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    std_x1 = np.std(x1, ddof=1)
    std_x2 = np.std(x2, ddof=1)
    n1, n2 = len(x1), len(x2)

    if n1 + n2 <= 2:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * std_x1**2 + (n2 - 1) * std_x2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_x1 - mean_x2) / pooled_std

# Keep private alias for backward compatibility
_cohens_d = cohens_d

def _compute_auc_variance(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """DeLong variance estimate for AUC"""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    auc = roc_auc_score(y_true, y_prob)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)

    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return max(variance, 1e-10)  # Avoid zero variance

def bootstrap_ci(
    data: np.ndarray,
    labels: np.ndarray = None,
    statistic_fn: Callable = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
    data2: np.ndarray = None
) -> BootstrapResult:
    """Bootstrap confidence intervals for any statistic

    If data2 is provided, resamples both data and data2 independently
    and passes both to statistic_fn(labels, data, data2)
    """
    rng = np.random.default_rng(random_seed)
    n = len(data)
    bootstrap_stats = []

    if data2 is None:
        # Single-data case: statistic_fn(labels, data)
        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            stat = statistic_fn(labels[indices], data[indices])
            bootstrap_stats.append(stat)
    else:
        # Two-data case: statistic_fn(labels, data, data2) where both are resampled
        n2 = len(data2)
        for _ in range(n_bootstrap):
            indices1 = rng.choice(n, size=n, replace=True)
            indices2 = rng.choice(n2, size=n2, replace=True)
            stat = statistic_fn(labels[indices1], data[indices1], data2[indices2])
            bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    point = np.mean(bootstrap_stats)

    return BootstrapResult(
        statistic=point,
        confidence_interval=(lower, upper),
        confidence_level=confidence,
        n_bootstrap=n_bootstrap
    )

def paired_t_test(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
    random_seed: int = 42
) -> StatisticalTestResult:
    """Paired t-test for comparing two classifiers"""
    np.random.seed(random_seed)

    t_stat, p_value = ttest_rel(y_prob_1, y_prob_2)
    effect_size = cohens_d(y_prob_1, y_prob_2)

    return StatisticalTestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(effect_size),
        test_name='Paired T-Test'
    )

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    random_seed: int = 42
) -> StatisticalTestResult:
    """McNemar test for classifier agreement"""
    np.random.seed(random_seed)

    # Build 2x2 contingency table: agreement on correct vs disagreement
    contingency_table = np.array([
        [np.sum((y_pred_1 == 1) & (y_pred_2 == 1)), np.sum((y_pred_1 == 0) & (y_pred_2 == 1))],
        [np.sum((y_pred_1 == 1) & (y_pred_2 == 0)), np.sum((y_pred_1 == 0) & (y_pred_2 == 0))]
    ])

    # McNemar statistic: (b - c)^2 / (b + c) where b, c are off-diagonal
    b = contingency_table[0, 1]
    c = contingency_table[1, 0]

    if b + c == 0:
        return StatisticalTestResult(
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            test_name="McNemar's Test"
        )

    chi2_stat = (b - c)**2 / (b + c)
    p_value = 1.0 - norm.cdf(np.sqrt(chi2_stat))
    effect_size = abs(b - c) / (b + c) if (b + c) > 0 else 0.0

    return StatisticalTestResult(
        statistic=float(chi2_stat),
        p_value=float(p_value),
        effect_size=float(effect_size),
        test_name="McNemar's Test"
    )

def delong_roc_test(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
    random_seed: int = 42
) -> StatisticalTestResult:
    """DeLong test for AUC comparison"""
    np.random.seed(random_seed)

    auc_1 = roc_auc_score(y_true, y_prob_1)
    auc_2 = roc_auc_score(y_true, y_prob_2)

    var_1 = _compute_auc_variance(y_true, y_prob_1)
    var_2 = _compute_auc_variance(y_true, y_prob_2)

    # Approximate covariance
    cov_12 = np.cov(y_prob_1, y_prob_2)[0, 1] if len(y_prob_1) > 1 else 0.0

    se_diff = np.sqrt(var_1 + var_2 - 2 * cov_12)
    if se_diff == 0:
        return StatisticalTestResult(
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            test_name="DeLong's ROC Test"
        )

    z_stat = (auc_1 - auc_2) / se_diff
    p_value = 2 * norm.sf(np.abs(z_stat))
    effect_size = abs(auc_1 - auc_2)

    return StatisticalTestResult(
        statistic=float(z_stat),
        p_value=float(p_value),
        effect_size=float(effect_size),
        test_name="DeLong's ROC Test"
    )
