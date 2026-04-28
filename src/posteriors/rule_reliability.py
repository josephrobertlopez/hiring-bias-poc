"""Beta posterior over rule reliability for banking MRM compliance.

Fits Beta distributions on cross-validated rule performance data.
Provides deterministic posterior means and credible intervals.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import StratifiedKFold

from ..rules.data import Resume
from ..features.rule_miner import AssociationRule


@dataclass
class RulePosterior:
    """Beta posterior parameters for a single rule."""
    rule_id: str
    alpha: float  # successes + 1
    beta_param: float  # failures + 1 (renamed to avoid conflict with scipy.stats.beta)
    n_observations: int

    @property
    def posterior_mean(self) -> float:
        """Deterministic posterior mean reliability."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def credible_interval_95(self) -> Tuple[float, float]:
        """95% credible interval."""
        dist = beta(self.alpha, self.beta_param)
        return dist.ppf(0.025), dist.ppf(0.975)


def fit_rule_posteriors(
    rules: List[AssociationRule],
    train_resumes: List[Resume],
    train_labels: List[bool],
    extractor,
    n_folds: int = 5
) -> Dict[str, RulePosterior]:
    """Fit Beta posteriors over rule reliability using cross-validation.

    For banking compliance: deterministic posterior estimates with quantified
    uncertainty, no sampling at prediction time.

    Args:
        rules: Association rules to fit posteriors for
        train_resumes: Training resumes
        train_labels: Training labels (True = advance/hire)
        extractor: Feature extractor for rule evaluation
        n_folds: CV folds for reliability estimation

    Returns:
        Dict mapping rule_id to RulePosterior
    """
    if not rules:
        return {}

    # Cross-validated reliability estimates
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rule_posteriors = {}

    for fold_idx, (_, val_indices) in enumerate(kf.split(train_resumes, train_labels)):
        val_resumes = [train_resumes[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]

        # Evaluate each rule on this fold
        for rule_idx, rule in enumerate(rules):
            rule_id = f"rule_{rule_idx}"

            # Count rule firings and successes
            firings = 0
            successes = 0

            for resume, label in zip(val_resumes, val_labels):
                # Check if rule fires on this resume
                rule_features = extractor.extract_features(resume)

                # Convert rule antecedent to feature check
                # Note: This is a simplified implementation - actual rule firing
                # logic would depend on the rule miner's implementation
                rule_fires = _check_rule_fires(rule, rule_features)

                if rule_fires:
                    firings += 1
                    if label:  # Positive outcome
                        successes += 1

            # Accumulate statistics across folds
            if rule_id not in rule_posteriors:
                rule_posteriors[rule_id] = {'total_firings': 0, 'total_successes': 0}

            rule_posteriors[rule_id]['total_firings'] += firings
            rule_posteriors[rule_id]['total_successes'] += successes

    # Fit Beta posteriors
    fitted_posteriors = {}
    for rule_idx, rule in enumerate(rules):
        rule_id = f"rule_{rule_idx}"

        if rule_id in rule_posteriors:
            stats = rule_posteriors[rule_id]
            successes = stats['total_successes']
            failures = stats['total_firings'] - successes

            # Beta prior: uniform Beta(1, 1), so posterior is Beta(successes + 1, failures + 1)
            fitted_posteriors[rule_id] = RulePosterior(
                rule_id=rule_id,
                alpha=float(successes + 1),
                beta_param=float(failures + 1),
                n_observations=stats['total_firings']
            )
        else:
            # No firing data - use uninformative prior
            fitted_posteriors[rule_id] = RulePosterior(
                rule_id=rule_id,
                alpha=1.0,
                beta_param=1.0,
                n_observations=0
            )

    return fitted_posteriors


def _check_rule_fires(rule: AssociationRule, features: Dict) -> bool:
    """Check if association rule fires given features.

    Simplified implementation - would need to match rule miner's actual logic.
    """
    # This is a placeholder - actual implementation would depend on
    # how the rule miner represents rule antecedents and how they map to features
    return False  # Conservative default