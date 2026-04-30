"""Beta posterior over rule reliability."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import StratifiedKFold

from ..rules.data import Resume
from ..features.rule_miner import AssociationRule, FairnessFilteredRuleMiner


@dataclass
class RulePosterior:
    """Beta posterior parameters for a single rule."""
    rule_id: str
    alpha: float
    beta_param: float
    n_observations: int
    passed_fairness_filter: bool

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
    """Fit Beta posteriors over rule reliability using cross-validation."""
    if not rules:
        return {}

    # Initialize fairness filter check - reuse from rule miner
    fairness_filter = FairnessFilteredRuleMiner()

    # Calculate effective folds to handle small datasets
    n_samples = len(train_resumes)
    n_pos = sum(train_labels)
    n_neg = n_samples - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)
    if effective_folds < 2:
        # Single-fold fallback: fit on all data, posterior is wider
        return _fit_single_fold(rules, train_resumes, train_labels, extractor)

    # Cross-validated reliability estimates
    kf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
    rule_stats = {}

    # Initialize statistics for each rule
    for rule_idx, rule in enumerate(rules):
        rule_id = f"rule_{rule_idx}"
        rule_stats[rule_id] = {
            'total_firings': 0,
            'total_successes': 0,
            'passed_fairness_filter': not _rule_contains_protected_attributes(rule, fairness_filter)
        }

    # Cross-validation loop
    for fold_idx, (_, val_indices) in enumerate(kf.split(train_resumes, train_labels)):
        val_resumes = [train_resumes[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]

        # Convert validation resumes to transactions for rule evaluation
        # Use same transaction logic as rule miner
        val_transactions = []
        for resume, label in zip(val_resumes, val_labels):
            transaction = _resume_to_transaction(resume, extractor)
            val_transactions.append((transaction, label))

        # Evaluate each rule on this fold
        for rule_idx, rule in enumerate(rules):
            rule_id = f"rule_{rule_idx}"

            # Count rule firings and successes on this fold
            for transaction, label in val_transactions:
                # Check if rule fires (antecedent subset of transaction)
                rule_fires = rule.antecedent.issubset(transaction)

                if rule_fires:
                    rule_stats[rule_id]['total_firings'] += 1
                    if label:  # Positive outcome (hired/advanced)
                        rule_stats[rule_id]['total_successes'] += 1

    # Fit Beta posteriors from accumulated statistics
    fitted_posteriors = {}
    for rule_idx, rule in enumerate(rules):
        rule_id = f"rule_{rule_idx}"
        stats = rule_stats[rule_id]

        successes = stats['total_successes']
        failures = stats['total_firings'] - successes

        # Beta prior: uniform Beta(1, 1), so posterior is Beta(successes + 1, failures + 1)
        fitted_posteriors[rule_id] = RulePosterior(
            rule_id=rule_id,
            alpha=float(successes + 1),
            beta_param=float(failures + 1),
            n_observations=stats['total_firings'],
            passed_fairness_filter=stats['passed_fairness_filter']
        )

    return fitted_posteriors


def _resume_to_transaction(resume: Resume, extractor) -> set:
    """Convert resume to transaction set for rule evaluation.

    Simplified version of rule miner's transaction logic.
    """
    features = extractor.extract_features(resume)
    transaction = set()

    # Add skill tokens
    transaction.update(resume.skill_tokens)

    # Add binned features that rules might reference
    if 'experience_bin' in features:
        transaction.add(features['experience_bin'])
    if 'education_level' in features:
        transaction.add(features['education_level'])
    if 'seniority_level' in features:
        transaction.add(features['seniority_level'])

    # Add domain background
    transaction.update(resume.domain_background)

    return transaction


def _rule_contains_protected_attributes(rule: AssociationRule, fairness_filter: FairnessFilteredRuleMiner) -> bool:
    """Check if rule contains protected attributes using existing fairness filter."""
    # Check both antecedent and consequent for protected attributes
    all_items = rule.antecedent | rule.consequent

    for item in all_items:
        if fairness_filter._is_protected_attribute(item):
            return True

    return False


def _fit_single_fold(rules: List[AssociationRule], train_resumes: List[Resume],
                    train_labels: List[bool], extractor) -> Dict[str, RulePosterior]:
    """Single-fold fallback for small datasets with wider uncertainty intervals."""
    from ..features.rule_miner import FairnessFilteredRuleMiner

    # Initialize fairness filter
    fairness_filter = FairnessFilteredRuleMiner()

    # Convert resumes to transactions
    transactions = []
    for resume, label in zip(train_resumes, train_labels):
        transaction = _resume_to_transaction(resume, extractor)
        transactions.append((transaction, label))

    posteriors = {}

    # Evaluate each rule on full dataset
    for rule_idx, rule in enumerate(rules):
        rule_id = f"rule_{rule_idx}"

        total_firings = 0
        total_successes = 0

        # Count rule firings and successes on full data
        for transaction, label in transactions:
            rule_fires = rule.antecedent.issubset(transaction)
            if rule_fires:
                total_firings += 1
                if label:  # Hired/advanced
                    total_successes += 1

        # Compute wider posterior (smaller effective sample)
        # Use half the actual observations to increase uncertainty
        if total_firings == 0:
            # Rule never fired: use uninformative prior only
            effective_firings = 0
            effective_successes = 0
            alpha = 1.0  # Prior
            beta_param = 1.0  # Prior
        else:
            # Rule fired: use half observations for wider uncertainty
            effective_firings = max(1, total_firings // 2)
            effective_successes = max(1, total_successes // 2) if total_successes > 0 else 0
            alpha = effective_successes + 1
            beta_param = (effective_firings - effective_successes) + 1

        posteriors[rule_id] = RulePosterior(
            rule_id=rule_id,
            alpha=alpha,
            beta_param=beta_param,
            n_observations=effective_firings,
            passed_fairness_filter=not _rule_contains_protected_attributes(rule, fairness_filter)
        )

    return posteriors