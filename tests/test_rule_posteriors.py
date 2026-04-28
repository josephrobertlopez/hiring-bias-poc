"""Tests for Beta posterior rule reliability fitting.

Banking MRM compliance tests: deterministic behavior, proper uncertainty,
fairness filter integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.posteriors.rule_reliability import fit_rule_posteriors, RulePosterior, _rule_contains_protected_attributes
from src.features.rule_miner import AssociationRule, FairnessFilteredRuleMiner
from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, JobRole


def test_posterior_is_deterministic():
    """Same input → byte-identical output across 100 calls."""
    # Create minimal test data
    rules = [
        AssociationRule(
            antecedent={'python'},
            consequent={'hired'},
            support=0.5, confidence=0.8, lift=1.2, conviction=1.5
        )
    ]

    resumes = [
        Resume(['python'], 3.0, 'bachelor', ['tech'], {}),
        Resume(['java'], 2.0, 'master', ['finance'], {})
    ]

    labels = [True, False]

    # Mock extractor
    extractor = Mock()
    extractor.extract_features.return_value = {'experience_bin': 'mid_level'}

    # Run 100 times and check determinism
    results = []
    for _ in range(100):
        posteriors = fit_rule_posteriors(rules, resumes, labels, extractor, n_folds=2)
        # Extract key values for comparison
        if 'rule_0' in posteriors:
            result = (
                posteriors['rule_0'].alpha,
                posteriors['rule_0'].beta_param,
                posteriors['rule_0'].n_observations,
                posteriors['rule_0'].posterior_mean,
                posteriors['rule_0'].credible_interval_95
            )
            results.append(result)

    # All results should be identical
    if results:
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Posterior fitting is not deterministic"


def test_posterior_mean_is_within_interval():
    """Sanity check: posterior mean is within credible interval."""
    posterior = RulePosterior(
        rule_id="test_rule",
        alpha=5.0,
        beta_param=3.0,
        n_observations=10,
        passed_fairness_filter=True
    )

    mean = posterior.posterior_mean
    lower, upper = posterior.credible_interval_95

    assert lower <= mean <= upper, f"Mean {mean} not in interval [{lower}, {upper}]"
    assert 0.0 <= mean <= 1.0, f"Mean {mean} not in [0, 1]"
    assert 0.0 <= lower <= 1.0, f"Lower bound {lower} not in [0, 1]"
    assert 0.0 <= upper <= 1.0, f"Upper bound {upper} not in [0, 1]"


def test_posterior_intervals_widen_with_less_data():
    """Synthetic check: uncertainty increases with less data."""
    # High data posterior
    high_data = RulePosterior("rule_high", alpha=50.0, beta_param=50.0, n_observations=100, passed_fairness_filter=True)

    # Low data posterior (same ratio)
    low_data = RulePosterior("rule_low", alpha=5.0, beta_param=5.0, n_observations=10, passed_fairness_filter=True)

    high_interval = high_data.credible_interval_95
    low_interval = low_data.credible_interval_95

    high_width = high_interval[1] - high_interval[0]
    low_width = low_interval[1] - low_interval[0]

    assert low_width > high_width, f"Low data interval ({low_width}) should be wider than high data ({high_width})"


def test_no_protected_attributes_in_fitted_rules():
    """Invariant: fairness filter properly identifies protected attributes."""
    fairness_filter = FairnessFilteredRuleMiner()

    # Rule with protected attribute
    protected_rule = AssociationRule(
        antecedent={'gender'},
        consequent={'hired'},
        support=0.3, confidence=0.7, lift=1.1, conviction=1.2
    )

    # Rule without protected attributes
    clean_rule = AssociationRule(
        antecedent={'python'},
        consequent={'hired'},
        support=0.5, confidence=0.8, lift=1.3, conviction=1.4
    )

    assert _rule_contains_protected_attributes(protected_rule, fairness_filter) == True
    assert _rule_contains_protected_attributes(clean_rule, fairness_filter) == False


def test_empty_rules_list():
    """Edge case: empty rules list returns empty posteriors."""
    rules = []
    resumes = [Resume(['python'], 3.0, 'bachelor', ['tech'], {})]
    labels = [True]
    extractor = Mock()

    posteriors = fit_rule_posteriors(rules, resumes, labels, extractor)
    assert posteriors == {}


def test_rule_with_zero_firings():
    """Edge case: rule that never fires gets uninformative posterior."""
    rules = [
        AssociationRule(
            antecedent={'nonexistent_skill'},
            consequent={'hired'},
            support=0.0, confidence=0.0, lift=1.0, conviction=1.0
        )
    ]

    resumes = [Resume(['python'], 3.0, 'bachelor', ['tech'], {})]
    labels = [True]

    extractor = Mock()
    extractor.extract_features.return_value = {}

    posteriors = fit_rule_posteriors(rules, resumes, labels, extractor, n_folds=2)

    assert 'rule_0' in posteriors
    posterior = posteriors['rule_0']
    assert posterior.n_observations == 0
    assert posterior.alpha == 1.0  # Prior
    assert posterior.beta_param == 1.0  # Prior
    assert abs(posterior.posterior_mean - 0.5) < 1e-6  # Uniform prior mean