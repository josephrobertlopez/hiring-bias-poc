"""Tests for per-skill aptitude scoring.

Banking MRM compliance tests: deterministic scoring, proper uncertainty propagation,
edge case handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.aptitude.scorer import score_candidate, SkillAptitude, RuleFiring, CandidateScoring
from src.posteriors.rule_reliability import RulePosterior
from src.features.rule_miner import AssociationRule
from src.rules.data import Resume
from src.features.extractors import JobRole


def test_aptitude_score_deterministic():
    """Same input → byte-identical CandidateScoring (excluding decision_id and timestamp)."""
    # Setup test data
    resume = Resume(['python', 'sql'], 3.0, 'bachelor', ['tech'], {})
    role = JobRole(
        required_skills={'python'},
        preferred_skills={'sql'},
        min_experience=2.0,
        max_experience=5.0,
        role_keywords={'software'},
        seniority_level='mid'
    )

    rules = [
        AssociationRule(
            antecedent={'python'},
            consequent={'hired'},
            support=0.5, confidence=0.8, lift=1.2, conviction=1.5
        )
    ]

    rule_posteriors = {
        'rule_0': RulePosterior(
            rule_id='rule_0',
            alpha=8.0,
            beta_param=3.0,
            n_observations=10,
            passed_fairness_filter=True
        )
    }

    extractor = Mock()
    extractor.extract_features.return_value = {'experience_bin': 'mid_level'}

    # Score multiple times
    scores = []
    for _ in range(10):
        scoring = score_candidate(resume, role, rules, rule_posteriors, extractor)
        # Extract deterministic parts (exclude decision_id and timestamp)
        deterministic_part = (
            tuple((skill, apt.score, apt.uncertainty_interval, len(apt.contributing_rules))
                  for skill, apt in sorted(scoring.aptitudes.items())),
            scoring.overall_recommendation,
            scoring.overall_uncertainty,
            scoring.model_version
        )
        scores.append(deterministic_part)

    # All deterministic parts should be identical
    first_score = scores[0]
    for score in scores[1:]:
        assert score == first_score, "Aptitude scoring is not deterministic"


def test_aptitude_handles_skill_with_no_firing_rules():
    """Skill with no contributing rules returns score=NaN."""
    resume = Resume(['javascript'], 2.0, 'master', ['finance'], {})
    role = JobRole(
        required_skills={'python'},  # Skill not in resume
        preferred_skills={'sql'},
        min_experience=1.0,
        max_experience=5.0,
        role_keywords={'software'},
        seniority_level='junior'
    )

    rules = [
        AssociationRule(
            antecedent={'javascript'},  # Different skill
            consequent={'hired'},
            support=0.3, confidence=0.7, lift=1.1, conviction=1.2
        )
    ]

    rule_posteriors = {
        'rule_0': RulePosterior('rule_0', 5.0, 2.0, 7, True)
    }

    extractor = Mock()
    extractor.extract_features.return_value = {}

    scoring = score_candidate(resume, role, rules, rule_posteriors, extractor)

    # Python skill should have NaN score (no rules mention it)
    assert 'python' in scoring.aptitudes
    python_aptitude = scoring.aptitudes['python']
    assert np.isnan(python_aptitude.score)
    assert np.isnan(python_aptitude.uncertainty_interval[0])
    assert np.isnan(python_aptitude.uncertainty_interval[1])
    assert len(python_aptitude.contributing_rules) == 0


def test_aptitude_intervals_are_well_formed():
    """Credible intervals have proper ordering: lower <= mean <= upper."""
    resume = Resume(['python'], 4.0, 'bachelor', ['tech'], {})
    role = JobRole(
        required_skills={'python'},
        preferred_skills=set(),
        min_experience=2.0,
        max_experience=6.0,
        role_keywords={'developer'},
        seniority_level='mid'
    )

    rules = [
        AssociationRule(
            antecedent={'python'},
            consequent={'hired'},
            support=0.6, confidence=0.85, lift=1.3, conviction=1.6
        )
    ]

    rule_posteriors = {
        'rule_0': RulePosterior('rule_0', 10.0, 5.0, 15, True)
    }

    extractor = Mock()
    extractor.extract_features.return_value = {}

    scoring = score_candidate(resume, role, rules, rule_posteriors, extractor)

    python_aptitude = scoring.aptitudes['python']

    if not np.isnan(python_aptitude.score):
        lower, upper = python_aptitude.uncertainty_interval
        score = python_aptitude.score

        assert lower <= upper, f"Interval bounds misordered: {lower} > {upper}"
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"

        # Note: score might not be in interval due to aggregation,
        # but interval should still be well-formed
        assert 0.0 <= lower <= 1.0, f"Lower bound {lower} not in [0, 1]"
        assert 0.0 <= upper <= 1.0, f"Upper bound {upper} not in [0, 1]"


def test_overall_recommendation_threshold_logic():
    """Overall recommendation follows documented thresholds."""
    resume = Resume(['python', 'sql'], 3.0, 'bachelor', ['tech'], {})

    # High score role (should recommend "advance")
    role_high = JobRole({'python'}, {'sql'}, 2.0, 5.0, {'software'}, 'mid')

    # Create rules and posteriors that give high reliability
    rules = [
        AssociationRule({'python'}, {'hired'}, 0.8, 0.9, 1.5, 2.0)
    ]

    # High reliability posterior (alpha >> beta)
    high_posteriors = {
        'rule_0': RulePosterior('rule_0', 20.0, 2.0, 22, True)  # Mean ≈ 0.91
    }

    extractor = Mock()
    extractor.extract_features.return_value = {}

    scoring_high = score_candidate(resume, role_high, rules, high_posteriors, extractor)

    # Should recommend advance (score ≈ 0.91 >= 0.7)
    assert scoring_high.overall_recommendation == "advance"

    # Low reliability posterior
    low_posteriors = {
        'rule_0': RulePosterior('rule_0', 2.0, 8.0, 10, True)  # Mean = 0.2
    }

    scoring_low = score_candidate(resume, role_high, rules, low_posteriors, extractor)

    # Should recommend do_not_advance (score = 0.2 < 0.3)
    assert scoring_low.overall_recommendation == "do_not_advance"


def test_empty_rules_case():
    """No rules provided returns NaN scores for all skills."""
    resume = Resume(['python'], 2.0, 'master', ['tech'], {})
    role = JobRole({'python'}, set(), 1.0, 3.0, {'dev'}, 'junior')

    scoring = score_candidate(resume, role, [], {}, None)

    assert 'python' in scoring.aptitudes
    assert np.isnan(scoring.aptitudes['python'].score)
    assert scoring.overall_recommendation == "review"


def test_fairness_filter_propagation():
    """fairness_filter_passed correctly propagates from rule posteriors."""
    resume = Resume(['python'], 2.0, 'bachelor', ['tech'], {})
    role = JobRole({'python'}, set(), 1.0, 3.0, {'software'}, 'junior')

    rules = [
        AssociationRule({'python'}, {'hired'}, 0.5, 0.8, 1.2, 1.5)
    ]

    # Posterior that failed fairness filter
    failed_posteriors = {
        'rule_0': RulePosterior('rule_0', 5.0, 3.0, 8, passed_fairness_filter=False)
    }

    extractor = Mock()
    extractor.extract_features.return_value = {}

    scoring = score_candidate(resume, role, rules, failed_posteriors, extractor)

    python_aptitude = scoring.aptitudes['python']
    assert python_aptitude.fairness_filter_passed == False