"""Test that hire-rate-as-score discrimination is eliminated."""

import pytest
from src.rules.implementations import EducationRuleImpl, DomainRuleImpl
from src.rules.data import Resume


def test_education_rule_content_neutral():
    """Verify EducationRuleImpl uses content-neutral scoring."""
    rule = EducationRuleImpl()

    # Should not learn hiring rates
    resumes = [Resume(['python'], 3.0, 'bachelor', ['tech'], {})]
    labels = [True]
    rule.fit(resumes, labels)

    # Should not have discriminatory attributes
    assert not hasattr(rule, 'education_scores')
    assert not hasattr(rule, 'education_counts')

    # Should use content-neutral scoring
    bachelor_resume = Resume(['python'], 3.0, 'bachelor', ['tech'], {})
    phd_resume = Resume(['python'], 3.0, 'phd', ['tech'], {})

    bachelor_score = rule.score(bachelor_resume)
    phd_score = rule.score(phd_resume)

    # Scores should be based on role requirements, not hiring rates
    assert bachelor_score == 0.6  # predefined appropriateness
    assert phd_score == 1.0  # predefined appropriateness
    assert bachelor_score != phd_score  # still meaningful differentiation


def test_domain_rule_content_neutral():
    """Verify DomainRuleImpl uses content-neutral scoring."""
    rule = DomainRuleImpl()

    # Should not learn hiring rates
    resumes = [Resume(['python'], 3.0, 'bachelor', ['finance'], {})]
    labels = [True]
    rule.fit(resumes, labels)

    # Should not have discriminatory attributes
    assert not hasattr(rule, 'domain_scores')
    assert not hasattr(rule, 'domain_counts')

    # Should use content-neutral scoring
    tech_resume = Resume(['python'], 3.0, 'bachelor', ['tech'], {})
    finance_resume = Resume(['python'], 3.0, 'bachelor', ['finance'], {})

    tech_score = rule.score(tech_resume)
    finance_score = rule.score(finance_resume)

    # Scores should be based on domain relevance, not hiring rates
    assert tech_score == 1.0  # predefined relevance
    assert finance_score == 0.9  # predefined relevance
    assert tech_score != finance_score  # still meaningful differentiation


def test_no_hiring_rate_in_explanations():
    """Verify explanations don't contain hiring rate data."""
    edu_rule = EducationRuleImpl()
    domain_rule = DomainRuleImpl()

    # Fit rules
    resume = Resume(['python'], 3.0, 'bachelor', ['tech'], {})
    edu_rule.fit([resume], [True])
    domain_rule.fit([resume], [True])

    # Get explanations
    edu_explain = edu_rule.explain(resume)
    domain_explain = domain_rule.explain(resume)

    # Should not contain hiring rate data
    assert 'hiring_rate' not in edu_explain
    assert 'hired_count' not in edu_explain
    assert 'total_count' not in edu_explain

    # Should contain content-neutral data
    assert 'appropriateness_score' in edu_explain
    assert 'content_neutral' in edu_explain

    for domain_data in domain_explain['domains']:
        assert 'hiring_rate' not in domain_data
        assert 'hired_count' not in domain_data
        assert 'total_count' not in domain_data
        assert 'relevance_score' in domain_data
        assert 'content_neutral' in domain_data