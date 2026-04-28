"""Test that hire-rate-as-score discrimination is eliminated."""

import pytest
from src.rules.implementations import EducationRuleImpl, DomainRuleImpl
from src.rules.data import Resume


def test_education_rule_content_neutral():
    """Verify EducationRuleImpl uses truly neutral scoring (no prestige bias)."""
    rule = EducationRuleImpl()

    # Should not learn hiring rates
    resumes = [Resume(['python'], 3.0, 'bachelor', ['tech'], {})]
    labels = [True]
    rule.fit(resumes, labels)

    # Should not have discriminatory attributes
    assert not hasattr(rule, 'education_scores')
    assert not hasattr(rule, 'education_counts')

    # Should use completely neutral scoring - no prestige bias
    bachelor_resume = Resume(['python'], 3.0, 'bachelor', ['tech'], {})
    phd_resume = Resume(['python'], 3.0, 'phd', ['tech'], {})

    bachelor_score = rule.score(bachelor_resume)
    phd_score = rule.score(phd_resume)

    # All education levels get neutral score - let EBM handle as categorical
    assert bachelor_score == 0.5  # neutral, no prestige bias
    assert phd_score == 0.5  # neutral, no prestige bias
    assert bachelor_score == phd_score  # truly equal treatment


def test_domain_rule_content_neutral():
    """Verify DomainRuleImpl uses truly neutral scoring (no domain bias)."""
    rule = DomainRuleImpl()

    # Should not learn hiring rates
    resumes = [Resume(['python'], 3.0, 'bachelor', ['finance'], {})]
    labels = [True]
    rule.fit(resumes, labels)

    # Should not have discriminatory attributes
    assert not hasattr(rule, 'domain_scores')
    assert not hasattr(rule, 'domain_counts')

    # Should use completely neutral scoring - no domain bias
    tech_resume = Resume(['python'], 3.0, 'bachelor', ['tech'], {})
    finance_resume = Resume(['python'], 3.0, 'bachelor', ['finance'], {})

    tech_score = rule.score(tech_resume)
    finance_score = rule.score(finance_resume)

    # All domains get neutral score - let EBM handle as categorical
    assert tech_score == 0.5  # neutral, no domain bias
    assert finance_score == 0.5  # neutral, no domain bias
    assert tech_score == finance_score  # truly equal treatment


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

    # Should contain basic info without discriminatory scoring
    assert 'rule_type' in edu_explain
    assert 'education_level' in edu_explain

    # Domain explanations should not contain hiring rates
    assert 'domains' in domain_explain
    for domain_data in domain_explain['domains']:
        assert 'hiring_rate' not in domain_data
        assert 'hired_count' not in domain_data
        assert 'total_count' not in domain_data