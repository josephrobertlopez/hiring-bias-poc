"""Unit tests for rule implementations."""
import pytest
from src.rules.implementations import (
    CombinationRuleImpl,
    ExperienceRuleImpl,
    EducationRuleImpl,
    DomainRuleImpl,
    GapRuleImpl,
    BiasRuleImpl,
)
from src.rules.data import Resume


class TestCombinationRuleImpl:
    """Test association rules for skill combinations."""

    def test_fit_sets_fitted_flag(self, sample_resumes, hired_rejected_labels):
        """Test that fit() sets fitted flag."""
        rule = CombinationRuleImpl()
        assert not rule.fitted
        rule.fit(sample_resumes, hired_rejected_labels)
        assert rule.fitted

    def test_learns_positive_negative_skills(self, sample_resumes, hired_rejected_labels):
        """Test that fit() learns positive and negative skills."""
        rule = CombinationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # Should have some positive skills (from hired candidates)
        assert len(rule.positive_skills) > 0
        # Positive skills should come from hired resumes
        assert "python" in rule.positive_skills  # Resumes 0 and 2 hired

    def test_score_within_bounds(self, sample_resumes, hired_rejected_labels):
        """Test that score() returns [0, 1]."""
        rule = CombinationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        for resume in sample_resumes:
            score = rule.score(resume)
            assert 0.0 <= score <= 1.0

    def test_score_favors_positive_skills(self, sample_resumes, hired_rejected_labels):
        """Test that resumes with positive skills score higher."""
        rule = CombinationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # Resume 0 (hired, python+sql) vs Resume 3 (rejected, sql only)
        score_hired = rule.score(sample_resumes[0])
        score_rejected = rule.score(sample_resumes[3])

        # Hired should generally score >= rejected
        assert score_hired >= score_rejected

    def test_explain_returns_dict(self, sample_resumes, hired_rejected_labels):
        """Test that explain() returns proper structure."""
        rule = CombinationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        explanation = rule.explain(sample_resumes[0])
        assert isinstance(explanation, dict)
        assert "rule_type" in explanation
        assert explanation["rule_type"] == "combination"
        assert "triggered_rules" in explanation


class TestExperienceRuleImpl:
    """Test experience thresholds per skill."""

    def test_fit_learns_thresholds(self, sample_resumes, hired_rejected_labels):
        """Test that fit() learns experience thresholds."""
        rule = ExperienceRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        assert rule.fitted
        assert len(rule.skill_experience_thresholds) > 0
        # Python appears in hired resumes (indices 0, 2)
        assert "python" in rule.skill_experience_thresholds

    def test_threshold_is_median(self, sample_resumes, hired_rejected_labels):
        """Test that learned threshold is median of hired candidates."""
        rule = ExperienceRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # Hired candidates (0, 2) have python with years [3.0, 5.0]
        # Median should be 4.0
        python_threshold = rule.skill_experience_thresholds.get("python")
        assert python_threshold == 4.0

    def test_score_based_on_experience(self, sample_resumes, hired_rejected_labels):
        """Test that score reflects experience adequacy."""
        rule = ExperienceRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # Sample 0 has 3.0 years (< threshold 4.0 for python)
        score = rule.score(sample_resumes[0])
        assert 0.0 <= score <= 1.0
        # Score should reflect: actual (3.0) / threshold (4.0) = 0.75
        assert score < 1.0  # Below threshold

    def test_matches_validates_experience(self, sample_resumes, hired_rejected_labels):
        """Test matches() checks if resume meets experience thresholds."""
        rule = ExperienceRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # Resume 2 has 5.0 years, threshold 4.0 for python - should match
        assert rule.matches(sample_resumes[2])

        # Resume 0 has 3.0 years, threshold 4.0 for python - should not match
        assert not rule.matches(sample_resumes[0])

    def test_explain_shows_gaps(self, sample_resumes, hired_rejected_labels):
        """Test that explain() details experience gaps."""
        rule = ExperienceRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        explanation = rule.explain(sample_resumes[0])
        assert explanation["rule_type"] == "experience"
        assert "skills" in explanation
        # Should show python gap
        python_skill = next((s for s in explanation["skills"] if s["skill"] == "python"), None)
        assert python_skill is not None
        assert python_skill["gap"] < 0  # Negative gap = shortfall


class TestEducationRuleImpl:
    """Test education level hiring patterns."""

    def test_education_rule_neutral_scoring(self, sample_resumes, hired_rejected_labels):
        """Test that education scoring is neutral (no prestige bias)."""
        rule = EducationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # All education levels should get neutral score
        for resume in sample_resumes:
            assert rule.score(resume) == 0.5

    def test_matches_checks_training_data(self, sample_resumes, hired_rejected_labels):
        """Test that matches() checks if education level in training data."""
        rule = EducationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        assert rule.matches(sample_resumes[0])  # master in training
        # Create unknown education level
        unknown = Resume(["python"], 3.0, "unknown_degree", ["tech"], {"gender": 0})
        assert not rule.matches(unknown)

    def test_explain_shows_education_info(self, sample_resumes, hired_rejected_labels):
        """Test that explain() shows education info without hiring rates."""
        rule = EducationRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        explanation = rule.explain(sample_resumes[0])
        assert explanation["rule_type"] == "education"
        assert explanation["education_level"] == "master"
        # No hiring rate in neutral explanation


class TestDomainRuleImpl:
    """Test domain background patterns."""

    def test_domain_rule_neutral_scoring(self, sample_resumes, hired_rejected_labels):
        """Test that domain scoring is neutral (no domain bias)."""
        rule = DomainRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        # All domains should get neutral score
        for resume in sample_resumes:
            assert rule.score(resume) == 0.5

    def test_matches_checks_any_domain(self, sample_resumes, hired_rejected_labels):
        """Test that matches() returns true if any domain in training."""
        rule = DomainRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        assert rule.matches(sample_resumes[0])  # finance in training

        unknown = Resume(["python"], 3.0, "master", ["unknown_domain"], {"gender": 0})
        assert not rule.matches(unknown)

    def test_explain_shows_domain_analysis(self, sample_resumes, hired_rejected_labels):
        """Test that explain() details domain stats."""
        rule = DomainRuleImpl()
        rule.fit(sample_resumes, hired_rejected_labels)

        explanation = rule.explain(sample_resumes[0])
        assert explanation["rule_type"] == "domain"
        assert "domains" in explanation
        assert len(explanation["domains"]) > 0


class TestGapRuleImpl:
    """Test critical skill gap detection."""

    def test_fit_identifies_critical_skills(self, sample_resumes, hired_rejected_labels):
        """Test that fit() identifies critical skills from hired candidates."""
        rule = GapRuleImpl(threshold=0.5)
        rule.fit(sample_resumes, hired_rejected_labels)

        assert rule.fitted
        # Hired: resumes 0, 2
        # Skills: python (2x), sql (1x), machine_learning (1x)
        # At threshold 0.5: python appears in 2/2 = 100% > 50%
        assert "python" in rule.critical_skills

    def test_score_penalizes_gaps(self, sample_resumes, hired_rejected_labels):
        """Test that score penalizes missing critical skills."""
        rule = GapRuleImpl(threshold=0.5)
        rule.fit(sample_resumes, hired_rejected_labels)

        # Resume 3 (only sql) vs Resume 2 (python + machine_learning)
        gap_score = rule.score(sample_resumes[3])
        complete_score = rule.score(sample_resumes[2])

        # Gap score should be lower
        assert gap_score < complete_score

    def test_score_neutral_when_complete(self, sample_resumes, hired_rejected_labels):
        """Test that score is 0.5 when all critical skills present."""
        rule = GapRuleImpl(threshold=0.5)
        rule.fit(sample_resumes, hired_rejected_labels)

        # Resume 2 has python + machine_learning (both critical)
        score = rule.score(sample_resumes[2])
        assert score == 0.5  # Neutral to let other rules discriminate

    def test_matches_checks_critical_skills(self, sample_resumes, hired_rejected_labels):
        """Test that matches() verifies all critical skills present."""
        rule = GapRuleImpl(threshold=0.5)
        rule.fit(sample_resumes, hired_rejected_labels)

        # Resume 2 has all critical skills
        assert rule.matches(sample_resumes[2])

        # Resume 3 (only sql) missing python
        assert not rule.matches(sample_resumes[3])

    def test_explain_lists_missing_skills(self, sample_resumes, hired_rejected_labels):
        """Test that explain() shows which skills are missing."""
        rule = GapRuleImpl(threshold=0.5)
        rule.fit(sample_resumes, hired_rejected_labels)

        explanation = rule.explain(sample_resumes[3])
        assert explanation["rule_type"] == "gap"
        assert "missing_skills" in explanation
        assert "python" in explanation["missing_skills"]


class TestBiasRuleImpl:
    """Test bias detection via fairness metrics."""

    def test_fit_tracks_demographics(self, bias_scenario_resumes, bias_imbalanced_labels):
        """Test that fit() tracks demographic groups."""
        rule = BiasRuleImpl()
        rule.fit(bias_scenario_resumes, bias_imbalanced_labels)

        assert rule.fitted
        assert "gender" in rule.demographic_groups
        # 2 males hired, 2 females rejected
        assert rule.demographic_groups["gender"][0] == 2  # Males hired
        assert rule.demographic_groups["gender"][1] == 0  # Females hired

    def test_bias_rule_neutral_scoring(self, bias_scenario_resumes, bias_imbalanced_labels):
        """Test that bias rule provides neutral scoring (no discrimination)."""
        rule = BiasRuleImpl()
        rule.fit(bias_scenario_resumes, bias_imbalanced_labels)

        # All resumes should get neutral score - bias detection is for monitoring, not scoring
        for resume in bias_scenario_resumes:
            assert rule.score(resume) == 0.5

    def test_detect_extreme_bias(self, bias_scenario_resumes, bias_imbalanced_labels):
        """Test that explain() detects extreme gender bias."""
        rule = BiasRuleImpl()
        rule.fit(bias_scenario_resumes, bias_imbalanced_labels)

        explanation = rule.explain(bias_scenario_resumes[0])
        assert explanation["rule_type"] == "bias"

        # Should show gender analysis with disparity
        gender_data = next(
            (d for d in explanation["demographics"] if d["attribute"] == "gender"),
            None
        )
        assert gender_data is not None
        # Disparity index should be low (0.0) for extreme bias
        assert gender_data["disparity_index"] == 0.0

    def test_no_bias_when_equal_rates(self, sample_resumes, hired_rejected_labels):
        """Test that equal hiring rates show no bias."""
        # Create balanced scenario: 50% male hired, 50% female hired
        balanced_resumes = [
            Resume(["python"], 3.0, "master", ["finance"], {"gender": 0}),  # Male hired
            Resume(["java"], 2.0, "bachelor", ["tech"], {"gender": 1}),     # Female hired
        ]
        balanced_labels = [True, True]

        rule = BiasRuleImpl()
        rule.fit(balanced_resumes, balanced_labels)

        explanation = rule.explain(balanced_resumes[0])
        gender_data = next(
            (d for d in explanation["demographics"] if d["attribute"] == "gender"),
            None
        )
        # Both groups at 100%, DI = 1.0
        assert gender_data["disparity_index"] == 1.0
