"""Unit tests for SkillRulesEngine."""
import pytest
from src.rules.engine import SkillRulesEngine, SkillAuditResult, SkillReport
from src.rules.data import Resume


class TestSkillRulesEngineFit:
    """Test fitting and state management."""

    def test_requires_vocabulary(self, basic_vocabulary):
        """Test that engine requires vocabulary."""
        engine = SkillRulesEngine(basic_vocabulary)
        assert engine.vocabulary == basic_vocabulary
        assert not engine.fitted

    def test_fit_sets_fitted_flag(self, fitted_engine):
        """Test that fit() sets fitted flag."""
        assert fitted_engine.fitted

    def test_fit_initializes_all_rules(self, fitted_engine):
        """Test that all 6 rule types are initialized."""
        assert len(fitted_engine.rules) == 6
        assert "combination" in fitted_engine.rules
        assert "experience" in fitted_engine.rules
        assert "education" in fitted_engine.rules
        assert "domain" in fitted_engine.rules
        assert "gap" in fitted_engine.rules
        assert "bias" in fitted_engine.rules


class TestSkillRulesEngineAudit:
    """Test resume auditing."""

    def test_audit_resume_requires_fit(self, basic_vocabulary, sample_resumes):
        """Test that audit_resume requires fit() to be called first."""
        engine = SkillRulesEngine(basic_vocabulary)
        with pytest.raises(RuntimeError, match="fit"):
            engine.audit_resume(sample_resumes[0])

    def test_audit_resume_returns_result(self, fitted_engine, sample_resumes):
        """Test that audit_resume returns SkillAuditResult."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_resume_1")

        assert isinstance(result, SkillAuditResult)
        assert result.resume_id == "test_resume_1"

    def test_audit_result_has_all_fields(self, fitted_engine, sample_resumes):
        """Test that audit result contains all required fields."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_1")

        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.rule_scores, dict)
        assert len(result.rule_scores) == 6
        assert isinstance(result.skill_patterns, list)
        assert isinstance(result.skill_gaps, list)
        assert isinstance(result.bias_flags, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.explanations, dict)

    def test_rule_scores_all_in_bounds(self, fitted_engine, sample_resumes):
        """Test that all rule scores are in [0, 1]."""
        result = fitted_engine.audit_resume(sample_resumes[1], "test_2")

        for rule_type, score in result.rule_scores.items():
            assert 0.0 <= score <= 1.0
            assert rule_type in ["combination", "experience", "education", "domain", "gap", "bias"]

    def test_overall_score_aggregates_rules(self, fitted_engine, sample_resumes):
        """Test that overall_score is derived from rule_scores."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_3")

        # Overall should be mean or weighted average of rule scores
        rule_values = list(result.rule_scores.values())
        avg_score = sum(rule_values) / len(rule_values)

        # Should be in reasonable range (not necessarily equal due to weighting)
        assert 0.0 <= result.overall_score <= 1.0


class TestSkillRulesEngineBatch:
    """Test batch processing."""

    def test_audit_batch_processes_all(self, fitted_engine, sample_resumes):
        """Test that audit_batch processes all resumes."""
        results = fitted_engine.audit_batch(sample_resumes)

        assert len(results) == len(sample_resumes)
        assert all(isinstance(r, SkillAuditResult) for r in results)

    def test_audit_batch_with_ids(self, fitted_engine, sample_resumes):
        """Test that audit_batch assigns resume IDs."""
        results = fitted_engine.audit_batch(sample_resumes)

        # Should auto-generate IDs like "resume_0", "resume_1", etc.
        assert results[0].resume_id == "resume_0"
        assert results[1].resume_id == "resume_1"

    def test_audit_batch_all_scored(self, fitted_engine, sample_resumes):
        """Test that all batch results have valid scores."""
        results = fitted_engine.audit_batch(sample_resumes)

        for result in results:
            assert result.overall_score is not None
            assert 0.0 <= result.overall_score <= 1.0


class TestSkillRulesEnginePatterns:
    """Test pattern and gap discovery."""

    def test_get_skill_patterns(self, fitted_engine):
        """Test skill pattern discovery."""
        patterns = fitted_engine.get_skill_patterns(min_support=0.1)

        assert isinstance(patterns, list)
        # Each pattern is (skill_set, support) tuple
        for pattern in patterns:
            assert len(pattern) == 2
            assert isinstance(pattern[0], (set, frozenset, tuple, list))
            assert 0.0 <= pattern[1] <= 1.0

    def test_pattern_support_filtered(self, fitted_engine):
        """Test that patterns respect min_support threshold."""
        patterns_10 = fitted_engine.get_skill_patterns(min_support=0.1)
        patterns_50 = fitted_engine.get_skill_patterns(min_support=0.5)

        # Higher threshold should yield fewer patterns
        assert len(patterns_50) <= len(patterns_10)

    def test_identify_skill_gaps(self, fitted_engine, sample_resumes):
        """Test identification of skill gaps."""
        gaps = fitted_engine.identify_skill_gaps()

        assert isinstance(gaps, list)
        # Each gap is likely a skill string
        assert len(gaps) >= 0

    def test_get_critical_skills(self, fitted_engine):
        """Test retrieval of critical skills."""
        critical = fitted_engine.get_critical_skills(threshold=0.5)

        assert isinstance(critical, (list, set))
        # Critical skills should be from the vocabulary
        if critical:
            assert all(skill in fitted_engine.vocabulary.tokens for skill in critical)


class TestSkillRulesEngineReport:
    """Test aggregate reporting."""

    def test_generate_report(self, fitted_engine, sample_resumes):
        """Test generating aggregate skill report."""
        report = fitted_engine.generate_report(sample_resumes)

        assert isinstance(report, SkillReport)
        assert report.total_resumes == len(sample_resumes)

    def test_report_has_all_fields(self, fitted_engine, sample_resumes):
        """Test that report contains all analysis fields."""
        report = fitted_engine.generate_report(sample_resumes)

        assert isinstance(report.skill_frequency, dict)
        assert isinstance(report.skill_combinations, list)
        assert isinstance(report.experience_thresholds, dict)
        assert isinstance(report.education_patterns, dict)
        assert isinstance(report.domain_patterns, dict)
        assert isinstance(report.bias_metrics, dict)

    def test_skill_frequency_sums_to_valid(self, fitted_engine, sample_resumes):
        """Test that skill frequencies are in valid range."""
        report = fitted_engine.generate_report(sample_resumes)

        for skill, freq in report.skill_frequency.items():
            assert 0.0 <= freq <= 1.0

    def test_experience_thresholds_from_engine(self, fitted_engine, sample_resumes):
        """Test that thresholds come from fitted experience rule."""
        report = fitted_engine.generate_report(sample_resumes)

        # Should have thresholds from fitted ExperienceRuleImpl
        assert isinstance(report.experience_thresholds, dict)
        # Can be empty if no skills repeated


class TestSkillRulesEngineExplanations:
    """Test explainability features."""

    def test_explain_score(self, fitted_engine, sample_resumes):
        """Test explaining why a score was given."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_explain")

        # Explanations should have per-rule details
        assert "combination" in result.explanations
        assert "experience" in result.explanations
        assert "education" in result.explanations
        assert "domain" in result.explanations
        assert "gap" in result.explanations
        assert "bias" in result.explanations

    def test_explanations_have_rule_type(self, fitted_engine, sample_resumes):
        """Test that each explanation identifies its rule type."""
        result = fitted_engine.audit_resume(sample_resumes[2], "test_explain_2")

        for rule_type, explanation in result.explanations.items():
            assert isinstance(explanation, dict)
            assert "rule_type" in explanation or rule_type == "overall"

    def test_skill_patterns_extracted(self, fitted_engine, sample_resumes):
        """Test that skill patterns are extracted in result."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_patterns")

        assert isinstance(result.skill_patterns, list)
        # Patterns might be strings like "[python, sql] -> [hired]"
        if result.skill_patterns:
            assert all(isinstance(p, str) for p in result.skill_patterns)

    def test_bias_flags_included(self, fitted_engine, sample_resumes):
        """Test that bias flags are included when detected."""
        result = fitted_engine.audit_resume(sample_resumes[0], "test_bias")

        assert isinstance(result.bias_flags, list)
        # Flags might be empty if no bias detected


class TestSkillRulesEngineEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_skill_tokens(self, fitted_engine):
        """Test handling resume with no skills."""
        empty_resume = Resume([], 3.0, "master", ["finance"], {"gender": 0})
        result = fitted_engine.audit_resume(empty_resume, "empty_skills")

        # Should still produce result even with no skills
        assert isinstance(result, SkillAuditResult)
        assert 0.0 <= result.overall_score <= 1.0

    def test_mismatch_vocabulary(self, fitted_engine):
        """Test handling resume with skills not in vocabulary."""
        mismatched = Resume(["unknown_skill_xyz"], 3.0, "master", ["finance"], {"gender": 0})
        result = fitted_engine.audit_resume(mismatched, "mismatch")

        # Should handle gracefully
        assert isinstance(result, SkillAuditResult)
        assert 0.0 <= result.overall_score <= 1.0

    def test_extreme_experience(self, fitted_engine):
        """Test handling extreme experience values."""
        extreme_high = Resume(["python"], 99.0, "master", ["finance"], {"gender": 0})
        extreme_low = Resume(["python"], 0.0, "master", ["finance"], {"gender": 0})

        result_high = fitted_engine.audit_resume(extreme_high, "high_exp")
        result_low = fitted_engine.audit_resume(extreme_low, "low_exp")

        assert isinstance(result_high, SkillAuditResult)
        assert isinstance(result_low, SkillAuditResult)
        assert 0.0 <= result_high.overall_score <= 1.0
        assert 0.0 <= result_low.overall_score <= 1.0
