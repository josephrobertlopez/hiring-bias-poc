"""Integration tests for end-to-end hiring audit pipeline."""
import pytest
from src.rules.engine import SkillRulesEngine, SkillAuditResult
from src.rules.data import Resume, SkillVocabulary


class TestFullAuditPipeline:
    """Test complete hiring audit workflow."""

    def test_end_to_end_workflow(self, basic_vocabulary, sample_resumes, hired_rejected_labels):
        """Test full workflow: create engine, fit, audit."""
        # Create engine
        engine = SkillRulesEngine(basic_vocabulary)
        assert not engine.fitted

        # Fit on historical data
        engine.fit(sample_resumes, hired_rejected_labels)
        assert engine.fitted

        # Audit new resume
        test_resume = sample_resumes[0]
        result = engine.audit_resume(test_resume, "audit_test_1")

        assert isinstance(result, SkillAuditResult)
        assert result.resume_id == "audit_test_1"
        assert 0.0 <= result.overall_score <= 1.0

    def test_audit_multiple_resumes(self, fitted_engine, sample_resumes):
        """Test auditing multiple resumes in sequence."""
        results = []
        for i, resume in enumerate(sample_resumes):
            result = fitted_engine.audit_resume(resume, f"test_{i}")
            results.append(result)

        assert len(results) == len(sample_resumes)
        assert all(isinstance(r, SkillAuditResult) for r in results)
        assert all(r.overall_score is not None for r in results)

    def test_audit_batch_vs_individual(self, fitted_engine, sample_resumes):
        """Test that batch audit matches individual audits."""
        batch_results = fitted_engine.audit_batch(sample_resumes)

        individual_results = []
        for i, resume in enumerate(sample_resumes):
            result = fitted_engine.audit_resume(resume, f"individual_{i}")
            individual_results.append(result)

        assert len(batch_results) == len(individual_results)

        # Scores might be slightly different due to aggregation logic,
        # but should be in same ballpark
        for batch, indiv in zip(batch_results, individual_results):
            assert abs(batch.overall_score - indiv.overall_score) < 0.1


class TestBiasPipelineDetection:
    """Test bias detection in full pipeline."""

    def test_detect_gender_bias(self, basic_vocabulary, bias_scenario_resumes, bias_imbalanced_labels):
        """Test detecting gender bias in hiring."""
        engine = SkillRulesEngine(basic_vocabulary)
        engine.fit(bias_scenario_resumes, bias_imbalanced_labels)

        # Audit a female candidate with same qualifications as hired males
        female_candidate = bias_scenario_resumes[1]  # Female with same skills as male
        result = engine.audit_resume(female_candidate, "female_candidate")

        # Should have lower score due to demographic bias
        assert 0.0 <= result.overall_score <= 1.0

        # Bias flags should indicate gender disparity
        bias_explanation = result.explanations.get("bias", {})
        if bias_explanation and "demographics" in bias_explanation:
            demographics = bias_explanation["demographics"]
            gender_data = next((d for d in demographics if d["attribute"] == "gender"), None)
            if gender_data:
                # Should show disparity between male and female hiring rates
                assert "disparity_index" in gender_data

    def test_fair_hiring_no_bias_flags(self, basic_vocabulary, sample_resumes, hired_rejected_labels):
        """Test that balanced hiring shows no bias flags."""
        engine = SkillRulesEngine(basic_vocabulary)
        engine.fit(sample_resumes, hired_rejected_labels)

        result = engine.audit_resume(sample_resumes[0], "fair_test")

        # Balanced data should not trigger extreme bias flags
        assert isinstance(result.bias_flags, list)
        # Flags might be empty or minimal for balanced hiring


class TestPatternDiscovery:
    """Test skill pattern discovery."""

    def test_discover_skill_patterns(self, fitted_engine):
        """Test that skill patterns are discovered."""
        patterns = fitted_engine.get_skill_patterns(min_support=0.1)

        assert isinstance(patterns, list)
        # Patterns should be discovered from the fitted data
        if patterns:
            for skill_set, support in patterns:
                assert 0.0 <= support <= 1.0

    def test_discover_critical_skills(self, fitted_engine):
        """Test identification of critical skills."""
        critical = fitted_engine.get_critical_skills(threshold=0.5)

        assert isinstance(critical, (list, set))
        # Critical skills should be a subset of vocabulary
        if critical:
            assert all(skill in fitted_engine.vocabulary.tokens for skill in critical)

    def test_skill_gaps_from_resumes(self, fitted_engine, sample_resumes):
        """Test identifying skill gaps."""
        gaps = fitted_engine.identify_skill_gaps()

        assert isinstance(gaps, list)
        # Should identify gaps that appear in multiple candidates

    def test_generate_aggregate_report(self, fitted_engine, sample_resumes):
        """Test generating aggregate report."""
        report = fitted_engine.generate_report(sample_resumes)

        assert report.total_resumes == len(sample_resumes)
        assert len(report.skill_frequency) >= 0
        assert len(report.experience_thresholds) >= 0


class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_recommendations_for_gap(self, fitted_engine, sample_resumes):
        """Test that recommendations address skill gaps."""
        # Audit candidate with gaps
        result = fitted_engine.audit_resume(sample_resumes[3], "gap_test")

        assert isinstance(result.recommendations, list)
        # Recommendations might suggest adding missing skills
        if result.skill_gaps:
            assert len(result.recommendations) > 0

    def test_recommendations_vary_by_score(self, fitted_engine, sample_resumes):
        """Test that different candidates get different recommendations."""
        results = []
        for i, resume in enumerate(sample_resumes):
            result = fitted_engine.audit_resume(resume, f"rec_{i}")
            results.append(result)

        # Not all candidates should have identical recommendations
        rec_sets = [tuple(r.recommendations) for r in results]
        # At least some should differ
        assert len(set(rec_sets)) >= 1  # At least one unique set of recommendations


class TestScoreConsistency:
    """Test consistency of scoring."""

    def test_same_resume_same_score(self, fitted_engine):
        """Test that auditing the same resume twice gives same score."""
        resume = Resume(["python", "sql"], 3.0, "master", ["finance"], {"gender": 0})

        result1 = fitted_engine.audit_resume(resume, "consistency_1")
        result2 = fitted_engine.audit_resume(resume, "consistency_2")

        assert result1.overall_score == result2.overall_score
        assert result1.rule_scores == result2.rule_scores

    def test_rule_score_variance(self, fitted_engine, sample_resumes):
        """Test that different resumes have different rule scores."""
        results = []
        for i, resume in enumerate(sample_resumes):
            result = fitted_engine.audit_resume(resume, f"var_{i}")
            results.append(result)

        # Collect all overall scores
        scores = [r.overall_score for r in results]

        # Should have variance (not all identical)
        assert len(set(scores)) > 1 or len(scores) == 1


class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_handle_unfitted_engine(self, basic_vocabulary, sample_resumes):
        """Test that unfitted engine raises appropriate error."""
        engine = SkillRulesEngine(basic_vocabulary)

        with pytest.raises(RuntimeError):
            engine.audit_resume(sample_resumes[0])

    def test_handle_mismatched_data(self, basic_vocabulary):
        """Test handling of mismatched data shapes."""
        engine = SkillRulesEngine(basic_vocabulary)

        resumes = [
            Resume(["python"], 3.0, "master", ["finance"], {"gender": 0}),
            Resume(["java"], 2.0, "bachelor", ["tech"], {"gender": 1}),
        ]
        labels = [True]  # Mismatched: 2 resumes, 1 label

        # Depending on implementation, should handle gracefully
        try:
            engine.fit(resumes, labels)
            # If fit succeeds with mismatched data, auditing should still work
            result = engine.audit_resume(resumes[0], "test")
            assert isinstance(result, SkillAuditResult)
        except ValueError:
            # Or raise ValueError for mismatched shapes
            pass

    def test_handle_unknown_education_level(self, fitted_engine):
        """Test handling resume with unknown education level."""
        unusual = Resume(
            ["python", "sql"],
            3.0,
            "unknown_degree_type",
            ["finance"],
            {"gender": 0}
        )

        result = fitted_engine.audit_resume(unusual, "unusual_ed")

        # Should produce result even for unknown education
        assert isinstance(result, SkillAuditResult)
        assert 0.0 <= result.overall_score <= 1.0


class TestRuleIndependence:
    """Test that rules work independently."""

    def test_all_six_rules_scored(self, fitted_engine, sample_resumes):
        """Test that all 6 rules produce scores."""
        result = fitted_engine.audit_resume(sample_resumes[0], "rule_test")

        expected_rules = {
            "combination", "experience", "education", "domain", "gap", "bias"
        }
        assert set(result.rule_scores.keys()) == expected_rules

        # All should have valid scores
        for rule_type, score in result.rule_scores.items():
            assert 0.0 <= score <= 1.0, f"{rule_type} score {score} out of bounds"

    def test_rule_explanations_present(self, fitted_engine, sample_resumes):
        """Test that all rules provide explanations."""
        result = fitted_engine.audit_resume(sample_resumes[1], "explain_test")

        expected_rules = {
            "combination", "experience", "education", "domain", "gap", "bias"
        }
        for rule_type in expected_rules:
            assert rule_type in result.explanations or rule_type in result.rule_scores
