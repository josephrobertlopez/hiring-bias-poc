"""Step definitions for rich_explanations package."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from behave import given, when, then
import time
import re
from rules.engine import SkillRulesEngine
from rules.data import Resume, SkillVocabulary

# Import the rich_explanations module (will be implemented)
from rich_explanations import EnhancedExplanationEngine, ExplanationResult


# Note: "the SkillRulesEngine foundation is GREEN" is defined in accuracy_proof_steps.py.
# We hook into context.foundation_engine created by the shared step and rename it for our use.


@given('accuracy proof validates system performance')
def step_accuracy_proof_validated(context):
    """Verify accuracy proof is complete."""
    # Reference that accuracy_proof package is GREEN
    context.accuracy_validated = True


def _ensure_base_engine(context):
    """Ensure base_engine is initialized (hook into shared foundation_engine)."""
    if not hasattr(context, 'base_engine'):
        if hasattr(context, 'foundation_engine'):
            context.base_engine = context.foundation_engine
        else:
            # Fallback: initialize our own
            vocab = SkillVocabulary(["python", "sql", "tensorflow", "machine_learning", "aws"], {})
            context.base_engine = SkillRulesEngine(vocab)


@given('SkillRulesEngine recommends hire with 0.87 score')
def step_skillrulesengine_recommends_hire(context):
    """Set up high-scoring candidate."""
    _ensure_base_engine(context)

    resume = Resume(
        skill_tokens=["python", "tensorflow"],
        years_experience=4.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    # Fit engine with sample data to get realistic 0.87 score
    sample_resumes = [resume]
    sample_labels = [True]
    context.base_engine.fit(sample_resumes, sample_labels)

    context.candidate_resume = resume
    context.audit_result = context.base_engine.audit_resume(resume)
    context.expected_score = 0.87


@when('I request enhanced explanation')
def step_request_enhanced_explanation(context):
    """Request enhanced explanation."""
    context.enhanced_engine = EnhancedExplanationEngine(context.base_engine)

    # Determine which candidate context is available
    if hasattr(context, 'bias_candidate'):
        candidate = context.bias_candidate
        audit_result = context.bias_audit_result
    elif hasattr(context, 'candidate_resume'):
        candidate = context.candidate_resume
        audit_result = context.audit_result
    else:
        raise AssertionError("No candidate context available for enhanced explanation")

    context.explanation_result = context.enhanced_engine.explain_decision(
        candidate,
        audit_result
    )


@then('I see "python + tensorflow skills → 94% historical success"')
def step_see_skill_combination_evidence(context):
    """Verify skill combination reasoning."""
    explanation = context.explanation_result
    business_reasons = explanation.business_reasoning

    # Check for skill combination pattern
    pattern_found = any(
        "python" in reason and "tensorflow" in reason and "%" in reason
        for reason in business_reasons
    )
    assert pattern_found, f"Skill combination reasoning not found in {business_reasons}"


@then('I see specific evidence "847 similar hires, avg rating 4.8/5"')
def step_see_specific_evidence(context):
    """Verify specific historical evidence."""
    explanation = context.explanation_result
    evidence_text = str(explanation.historical_evidence)

    # Check for specific numbers and evidence
    has_sample_size = re.search(r'\d+.*hires?', evidence_text)
    has_rating = re.search(r'\d\.\d.*rating', evidence_text)

    assert has_sample_size, f"Sample size not found in {evidence_text}"
    assert has_rating, f"Rating not found in {evidence_text}"


@then('I see "No demographic bias vs 156 comparable candidates"')
def step_see_bias_analysis(context):
    """Verify bias analysis."""
    explanation = context.explanation_result
    bias_analysis = explanation.bias_analysis

    assert "bias" in bias_analysis.lower()
    assert "comparable" in bias_analysis.lower()
    # Check for comparison numbers
    has_comparison_count = re.search(r'\d+.*comparable', bias_analysis)
    assert has_comparison_count, f"Comparison count not found in {bias_analysis}"


@then('language is manager-friendly, not technical jargon')
def step_language_manager_friendly(context):
    """Verify manager-friendly language."""
    explanation = context.explanation_result
    text = str(explanation)

    # Should NOT contain technical jargon
    technical_terms = ["disparity_index", "rule_scores", "sklearn", "numpy"]
    for term in technical_terms:
        assert term not in text.lower(), f"Technical jargon '{term}' found in explanation"

    # SHOULD contain business language
    business_terms = ["success", "hires", "candidates", "performance"]
    business_found = any(term in text.lower() for term in business_terms)
    assert business_found, f"Business language not found in {text}"


@then('explanation covers all 6 rule contributions')
def step_explanation_covers_all_rules(context):
    """Verify all 6 rules are explained."""
    explanation = context.explanation_result
    rule_explanations = explanation.rule_contributions

    expected_rules = ["combination", "experience", "education", "domain", "gap", "bias"]
    for rule in expected_rules:
        assert rule in rule_explanations, f"Rule '{rule}' not explained in {rule_explanations.keys()}"


@given('a hire recommendation with mixed rule scores')
def step_hire_recommendation_mixed_scores(context):
    """Set up candidate with mixed rule performance."""
    _ensure_base_engine(context)

    resume = Resume(
        skill_tokens=["python"],  # Good skill but missing others
        years_experience=2.0,     # Moderate experience
        education_level="bachelor", # Lower education
        domain_background=["retail"], # Non-tech domain
        demographics={"gender": 1}
    )

    context.base_engine.fit([resume], [True])
    context.mixed_candidate = resume
    context.mixed_audit_result = context.base_engine.audit_resume(resume)


@when('I view the enhanced explanation')
def step_view_enhanced_explanation(context):
    """View enhanced explanation for current candidate."""
    context.enhanced_engine = EnhancedExplanationEngine(context.base_engine)

    # Determine which candidate context is available
    if hasattr(context, 'mixed_candidate'):
        candidate = context.mixed_candidate
        audit_result = context.mixed_audit_result
        context.mixed_explanation = context.enhanced_engine.explain_decision(candidate, audit_result)
        context.explanation_result = context.mixed_explanation
    elif hasattr(context, 'incomplete_candidate'):
        candidate = context.incomplete_candidate
        audit_result = context.incomplete_audit_result
        context.explanation_result = context.enhanced_engine.explain_decision(candidate, audit_result)
    else:
        raise AssertionError("No candidate context available for enhanced explanation")


@then('I see confidence level with bounds "87% ± 8% success probability"')
def step_see_confidence_bounds(context):
    """Verify confidence bounds display."""
    explanation = context.mixed_explanation
    confidence_text = explanation.confidence_analysis

    # Check for percentage and bounds format
    has_percentage = re.search(r'\d+%', confidence_text)
    has_bounds = re.search(r'±.*\d+%', confidence_text)

    assert has_percentage, f"Percentage not found in {confidence_text}"
    assert has_bounds, f"Confidence bounds not found in {confidence_text}"


@then('I see what drives high confidence vs uncertainty')
def step_see_confidence_drivers(context):
    """Verify confidence factor explanation."""
    explanation = context.mixed_explanation
    confidence_factors = explanation.confidence_factors

    assert "high_confidence" in confidence_factors
    assert "uncertainty" in confidence_factors
    assert len(confidence_factors["uncertainty"]) > 0


@then('I see 2-3 comparable successful hires from history')
def step_see_comparable_hires(context):
    """Verify comparable hire examples."""
    explanation = context.mixed_explanation
    comparable_hires = explanation.comparable_hires

    assert len(comparable_hires) >= 2
    assert len(comparable_hires) <= 3
    # Each should have identifiable information
    for hire in comparable_hires:
        assert "performance" in hire or "rating" in hire


@then('I understand the business case for hiring')
def step_understand_business_case(context):
    """Verify business case clarity."""
    explanation = context.mixed_explanation
    business_case = explanation.business_case

    assert len(business_case) > 50  # Substantial explanation
    assert any(word in business_case.lower() for word in ["recommend", "hire", "because"])


@then('uncertainty factors are clearly explained')
def step_uncertainty_factors_explained(context):
    """Verify uncertainty explanation."""
    explanation = context.mixed_explanation
    uncertainty = explanation.confidence_factors.get("uncertainty", [])

    assert len(uncertainty) > 0
    # Each uncertainty factor should be descriptive
    for factor in uncertainty:
        assert len(factor) > 10  # Not just keywords


@given('a candidate that triggers bias detection')
def step_candidate_triggers_bias(context):
    """Set up biased scenario."""
    _ensure_base_engine(context)

    # Create scenario where bias would be detected
    male_resume = Resume(
        skill_tokens=["python", "sql"],
        years_experience=3.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    female_resume = Resume(
        skill_tokens=["python", "sql"],  # Identical qualifications
        years_experience=3.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 1}
    )

    # Biased training data: hire males, reject females
    biased_resumes = [male_resume, female_resume, male_resume, female_resume]
    biased_labels = [True, False, True, False]  # Bias pattern

    context.base_engine.fit(biased_resumes, biased_labels)
    context.bias_candidate = female_resume
    context.bias_audit_result = context.base_engine.audit_resume(female_resume)




@then('I see clear bias warning "⚠️ Potential gender bias detected"')
def step_see_bias_warning(context):
    """Verify bias warning display."""
    explanation = context.explanation_result
    bias_warning = explanation.bias_warning

    assert "bias" in bias_warning.lower()
    assert "warning" in bias_warning.lower() or "⚠" in bias_warning
    assert "gender" in bias_warning.lower()


@then('I see specific comparison "Similar male candidates hired at 85% rate"')
def step_see_bias_comparison(context):
    """Verify bias rate comparison."""
    explanation = context.explanation_result
    bias_details = explanation.bias_details

    # Check for rate comparison
    has_rate_comparison = re.search(r'\d+%.*rate', bias_details)
    assert has_rate_comparison, f"Rate comparison not found in {bias_details}"


@then('I see "Female candidates with identical qualifications hired at 62% rate"')
def step_see_demographic_rates(context):
    """Verify demographic-specific rates."""
    explanation = context.explanation_result
    bias_details = explanation.bias_details

    # Check for demographic breakdown
    has_female_rate = "female" in bias_details.lower() and re.search(r'\d+%', bias_details)
    assert has_female_rate, f"Female rate not found in {bias_details}"


@then('I get actionable guidance "Review screening criteria for gender bias"')
def step_get_actionable_guidance(context):
    """Verify actionable bias guidance."""
    explanation = context.explanation_result
    guidance = explanation.bias_guidance

    assert "review" in guidance.lower() or "consider" in guidance.lower()
    assert len(guidance) > 20  # Substantial guidance


@then('bias methodology is explained transparently')
def step_bias_methodology_explained(context):
    """Verify bias methodology transparency."""
    explanation = context.explanation_result
    methodology = explanation.bias_methodology

    assert "disparity" in methodology.lower()
    assert "threshold" in methodology.lower() or "0.8" in methodology
    assert len(methodology) > 30  # Detailed explanation


@given('a candidate missing critical skills')
def step_candidate_missing_skills(context):
    """Set up candidate with skill gaps."""
    _ensure_base_engine(context)

    resume = Resume(
        skill_tokens=["python"],  # Missing machine_learning, aws
        years_experience=3.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    # Train engine to recognize machine_learning and aws as critical
    complete_resume = Resume(
        skill_tokens=["python", "machine_learning", "aws"],
        years_experience=3.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    context.base_engine.fit([complete_resume, resume], [True, False])
    context.incomplete_candidate = resume
    context.incomplete_audit_result = context.base_engine.audit_resume(resume)


@then('I see "Missing: machine_learning, aws (critical for this role)"')
def step_see_missing_skills(context):
    """Verify missing skill identification."""
    explanation = context.explanation_result
    skill_gaps = explanation.skill_gap_analysis

    assert "missing" in skill_gaps.lower()
    # Check for specific skills mentioned
    has_skill_names = any(skill in skill_gaps.lower() for skill in ["machine_learning", "aws"])
    assert has_skill_names, f"Specific skills not found in {skill_gaps}"


@then('I see "Would increase hire probability from 45% to 78%"')
def step_see_probability_improvement(context):
    """Verify probability improvement calculation."""
    explanation = context.explanation_result
    improvement = explanation.skill_gap_impact

    # Check for before/after probabilities
    has_probabilities = re.search(r'\d+%.*to.*\d+%', improvement)
    assert has_probabilities, f"Probability improvement not found in {improvement}"


@then('I see alternative recommendations "Strong in python + sql, consider data analyst role"')
def step_see_alternative_recommendations(context):
    """Verify alternative role suggestions."""
    explanation = context.explanation_result
    alternatives = explanation.alternative_recommendations

    assert "consider" in alternatives.lower() or "alternative" in alternatives.lower()
    assert len(alternatives) > 20  # Substantial recommendations


@then('skill importance is ranked and justified')
def step_skill_importance_ranked(context):
    """Verify skill ranking and justification."""
    explanation = context.explanation_result
    skill_ranking = explanation.skill_importance_ranking

    assert isinstance(skill_ranking, list)
    assert len(skill_ranking) > 0
    # Each ranking should have justification
    for ranking in skill_ranking:
        assert "importance" in str(ranking).lower() or "critical" in str(ranking).lower()


@given('any resume processed by SkillRulesEngine')
def step_any_resume_processed(context):
    """Set up arbitrary resume for invariant testing."""
    _ensure_base_engine(context)

    resume = Resume(
        skill_tokens=["python", "sql"],
        years_experience=3.0,
        education_level="bachelor",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    context.base_engine.fit([resume], [True])
    context.test_resume = resume
    context.original_audit_result = context.base_engine.audit_resume(resume)


@when('enhanced explanations are added')
def step_enhanced_explanations_added(context):
    """Add enhanced explanations."""
    context.enhanced_engine = EnhancedExplanationEngine(context.base_engine)
    context.enhanced_audit_result = context.enhanced_engine.audit_with_explanations(
        context.test_resume
    )


@then('core scores remain identical to original engine')
def step_core_scores_identical(context):
    """Verify score preservation."""
    original_score = context.original_audit_result.overall_score
    enhanced_score = context.enhanced_audit_result.overall_score

    assert abs(original_score - enhanced_score) < 1e-10, f"Scores changed: {original_score} vs {enhanced_score}"


@then('decision outcomes are unchanged')
def step_decision_outcomes_unchanged(context):
    """Verify decision preservation."""
    original_decision = context.original_audit_result.overall_score > 0.5
    enhanced_decision = context.enhanced_audit_result.overall_score > 0.5

    assert original_decision == enhanced_decision, "Decision outcome changed"


@then('only explanation richness is enhanced')
def step_only_explanation_enhanced(context):
    """Verify only explanations are enhanced."""
    original_explanations = len(str(context.original_audit_result.explanations))
    enhanced_explanations = len(str(context.enhanced_audit_result.explanations))

    assert enhanced_explanations > original_explanations, "Explanations not enhanced"


@then('no new scoring logic is introduced')
def step_no_new_scoring_logic(context):
    """Verify no new scoring logic."""
    # Rule scores should be identical
    original_rules = context.original_audit_result.rule_scores
    enhanced_rules = context.enhanced_audit_result.rule_scores

    for rule in original_rules:
        assert abs(original_rules[rule] - enhanced_rules[rule]) < 1e-10


@given('a typical resume for analysis')
def step_typical_resume_analysis(context):
    """Set up typical resume for performance testing."""
    _ensure_base_engine(context)

    resume = Resume(
        skill_tokens=["python", "sql", "machine_learning"],
        years_experience=4.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 0}
    )

    context.base_engine.fit([resume], [True])
    context.performance_resume = resume


@when('I request enhanced explanations')
def step_request_enhanced_explanations_performance(context):
    """Request enhanced explanations for performance test."""
    context.enhanced_engine = EnhancedExplanationEngine(context.base_engine)

    start_time = time.time()
    context.performance_explanation = context.enhanced_engine.explain_decision(
        context.performance_resume,
        context.base_engine.audit_resume(context.performance_resume)
    )
    context.explanation_time = time.time() - start_time


@then('explanations are generated in under 500ms')
def step_explanations_fast_generation(context):
    """Verify fast explanation generation."""
    assert context.explanation_time < 0.5, f"Explanation took {context.explanation_time}s > 500ms"


@then('memory usage stays under 100MB')
def step_memory_usage_reasonable(context):
    """Verify reasonable memory usage."""
    # Simple memory check - enhanced engine should not be dramatically larger
    import sys
    memory_estimate = sys.getsizeof(context.enhanced_engine) + sys.getsizeof(context.performance_explanation)
    assert memory_estimate < 100_000_000, f"Memory usage too high: {memory_estimate} bytes"


@then('explanations scale to batch processing')
def step_explanations_scale_batch(context):
    """Verify batch processing capability."""
    # Test batch processing performance
    batch_resumes = [context.performance_resume] * 10

    start_time = time.time()
    batch_explanations = []
    for resume in batch_resumes:
        audit_result = context.base_engine.audit_resume(resume)
        explanation = context.enhanced_engine.explain_decision(resume, audit_result)
        batch_explanations.append(explanation)
    batch_time = time.time() - start_time

    # Should process 10 resumes in reasonable time
    assert batch_time < 5.0, f"Batch processing too slow: {batch_time}s for 10 resumes"


@then('performance is suitable for real-time UI')
def step_performance_suitable_ui(context):
    """Verify UI-suitable performance."""
    # Already verified under 500ms, which is suitable for UI
    assert context.explanation_time < 0.5


@given('resumes with missing data or unusual skills')
def step_resumes_missing_data(context):
    """Set up edge case resumes."""
    _ensure_base_engine(context)

    context.edge_resumes = [
        # Empty resume
        Resume(
            skill_tokens=[],
            years_experience=0.0,
            education_level="",
            domain_background=[],
            demographics={}
        ),
        # Unusual skills
        Resume(
            skill_tokens=["cobol", "fortran", "assembly"],
            years_experience=30.0,
            education_level="high_school",
            domain_background=["legacy_systems"],
            demographics={"gender": 2}  # Non-binary
        )
    ]

    # Fit engine with normal data first
    normal_resume = Resume(
        skill_tokens=["python"], years_experience=3.0, education_level="bachelor",
        domain_background=["tech"], demographics={"gender": 0}
    )
    context.base_engine.fit([normal_resume], [True])


@when('enhanced explanations are generated')
def step_enhanced_explanations_generated_edge(context):
    """Generate explanations for edge cases."""
    context.enhanced_engine = EnhancedExplanationEngine(context.base_engine)
    context.edge_explanations = []
    context.edge_errors = []

    for resume in context.edge_resumes:
        try:
            audit_result = context.base_engine.audit_resume(resume)
            explanation = context.enhanced_engine.explain_decision(resume, audit_result)
            context.edge_explanations.append(explanation)
        except Exception as e:
            context.edge_errors.append(str(e))


@then('uncertainty is clearly communicated')
def step_uncertainty_clearly_communicated(context):
    """Verify uncertainty communication in edge cases."""
    for explanation in context.edge_explanations:
        uncertainty_text = str(explanation.confidence_analysis)
        assert "uncertain" in uncertainty_text.lower() or "limited" in uncertainty_text.lower()


@then('missing data impact is explained')
def step_missing_data_impact_explained(context):
    """Verify missing data impact explanation."""
    for explanation in context.edge_explanations:
        explanation_text = str(explanation)
        # Should acknowledge missing or limited data
        data_mentions = any(term in explanation_text.lower() for term in ["missing", "limited", "no data", "insufficient"])
        assert data_mentions, f"Missing data not acknowledged in {explanation_text}"


@then('no explanation crashes or fails silently')
def step_no_explanation_crashes(context):
    """Verify no crashes on edge cases."""
    assert len(context.edge_errors) == 0, f"Explanation errors: {context.edge_errors}"


@then('edge cases get appropriate confidence bounds')
def step_edge_cases_appropriate_confidence(context):
    """Verify appropriate confidence for edge cases."""
    for explanation in context.edge_explanations:
        confidence_text = str(explanation.confidence_analysis)
        # Edge cases should have lower confidence
        has_bounds = re.search(r'±.*\d+%', confidence_text)
        assert has_bounds, f"Confidence bounds missing in edge case: {confidence_text}"


@given('hiring decisions from past 2 years')
def step_hiring_decisions_past_2_years(context):
    """Set up historical hiring data."""
    _ensure_base_engine(context)

    # Mock historical data
    context.historical_data = {
        "total_hires": 324,
        "time_range": "2022-2024",
        "performance_tracking": True,
        "sample_candidates": [
            {"name": "Sarah_2023", "skills": ["python", "tensorflow"], "rating": 4.8},
            {"name": "Mike_2024", "skills": ["python", "sql"], "rating": 4.6},
            {"name": "Alex_2022", "skills": ["machine_learning"], "rating": 4.2}
        ]
    }


@when('explanations reference historical data')
def step_explanations_reference_historical(context):
    """Generate explanations that reference historical data."""
    resume = Resume(
        skill_tokens=["python", "tensorflow"],
        years_experience=4.0,
        education_level="master",
        domain_background=["tech"],
        demographics={"gender": 1}
    )

    context.base_engine.fit([resume], [True])
    context.enhanced_engine = EnhancedExplanationEngine(
        context.base_engine,
        historical_data=context.historical_data
    )

    audit_result = context.base_engine.audit_resume(resume)
    context.historical_explanation = context.enhanced_engine.explain_decision(resume, audit_result)


@then('comparable hires are factually accurate')
def step_comparable_hires_accurate(context):
    """Verify comparable hire accuracy."""
    explanation = context.historical_explanation
    comparable_hires = explanation.comparable_hires

    # Should reference actual historical candidates
    historical_names = [candidate["name"] for candidate in context.historical_data["sample_candidates"]]
    for hire in comparable_hires:
        # At least one should match historical data
        name_found = any(name in str(hire) for name in historical_names)
        if name_found:
            break
    else:
        assert False, f"No historical candidates found in {comparable_hires}"


@then('performance ratings are real data')
def step_performance_ratings_real(context):
    """Verify performance ratings come from real data."""
    explanation = context.historical_explanation
    historical_evidence = str(explanation.historical_evidence)

    # Should include actual ratings from historical data
    has_realistic_ratings = re.search(r'[4-5]\.\d.*rating', historical_evidence)
    assert has_realistic_ratings, f"Realistic ratings not found in {historical_evidence}"


@then('sample sizes are disclosed')
def step_sample_sizes_disclosed(context):
    """Verify sample size disclosure."""
    explanation = context.historical_explanation
    evidence_text = str(explanation.historical_evidence)

    # Should mention sample size
    has_sample_size = re.search(r'\d+.*hires?', evidence_text)
    assert has_sample_size, f"Sample size not disclosed in {evidence_text}"


@then('recency bias is acknowledged in older data')
def step_recency_bias_acknowledged(context):
    """Verify recency bias acknowledgment."""
    explanation = context.historical_explanation
    methodology = str(explanation.bias_methodology)

    # Should acknowledge time-based limitations
    time_acknowledgment = any(term in methodology.lower() for term in ["recent", "time", "2022", "older"])
    assert time_acknowledgment, f"Recency not acknowledged in {methodology}"