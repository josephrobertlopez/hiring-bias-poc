#!/usr/bin/env python3
"""Smoke test for demo application.

Runs each screen's compute path headlessly to verify no exceptions.
This ensures the demo will work reliably during presentations.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.demo.components.data_loaders import (
    load_sample_data,
    get_demo_model_components,
    create_prediction_function,
    get_model_features,
    get_real_audit_decisions,
)
from src.demo.components.pdf_renderer import (
    generate_audit_pdf,
    inject_biased_decision,
    process_reviewer_action,
)
from src.features.extractors import ContentNeutralExtractor
from src.features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from src.posteriors.rule_reliability import fit_rule_posteriors
from src.aptitude.scorer import score_candidate
from src.fairness.counterfactual import CounterfactualAnalyzer


def test_data_loading():
    """Test sample data loading."""
    print("Testing data loading...")

    resumes, roles = load_sample_data()
    assert len(resumes) == 8, f"Expected 8 resumes, got {len(resumes)}"
    assert len(roles) == 2, f"Expected 2 roles, got {len(roles)}"

    # Verify data structure
    first_resume_id = list(resumes.keys())[0]
    first_role_id = list(roles.keys())[0]

    assert 'name' in resumes[first_resume_id], "Resume missing name field"
    assert 'resume' in resumes[first_resume_id], "Resume missing resume object"
    assert 'title' in roles[first_role_id], "Role missing title field"
    assert 'role' in roles[first_role_id], "Role missing role object"

    print("✅ Data loading test passed")


def test_model_components():
    """Test model component initialization."""
    print("Testing model components...")

    extractor, miner, posteriors = get_demo_model_components()
    assert len(extractor.vocabulary.tokens) > 0, "Vocabulary is empty"
    assert hasattr(miner, 'rules'), "Miner should have rules attribute"
    assert len(posteriors) > 0, f"Expected posteriors, got {len(posteriors)}"

    print("✅ Model components test passed")


def test_candidate_assessment():
    """Test candidate assessment (Screen 1: Candidate View)."""
    print("Testing candidate assessment...")

    # Load data
    resumes, roles = load_sample_data()
    base_extractor, miner, posteriors = get_demo_model_components()

    # Get test candidate and role
    first_resume_id = list(resumes.keys())[0]
    first_role_id = list(roles.keys())[0]
    resume = resumes[first_resume_id]['resume']
    role = roles[first_role_id]['role']

    # Set up pipeline
    extractor = ContentNeutralExtractor(base_extractor.vocabulary, role)

    # Get training data for role-specific posterior fitting
    train_resumes = [data["resume"] for data in resumes.values()]
    train_labels = [True, True, False, True, False, False, False, True]  # Mock labels for demo

    # Use shared miner but fit role-specific posteriors
    rule_posteriors = fit_rule_posteriors(
        miner.rules,
        train_resumes,
        train_labels,
        extractor,
        n_folds=3
    )

    # Test scoring
    scoring = score_candidate(
        resume=resume,
        role=role,
        rules=miner.rules,
        rule_posteriors=rule_posteriors,
        extractor=extractor
    )

    assert hasattr(scoring, 'decision_id'), "Scoring missing decision_id"
    assert hasattr(scoring, 'overall_recommendation'), "Scoring missing overall_recommendation"
    assert hasattr(scoring, 'aptitudes'), "Scoring missing aptitudes"
    assert scoring.overall_recommendation in ['advance', 'review', 'do_not_advance'], \
        f"Invalid recommendation: {scoring.overall_recommendation}"

    print("✅ Candidate assessment test passed")


def test_counterfactual_analysis():
    """Test counterfactual analysis (Screen 2: Counterfactual Matrix)."""
    print("Testing counterfactual analysis...")

    # Load data
    resumes, roles = load_sample_data()

    # Get test candidate and role
    first_resume_id = list(resumes.keys())[0]
    first_role_id = list(roles.keys())[0]
    resume = resumes[first_resume_id]['resume']
    role = roles[first_role_id]['role']

    # Create prediction function
    predict_fn, extractor = create_prediction_function(role)

    # Test feature extraction
    features = get_model_features(resume, role, extractor)
    assert len(features) > 0, "No features extracted"

    # Test counterfactual analyzer
    analyzer = CounterfactualAnalyzer()
    cf_results = analyzer.analyze_counterfactual_fairness(
        [resume],
        predict_fn,
        threshold=0.001,
        feature_extractor=extractor
    )

    assert 'gender' in cf_results, "Missing gender counterfactual results"
    assert 'race' in cf_results, "Missing race counterfactual results"
    assert 'ethnicity' in cf_results, "Missing ethnicity counterfactual results"

    # Test detailed comparisons
    detailed = analyzer.get_detailed_comparisons(
        [resume], 'gender', predict_fn, top_k=5, feature_extractor=extractor
    )
    # Note: detailed may be empty if no swaps produce different feature vectors

    print("✅ Counterfactual analysis test passed")


def test_governance_dashboard():
    """Test governance dashboard (Screen 3: Governance Dashboard)."""
    print("Testing governance dashboard...")

    # Test mock audit decisions
    decisions = get_real_audit_decisions()
    assert len(decisions) == 16, f"Expected 16 real decisions (8 resumes × 2 roles), got {len(decisions)}"

    # Test bias injection
    biased_decision = inject_biased_decision()
    assert biased_decision['bias_flagged'] is True, "Biased decision not flagged"
    assert biased_decision.get('simulation') is True, "Biased decision not marked as simulation"

    # Test reviewer action processing
    result = process_reviewer_action('test_id', 'Reject', 'Bias detected in testing')
    assert result['reviewer_action'] == 'Reject', "Reviewer action not processed correctly"
    assert result['reviewer_comment'] == 'Bias detected in testing', "Reviewer comment not stored"

    print("✅ Governance dashboard test passed")


def test_audit_report_generation():
    """Test audit report generation (Screen 4: Generate Report)."""
    print("Testing audit report generation...")

    progress_messages = []

    def test_callback(message):
        progress_messages.append(message)

    # Test PDF generation for each scope
    scopes = ["Single Decision", "Last Week", "Last Month", "All Decisions"]

    for scope in scopes:
        print(f"  Testing scope: {scope}")

        pdf_buffer = generate_audit_pdf(scope, test_callback)

        assert pdf_buffer.getvalue(), f"Empty PDF generated for scope: {scope}"
        assert len(pdf_buffer.getvalue()) > 1000, f"PDF too small for scope: {scope}"

    # Verify progress messages were logged
    assert len(progress_messages) > 0, "No progress messages logged"
    assert any("Loading decisions" in msg for msg in progress_messages), "Missing loading progress"
    assert any("PDF generated successfully" in msg for msg in progress_messages), "Missing completion progress"

    print("✅ Audit report generation test passed")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("Testing end-to-end workflow...")

    # This simulates a complete demo run

    # 1. Load data
    resumes, roles = load_sample_data()

    # 2. Score a candidate
    first_resume_id = list(resumes.keys())[0]
    first_role_id = list(roles.keys())[0]
    resume = resumes[first_resume_id]['resume']
    role = roles[first_role_id]['role']

    base_extractor, miner, posteriors = get_demo_model_components()
    extractor = ContentNeutralExtractor(base_extractor.vocabulary, role)

    # Get training data for role-specific posterior fitting
    train_resumes = [data["resume"] for data in resumes.values()]
    train_labels = [True, True, False, True, False, False, False, True]  # Mock labels for demo

    # Use shared miner but fit role-specific posteriors
    rule_posteriors = fit_rule_posteriors(miner.rules, train_resumes, train_labels, extractor, n_folds=3)

    scoring = score_candidate(
        resume=resume,
        role=role,
        rules=miner.rules,
        rule_posteriors=rule_posteriors,
        extractor=extractor
    )

    # 3. Run counterfactual analysis
    predict_fn, _ = create_prediction_function(role)
    analyzer = CounterfactualAnalyzer()
    cf_results = analyzer.analyze_counterfactual_fairness([resume], predict_fn)

    # 4. Test governance workflow
    decisions = get_real_audit_decisions()
    biased_decision = inject_biased_decision()

    # 5. Generate audit report
    pdf_buffer = generate_audit_pdf("All Decisions")

    print("✅ End-to-end workflow test passed")


def run_smoke_tests():
    """Run all smoke tests."""
    print("🔥 Running demo smoke tests...")
    print("="*50)

    tests = [
        test_data_loading,
        test_model_components,
        test_candidate_assessment,
        test_counterfactual_analysis,
        test_governance_dashboard,
        test_audit_report_generation,
        test_end_to_end_workflow
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
            print()

    print("="*50)
    print(f"Smoke test results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All smoke tests passed! Demo is ready.")
        return True
    else:
        print("💥 Some smoke tests failed. Fix issues before demo.")
        return False


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)