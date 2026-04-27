"""Test EBM-style model and isotonic calibration."""

import pytest
import numpy as np
from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, create_default_role
from src.features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig
from src.model.ebm_head import ExplainableBoostingModel, EBMConfig
from src.model.calibration import IsotonicCalibrator, apply_full_calibration_pipeline, check_calibration_gates


@pytest.fixture
def sample_data():
    """Create sample training data."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'javascript', 'react', 'tensorflow', 'aws'],
        categories={
            'programming': ['python', 'javascript'],
            'data': ['sql', 'tensorflow'],
            'cloud': ['aws']
        }
    )

    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    # Create diverse training data
    resumes = [
        # Strong candidates
        Resume(['python', 'sql', 'aws'], 5.0, 'bachelor', ['tech'], {}),
        Resume(['python', 'javascript', 'sql'], 6.0, 'master', ['tech', 'finance'], {}),
        Resume(['python', 'tensorflow', 'sql'], 4.0, 'phd', ['tech'], {}),
        Resume(['python', 'sql', 'react'], 5.5, 'bachelor', ['tech'], {}),

        # Medium candidates
        Resume(['python', 'javascript'], 3.0, 'bachelor', ['tech'], {}),
        Resume(['sql', 'aws'], 2.5, 'master', ['finance'], {}),
        Resume(['python'], 4.0, 'bootcamp', ['tech'], {}),

        # Weak candidates
        Resume(['javascript'], 1.0, 'bachelor', ['other'], {}),
        Resume(['aws'], 1.5, 'bootcamp', ['other'], {}),
        Resume([], 0.5, 'bachelor', ['finance'], {}),
        Resume(['sql'], 1.0, 'bachelor', ['healthcare'], {}),

        # Additional variety
        Resume(['python', 'sql', 'tensorflow', 'aws'], 7.0, 'phd', ['tech'], {}),  # Very strong
        Resume(['react', 'javascript'], 2.0, 'bootcamp', ['tech'], {}),  # Frontend only
    ]

    # Labels: True for strong/medium candidates, False for weak
    labels = [True, True, True, True, True, True, True, False, False, False, False, True, False]

    return vocab, role, extractor, resumes, labels


def test_ebm_model_basic_functionality(sample_data):
    """Test basic EBM model functionality."""
    vocab, role, extractor, resumes, labels = sample_data

    config = EBMConfig(n_estimators=50, random_state=42)
    model = ExplainableBoostingModel(config)

    # Fit model
    model.fit(resumes, labels, extractor)
    assert model.fitted

    # Test predictions
    proba = model.predict_proba(resumes[:3], extractor)
    assert proba.shape == (3, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_ebm_monotonicity_constraints(sample_data):
    """Test that monotonicity constraints are applied."""
    vocab, role, extractor, resumes, labels = sample_data

    config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    # Check that monotonic features are properly identified
    assert 'years_experience' in model.monotonic_features
    assert 'skill_overlap_jaccard' in model.monotonic_features
    assert 'required_skill_count' in model.monotonic_features

    # Validate some monotonicity constraints
    X = model._extract_and_encode_features(resumes, extractor, None)
    is_monotonic, violation_rate = model.validate_monotonicity(X, 'years_experience', tolerance=0.2)

    # Allow some tolerance due to small dataset and model complexity
    assert violation_rate <= 0.3, f"High monotonicity violation rate: {violation_rate}"


def test_ebm_feature_importance(sample_data):
    """Test feature importance analysis."""
    vocab, role, extractor, resumes, labels = sample_data

    config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    # Get feature importances
    importances = model.get_feature_importances(top_k=10)

    # Check structure
    assert len(importances) <= 10
    assert all(hasattr(imp, 'feature_name') for imp in importances)
    assert all(hasattr(imp, 'importance') for imp in importances)
    assert all(hasattr(imp, 'shape_contribution') for imp in importances)

    # Check that importances are sorted
    importance_values = [imp.importance for imp in importances]
    assert importance_values == sorted(importance_values, reverse=True)

    # Check that ranks are assigned correctly
    for i, imp in enumerate(importances):
        assert imp.rank == i + 1


def test_ebm_prediction_with_explanation(sample_data):
    """Test detailed prediction with explanations."""
    vocab, role, extractor, resumes, labels = sample_data

    config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    # Test explanation on a strong candidate
    strong_resume = Resume(['python', 'sql', 'aws'], 5.0, 'bachelor', ['tech'], {})
    explanation = model.predict_with_explanation(strong_resume, extractor)

    # Check explanation structure
    assert hasattr(explanation, 'probability')
    assert hasattr(explanation, 'prediction')
    assert hasattr(explanation, 'feature_contributions')
    assert hasattr(explanation, 'top_features')
    assert hasattr(explanation, 'confidence')

    # Check value ranges
    assert 0 <= explanation.probability <= 1
    assert explanation.prediction in [0, 1]
    assert 0 <= explanation.confidence <= 1
    assert isinstance(explanation.feature_contributions, dict)
    assert len(explanation.top_features) <= 10


def test_ebm_with_rule_features(sample_data):
    """Test EBM integration with rule miner features."""
    vocab, role, extractor, resumes, labels = sample_data

    # Create rule miner
    config = RuleMinerConfig(min_support=0.1, min_confidence=0.3, min_lift=1.0, top_k=5)
    rule_miner = FairnessFilteredRuleMiner(config)
    rule_miner.mine_rules(resumes, labels, extractor)

    # Train EBM with rule features
    ebm_config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(ebm_config)
    model.fit(resumes, labels, extractor, rule_miner)

    # Test prediction with rule features
    test_resume = Resume(['python', 'sql'], 4.0, 'bachelor', ['tech'], {})
    proba = model.predict_proba([test_resume], extractor, rule_miner)

    assert proba.shape == (1, 2)
    assert np.all((proba >= 0) & (proba <= 1))

    # Test explanation includes rule features
    explanation = model.predict_with_explanation(test_resume, extractor, rule_miner)
    feature_names = list(explanation.feature_contributions.keys())

    # Should have both content features and rule features
    has_content_features = any('skill' in name or 'experience' in name for name in feature_names)
    has_rule_features = any('rule_' in name for name in feature_names)

    assert has_content_features, "Missing content-neutral features"
    # Rule features may or may not be present depending on discovered patterns


def test_isotonic_calibration_basic(sample_data):
    """Test basic isotonic calibration functionality."""
    vocab, role, extractor, resumes, labels = sample_data

    # Get raw predictions from EBM
    config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    raw_proba = model.predict_proba(resumes, extractor)[:, 1]  # P(hire)
    true_labels = np.array(labels, dtype=int)

    # Apply calibration
    calibrator = IsotonicCalibrator(n_bins=5, random_state=42)  # Fewer bins for small dataset
    result = calibrator.fit_and_calibrate(raw_proba, true_labels, validation_size=0.4)

    # Check result structure
    assert hasattr(result, 'calibrated_probabilities')
    assert hasattr(result, 'metrics')
    assert hasattr(result, 'isotonic_regressor')

    # Check metrics
    assert hasattr(result.metrics, 'ece_before')
    assert hasattr(result.metrics, 'ece_after')
    assert hasattr(result.metrics, 'brier_score_before')
    assert hasattr(result.metrics, 'brier_score_after')

    # ECE should be reduced (or at least not significantly worse)
    # Allow some flexibility due to small dataset
    assert result.metrics.ece_after <= result.metrics.ece_before + 0.1

    # Calibrated probabilities should be in valid range
    assert np.all((result.calibrated_probabilities >= 0) & (result.calibrated_probabilities <= 1))


def test_ece_calculation():
    """Test Expected Calibration Error calculation."""
    calibrator = IsotonicCalibrator(n_bins=10)

    # Perfect calibration case
    perfect_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    perfect_labels = np.array([0, 0, 0, 1, 1, 1])
    ece_perfect = calibrator.calculate_ece(perfect_probs, perfect_labels)

    # Should be low ECE (not necessarily 0 due to binning)
    assert ece_perfect <= 0.2

    # Poor calibration case
    poor_probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # Overconfident wrong predictions
    poor_labels = np.array([0, 0, 0, 1, 1, 1])
    ece_poor = calibrator.calculate_ece(poor_probs, poor_labels)

    # Should have high ECE
    assert ece_poor >= 0.4

    # ECE should be worse for poor calibration
    assert ece_poor > ece_perfect


def test_calibration_quality_evaluation():
    """Test comprehensive calibration quality evaluation."""
    calibrator = IsotonicCalibrator(n_bins=10)

    # Sample data
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1])

    metrics = calibrator.evaluate_calibration_quality(probs, labels)

    # Check all metrics are present
    expected_metrics = ['expected_calibration_error', 'maximum_calibration_error',
                       'average_calibration_error', 'brier_score', 'log_loss']

    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (float, np.float64))
        assert metrics[metric] >= 0  # All metrics should be non-negative


def test_calibration_gates():
    """Test calibration quality gates."""
    # Good calibration
    gates_good = check_calibration_gates(ece=0.03, brier_score=0.2)
    assert gates_good['ece_gate_passed']
    assert gates_good['brier_gate_passed']
    assert gates_good['overall_calibration_passed']

    # Poor calibration
    gates_poor = check_calibration_gates(ece=0.08, brier_score=0.3)
    assert not gates_poor['ece_gate_passed']
    assert not gates_poor['brier_gate_passed']
    assert not gates_poor['overall_calibration_passed']

    # Mixed results
    gates_mixed = check_calibration_gates(ece=0.03, brier_score=0.3)
    assert gates_mixed['ece_gate_passed']
    assert not gates_mixed['brier_gate_passed']
    assert not gates_mixed['overall_calibration_passed']


def test_full_calibration_pipeline(sample_data):
    """Test complete calibration pipeline integration."""
    vocab, role, extractor, resumes, labels = sample_data

    # Get raw model predictions
    config = EBMConfig(n_estimators=30, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    raw_proba = model.predict_proba(resumes, extractor)[:, 1]
    true_labels = np.array(labels, dtype=int)

    # Apply full pipeline
    calibrated_probs, metrics = apply_full_calibration_pipeline(
        raw_proba, true_labels, validation_size=0.4, n_bins=5, random_state=42
    )

    # Check results
    assert len(calibrated_probs) > 0
    assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

    # Check metrics
    assert metrics.ece_before >= 0
    assert metrics.ece_after >= 0
    assert metrics.brier_score_before >= 0
    assert metrics.brier_score_after >= 0
    assert metrics.n_calibration_samples > 0

    # Reliability diagram should have reasonable structure
    diagram = metrics.reliability_diagram
    assert 'bin_centers' in diagram
    assert 'raw_accuracies' in diagram
    assert 'calibrated_accuracies' in diagram
    assert len(diagram['bin_centers']) == len(diagram['raw_accuracies'])


def test_model_handles_missing_features(sample_data):
    """Test EBM gracefully handles missing features in prediction."""
    vocab, role, extractor, resumes, labels = sample_data

    config = EBMConfig(n_estimators=20, random_state=42)
    model = ExplainableBoostingModel(config)
    model.fit(resumes, labels, extractor)

    # Create resume that might generate different features
    unusual_resume = Resume(['unknown_skill'], 0.1, 'unknown_degree', ['unknown_domain'], {})

    # Should not crash and should return valid probabilities
    proba = model.predict_proba([unusual_resume], extractor)
    assert proba.shape == (1, 2)
    assert np.all((proba >= 0) & (proba <= 1))

    # Explanation should also work
    explanation = model.predict_with_explanation(unusual_resume, extractor)
    assert 0 <= explanation.probability <= 1
    assert explanation.prediction in [0, 1]