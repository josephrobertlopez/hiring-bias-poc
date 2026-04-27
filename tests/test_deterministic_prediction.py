"""Test that predictions are deterministic after removing stochastic sampling."""

import pytest
from src.rules.thompson_classifier import ThompsonRulesClassifier
from src.rules.data import Resume, SkillVocabulary


def test_thompson_predictions_are_deterministic():
    """Verify ThompsonRulesClassifier.predict returns identical results for same input."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    classifier = ThompsonRulesClassifier(vocab)

    # Fit on some data
    train_resumes = [Resume(['python'], 2.0, 'bachelor', ['tech'], {})]
    classifier.fit(train_resumes, [True])

    # Test deterministic prediction
    test_resume = Resume(['sql'], 1.0, 'master', ['finance'], {})

    # Multiple calls should return identical results
    pred1 = classifier.predict(test_resume)
    pred2 = classifier.predict(test_resume)

    # All components should be identical
    assert pred1.prediction == pred2.prediction
    assert pred1.confidence == pred2.confidence
    assert pred1.rule_weights == pred2.rule_weights
    assert pred1.exploration_bonus == pred2.exploration_bonus
    assert pred1.regret_bound == pred2.regret_bound

    # Audit results should also be identical
    assert pred1.audit_result.overall_score == pred2.audit_result.overall_score
    assert pred1.audit_result.rule_scores == pred2.audit_result.rule_scores


def test_thompson_batch_predictions_deterministic():
    """Verify batch predictions are deterministic."""
    vocab = SkillVocabulary(
        tokens=['python'],
        categories={'programming': ['python']}
    )
    classifier = ThompsonRulesClassifier(vocab)
    classifier.fit([Resume(['python'], 3.0, 'bachelor', ['tech'], {})], [True])

    test_resumes = [
        Resume(['python'], 2.0, 'master', ['finance'], {}),
        Resume([], 1.0, 'bachelor', ['tech'], {})
    ]

    # Multiple batch calls should return identical results
    batch1 = classifier.predict_batch(test_resumes)
    batch2 = classifier.predict_batch(test_resumes)

    assert len(batch1) == len(batch2)
    for pred1, pred2 in zip(batch1, batch2):
        assert pred1.prediction == pred2.prediction
        assert pred1.confidence == pred2.confidence
        assert pred1.rule_weights == pred2.rule_weights


def test_no_random_sampling_in_rule_weights():
    """Verify rule weights are based on posterior means, not random samples."""
    vocab = SkillVocabulary(
        tokens=['python'],
        categories={'programming': ['python']}
    )
    classifier = ThompsonRulesClassifier(vocab)
    classifier.fit([Resume(['python'], 3.0, 'bachelor', ['tech'], {})], [True])

    test_resume = Resume(['python'], 2.0, 'bachelor', ['tech'], {})
    pred = classifier.predict(test_resume)

    # Rule weights should be deterministic posterior means
    # Verify they are in [0, 1] range (valid Beta posterior means)
    for rule_name, weight in pred.rule_weights.items():
        assert 0.0 <= weight <= 1.0, f"Rule weight {rule_name}={weight} out of range"

    # Verify weight calculation matches posterior mean formula
    for i, rule_name in enumerate(classifier.rule_names):
        alpha, beta = classifier.thompson_sampler.get_posterior_params(i)
        expected_mean = alpha / (alpha + beta)
        actual_weight = pred.rule_weights[rule_name]
        assert abs(actual_weight - expected_mean) < 1e-10, f"Weight mismatch for {rule_name}"