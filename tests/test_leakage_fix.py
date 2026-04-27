"""Test that train-test leakage is eliminated from SkillRulesEngine."""

import pytest
from src.rules.engine import SkillRulesEngine
from src.rules.data import Resume, SkillVocabulary


def test_no_training_data_storage():
    """Verify engine doesn't store training data after fit() - core leakage fix."""
    # Setup
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'java'],
        categories={'programming': ['python', 'java']}
    )
    engine = SkillRulesEngine(vocab)

    resumes = [
        Resume(['python'], 3.0, 'bachelor', ['tech'], {}),
        Resume(['java'], 2.0, 'master', ['finance'], {})
    ]
    labels = [True, False]

    # Act
    engine.fit(resumes, labels)

    # Assert - core leakage fix verification
    assert not hasattr(engine, '_training_labels'), "Engine should not store training labels"
    assert not hasattr(engine, '_training_resumes'), "Engine should not store training resumes"
    assert engine.fitted, "Engine should still be marked as fitted"


def test_inference_works_without_training_data():
    """Verify audit_resume works with only fitted parameters."""
    # Setup
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    engine = SkillRulesEngine(vocab)

    # Fit on some data
    train_resumes = [Resume(['python'], 2.0, 'bachelor', ['tech'], {})]
    engine.fit(train_resumes, [True])

    # Test inference on new data
    test_resume = Resume(['sql'], 1.0, 'master', ['finance'], {})

    # Should work without accessing training data
    result = engine.audit_resume(test_resume)

    assert result is not None
    assert hasattr(result, 'overall_score')
    assert 0 <= result.overall_score <= 1


def test_deterministic_inference():
    """Verify inference is deterministic - no randomness."""
    vocab = SkillVocabulary(
        tokens=['python'],
        categories={'programming': ['python']}
    )
    engine = SkillRulesEngine(vocab)
    engine.fit([Resume(['python'], 3.0, 'bachelor', ['tech'], {})], [True])

    test_resume = Resume(['python'], 2.0, 'master', ['finance'], {})

    # Multiple calls should return identical results
    result1 = engine.audit_resume(test_resume)
    result2 = engine.audit_resume(test_resume)

    assert result1.overall_score == result2.overall_score
    assert result1.rule_scores == result2.rule_scores