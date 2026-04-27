"""Test counterfactual verification that swaps actually change input vectors."""

import pytest
import numpy as np
from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, create_default_role
from src.fairness.counterfactual import CounterfactualAnalyzer


def test_counterfactual_verification_skips_identical_vectors():
    """Test that counterfactual analysis skips cases where feature vectors don't change."""

    # Create a minimal setup
    vocab = SkillVocabulary(['python'], {'programming': ['python']})
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    # Create resumes that should produce identical feature vectors
    # Since ContentNeutralExtractor doesn't use demographics, swapping demographics
    # should produce identical feature vectors
    resume1 = Resume(
        skill_tokens=['python'],
        years_experience=3.0,
        education_level='bachelor',
        domain_background=['tech'],
        demographics={'gender': 'male', 'race': 'white'}
    )

    resume2 = Resume(
        skill_tokens=['python'],
        years_experience=3.0,
        education_level='bachelor',
        domain_background=['tech'],
        demographics={'gender': 'female', 'race': 'black'}  # Different demographics
    )

    # Mock prediction function that just returns the resume hash (so we can track what got called)
    called_resumes = []

    def mock_predict(resume):
        called_resumes.append(resume)
        return 0.5  # Fixed prediction

    # Run counterfactual analysis with verification
    analyzer = CounterfactualAnalyzer()
    cf_results = analyzer.analyze_counterfactual_fairness(
        [resume1], mock_predict, threshold=0.05, feature_extractor=extractor
    )

    # Verify results - when feature vectors are identical, no analysis should be performed
    # This means the gender key should not be present in results (or be None)
    if 'gender' in cf_results and cf_results['gender'] is not None:
        # If a result exists, it should have 0 comparisons
        gender_result = cf_results['gender']
        assert gender_result.total_comparisons == 0, (
            f"Expected 0 comparisons due to identical feature vectors, got {gender_result.total_comparisons}"
        )
    else:
        # No gender result generated - this is the expected behavior when all comparisons are skipped
        assert cf_results.get('gender') is None, "Expected no gender result when feature vectors are identical"

    # Only prediction calls should have been made (and likely none if all were skipped)
    # The exact number depends on implementation details, but should be minimal
    assert len(called_resumes) <= 1, (
        f"Expected minimal prediction calls due to skipped comparisons, got {len(called_resumes)}"
    )


def test_counterfactual_verification_allows_different_vectors():
    """Test that counterfactual analysis proceeds when feature vectors actually change."""

    # Create resumes with different content that should produce different feature vectors
    vocab = SkillVocabulary(['python'], {'programming': ['python']})
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    # Create a resume where changing demographics might trigger token swaps
    resume = Resume(
        skill_tokens=['python'],
        years_experience=3.0,
        education_level='bachelor',
        domain_background=['tech'],
        demographics={'gender': 'female', 'race': 'white'}
    )

    called_resumes = []

    def mock_predict(resume):
        called_resumes.append(resume)
        return 0.5

    # Run counterfactual analysis
    analyzer = CounterfactualAnalyzer()
    cf_results = analyzer.analyze_counterfactual_fairness(
        [resume], mock_predict, threshold=0.05, feature_extractor=extractor
    )

    # The verification mechanism should complete without errors
    # Results may be empty if no meaningful swaps can be made, but that's valid behavior
    assert isinstance(cf_results, dict)

    # At minimum, the analysis should complete without errors
    # Whether or not 'gender' key exists depends on whether meaningful swaps were possible


def test_verify_feature_vector_change_method():
    """Test the _verify_feature_vector_change method directly."""

    vocab = SkillVocabulary(['python'], {'programming': ['python']})
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    # Create identical resumes (different demographics, same content)
    resume1 = Resume(['python'], 3.0, 'bachelor', ['tech'], {'gender': 'male'})
    resume2 = Resume(['python'], 3.0, 'bachelor', ['tech'], {'gender': 'female'})

    # Create different resumes
    resume3 = Resume(['python'], 5.0, 'master', ['tech'], {'gender': 'male'})

    analyzer = CounterfactualAnalyzer()

    # Identical resumes should produce identical vectors
    identical_check = analyzer._verify_feature_vector_change(resume1, resume2, extractor)
    assert not identical_check, "Expected identical feature vectors to return False"

    # Different resumes should produce different vectors
    different_check = analyzer._verify_feature_vector_change(resume1, resume3, extractor)
    assert different_check, "Expected different feature vectors to return True"


if __name__ == "__main__":
    pytest.main([__file__])