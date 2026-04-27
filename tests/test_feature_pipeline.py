"""Test content-neutral feature extraction and rule mining pipeline."""

import pytest
from src.rules.data import Resume, SkillVocabulary
from src.features.extractors import ContentNeutralExtractor, JobRole, create_default_role
from src.features.rule_miner import FairnessFilteredRuleMiner, RuleMinerConfig


def test_content_neutral_extractor():
    """Test content-neutral feature extraction."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'javascript', 'react', 'tensorflow'],
        categories={'programming': ['python', 'javascript'], 'data': ['sql', 'tensorflow']}
    )

    role = JobRole(
        required_skills={'python', 'sql'},
        preferred_skills={'javascript', 'react'},
        min_experience=2.0,
        max_experience=8.0,
        role_keywords={'software', 'engineer', 'api'},
        seniority_level='mid'
    )

    extractor = ContentNeutralExtractor(vocab, role)

    # Test resume with good skill match
    resume = Resume(
        skill_tokens=['python', 'sql', 'javascript'],
        years_experience=5.0,
        education_level='bachelor',
        domain_background=['tech', 'finance'],
        demographics={}  # Should not be used
    )

    features = extractor.extract_features(resume)

    # Check required feature presence
    assert 'required_skill_count' in features
    assert 'preferred_skill_count' in features
    assert 'skill_overlap_jaccard' in features
    assert 'years_experience_match' in features
    assert 'role_keyword_count' in features
    assert 'seniority_match' in features

    # Check specific values
    assert features['required_skill_count'] == 2  # python, sql
    assert features['preferred_skill_count'] == 1  # javascript
    assert features['required_skill_ratio'] == 1.0  # 2/2
    assert features['experience_in_range'] == 1.0  # 5.0 is in [2.0, 8.0]
    assert features['education_level'] == 'bachelor'
    assert features['domain_count'] == 2


def test_extractor_avoids_protected_attributes():
    """Verify extractor doesn't use protected attributes from demographics."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    # Resume with protected attributes in demographics
    resume = Resume(
        skill_tokens=['python'],
        years_experience=3.0,
        education_level='bachelor',
        domain_background=['tech'],
        demographics={'gender': 'female', 'race': 'asian', 'age': 28, 'zip_code': '90210'}
    )

    features = extractor.extract_features(resume)

    # Ensure no protected attributes appear in features
    protected_keys = {'gender', 'race', 'age', 'zip_code', 'demographics'}
    feature_keys = set(features.keys())

    assert len(feature_keys & protected_keys) == 0, f"Found protected attributes: {feature_keys & protected_keys}"


def test_fairness_filtered_rule_miner():
    """Test rule mining with fairness filtering."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql', 'javascript'],
        categories={'programming': ['python', 'javascript']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    config = RuleMinerConfig(
        min_support=0.1,  # Lower threshold for small test dataset
        min_confidence=0.5,
        min_lift=1.1,
        top_k=10
    )
    miner = FairnessFilteredRuleMiner(config)

    # Create test dataset
    resumes = [
        Resume(['python', 'sql'], 5.0, 'bachelor', ['tech'], {}),
        Resume(['python', 'sql'], 4.0, 'bachelor', ['tech'], {}),
        Resume(['python'], 3.0, 'bachelor', ['finance'], {}),
        Resume(['javascript'], 2.0, 'bootcamp', ['tech'], {}),
        Resume(['sql'], 1.0, 'master', ['finance'], {}),
    ]
    labels = [True, True, False, False, True]

    # Mine rules
    rules = miner.mine_rules(resumes, labels, extractor)

    # Should find some rules
    assert len(rules) > 0

    # Check rule structure
    for rule in rules:
        assert hasattr(rule, 'antecedent')
        assert hasattr(rule, 'consequent')
        assert hasattr(rule, 'support')
        assert hasattr(rule, 'confidence')
        assert hasattr(rule, 'lift')

        # Check metrics are in valid ranges
        assert 0 <= rule.support <= 1
        assert 0 <= rule.confidence <= 1
        assert rule.lift >= config.min_lift


def test_rule_miner_filters_protected_attributes():
    """Verify rule miner filters out protected attributes."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    config = RuleMinerConfig(min_support=0.1, min_confidence=0.3, min_lift=1.0, top_k=50)
    miner = FairnessFilteredRuleMiner(config)

    # Test that protected attributes are identified
    assert miner._is_protected_attribute('gender')
    assert miner._is_protected_attribute('race')
    assert miner._is_protected_attribute('zip_code')
    assert miner._is_protected_attribute('ivy_league')
    assert miner._is_protected_attribute('age_bucket')

    # Test that normal attributes are not filtered
    assert not miner._is_protected_attribute('skill_python')
    assert not miner._is_protected_attribute('exp_mid_level')
    assert not miner._is_protected_attribute('edu_bachelor')


def test_rule_feature_generation():
    """Test binary rule feature generation."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    config = RuleMinerConfig(min_support=0.1, min_confidence=0.3, min_lift=1.0, top_k=5)
    miner = FairnessFilteredRuleMiner(config)

    # Train on small dataset
    train_resumes = [
        Resume(['python', 'sql'], 5.0, 'bachelor', ['tech'], {}),
        Resume(['python'], 3.0, 'bachelor', ['tech'], {}),
        Resume(['sql'], 2.0, 'master', ['finance'], {}),
    ]
    train_labels = [True, False, True]

    rules = miner.mine_rules(train_resumes, train_labels, extractor)

    # Test feature generation on new resume
    test_resume = Resume(['python', 'sql'], 4.0, 'bachelor', ['tech'], {})
    rule_features = miner.get_rule_features(test_resume, extractor)

    # Should have binary features for each rule
    assert all(isinstance(value, int) for value in rule_features.values())
    assert all(value in [0, 1] for value in rule_features.values())

    # Feature count should match number of rules
    assert len(rule_features) == len(rules)


def test_rule_explanations():
    """Test rule explanation generation."""
    vocab = SkillVocabulary(
        tokens=['python', 'sql'],
        categories={'programming': ['python']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    config = RuleMinerConfig(min_support=0.2, min_confidence=0.4, min_lift=1.0, top_k=3)
    miner = FairnessFilteredRuleMiner(config)

    # Train and get explanations
    train_resumes = [
        Resume(['python', 'sql'], 5.0, 'bachelor', ['tech'], {}),
        Resume(['python'], 3.0, 'bachelor', ['tech'], {}),
        Resume(['sql'], 2.0, 'master', ['finance'], {}),
        Resume(['python', 'sql'], 4.0, 'master', ['finance'], {}),
        Resume([], 1.0, 'bootcamp', ['other'], {}),
    ]
    train_labels = [True, False, True, True, False]

    rules = miner.mine_rules(train_resumes, train_labels, extractor)
    explanations = miner.get_rule_explanations()

    # Should have explanations for each rule
    assert len(explanations) == len(rules)

    for explanation in explanations:
        assert 'rule_id' in explanation
        assert 'antecedent' in explanation
        assert 'consequent' in explanation
        assert 'support' in explanation
        assert 'confidence' in explanation
        assert 'lift' in explanation
        assert 'human_readable' in explanation

        # Check explanation structure
        assert isinstance(explanation['antecedent'], list)
        assert isinstance(explanation['consequent'], list)
        assert isinstance(explanation['human_readable'], str)


def test_feature_type_categorization():
    """Test that features are correctly categorized by type."""
    vocab = SkillVocabulary(
        tokens=['python'],
        categories={'programming': ['python']}
    )
    role = create_default_role(vocab)
    extractor = ContentNeutralExtractor(vocab, role)

    categorical_features = extractor.get_categorical_features()
    binary_features = extractor.get_binary_features()
    numeric_features = extractor.get_numeric_features()

    # Check expected features are in right categories
    assert 'experience_bin' in categorical_features
    assert 'education_level' in categorical_features
    assert 'seniority_level' in categorical_features

    assert 'experience_in_range' in binary_features
    assert 'seniority_match' in binary_features

    assert 'years_experience' in numeric_features
    assert 'skill_overlap_jaccard' in numeric_features
    assert 'required_skill_count' in numeric_features

    # No overlap between categories
    all_feature_sets = [set(categorical_features), set(binary_features), set(numeric_features)]
    for i, set1 in enumerate(all_feature_sets):
        for j, set2 in enumerate(all_feature_sets):
            if i != j:
                assert len(set1 & set2) == 0, f"Feature type overlap between categories {i} and {j}"