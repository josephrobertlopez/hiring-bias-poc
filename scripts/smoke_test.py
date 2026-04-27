"""Quick smoke test for Phase 1 implementation."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.resume_processor import ResumeProcessor, load_resume_dataset
from src.rules.data import Resume, SkillVocabulary
from src.rules.engine import SkillRulesEngine
import json


def test_resume_processor():
    """Test ResumeProcessor."""
    print("Test 1: ResumeProcessor")

    processor = ResumeProcessor()

    # Test skill extraction
    sample_resume = """
    Senior Software Engineer with 5 years of experience.
    Proficient in Python, Java, JavaScript, and SQL.
    Expert in AWS and Docker. Experience with Kubernetes and Terraform.
    Master's degree in Computer Science.
    Leadership and team management skills.
    """

    skills = processor.extract_skills(sample_resume)
    assert len(skills) > 0, "Should extract skills"
    assert "python" in [s.lower() for s in skills], "Should find Python"
    print(f"  ✓ Skill extraction: {len(skills)} skills found")

    # Test experience extraction
    experience = processor.extract_experience(sample_resume)
    assert experience == 5.0, "Should extract 5 years"
    print(f"  ✓ Experience extraction: {experience} years")

    # Test education extraction
    education = processor.extract_education(sample_resume)
    assert education == "master", "Should detect master's degree"
    print(f"  ✓ Education extraction: {education}")

    # Test domain extraction
    domain = processor.extract_domain(sample_resume)
    assert "technology" in domain, "Should detect technology domain"
    print(f"  ✓ Domain extraction: {domain}")

    print()


def test_skill_vocabulary():
    """Test SkillVocabulary."""
    print("Test 2: SkillVocabulary")

    processor = ResumeProcessor()
    vocab = processor.vocabulary

    # Check vocabulary size
    assert len(vocab.tokens) > 50, "Should have 50+ skills"
    print(f"  ✓ Vocabulary size: {len(vocab.tokens)} tokens")

    # Check categories
    assert len(vocab.categories) > 0, "Should have categories"
    print(f"  ✓ Categories: {list(vocab.categories.keys())}")

    # Test token_to_index
    if "python" in vocab.tokens:
        idx = vocab.token_to_index("python")
        assert idx >= 0, "Should map to valid index"
        print(f"  ✓ Token mapping: python → {idx}")

    # Test category mask
    mask = vocab.get_category_mask("programming")
    assert mask.sum() > 0, "Should have programming skills"
    print(f"  ✓ Category masks: {mask.sum()} programming skills")

    print()


def test_resume_dataclass():
    """Test Resume dataclass."""
    print("Test 3: Resume Dataclass")

    processor = ResumeProcessor()
    vocab = processor.vocabulary

    resume = Resume(
        skill_tokens=["python", "sql", "aws"],
        years_experience=5.0,
        education_level="master",
        domain_background=["technology", "finance"],
        demographics={"gender": 0}
    )

    # Test skill vector
    vector = resume.get_skill_vector(vocab)
    assert len(vector) == len(vocab.tokens), "Vector should match vocab size"
    assert vector.sum() == 3, "Should have 3 skills"
    print(f"  ✓ Skill vector: {vector.sum()} / {len(vocab.tokens)} skills")

    # Test experience features
    exp_features = resume.get_experience_features()
    assert exp_features["years_experience"] == 5.0
    print(f"  ✓ Experience features: {exp_features}")

    print()


def test_rules_engine():
    """Test SkillRulesEngine."""
    print("Test 4: SkillRulesEngine")

    processor = ResumeProcessor()
    vocab = processor.vocabulary
    engine = SkillRulesEngine(vocab)

    # Create dummy training data
    sample_resumes = [
        Resume(
            skill_tokens=["python", "sql"],
            years_experience=3.0,
            education_level="bachelor",
            domain_background=["technology"],
            demographics={}
        ),
        Resume(
            skill_tokens=["java", "sql"],
            years_experience=5.0,
            education_level="master",
            domain_background=["technology"],
            demographics={}
        ),
        Resume(
            skill_tokens=["python", "java"],
            years_experience=2.0,
            education_level="bootcamp",
            domain_background=["technology"],
            demographics={}
        ),
    ]

    labels = [1, 1, 0]  # Hired vs not hired

    # Test fit
    engine.fit(sample_resumes, labels)
    assert engine.fitted, "Engine should be fitted"
    print(f"  ✓ Engine fit on {len(sample_resumes)} samples")

    # Test audit_resume
    test_resume = Resume(
        skill_tokens=["python", "sql"],
        years_experience=4.0,
        education_level="master",
        domain_background=["technology"],
        demographics={}
    )

    result = engine.audit_resume(test_resume, "test_001")
    assert 0 <= result.overall_score <= 1, "Score should be in [0-1]"
    assert len(result.rule_scores) == 6, "Should have 6 rule scores"
    print(f"  ✓ Audit result:")
    print(f"    - Overall score: {result.overall_score:.3f}")
    print(f"    - Rule scores: {result.rule_scores}")

    # Test batch audit
    batch_results = engine.audit_batch([test_resume, test_resume])
    assert len(batch_results) == 2, "Should process batch"
    print(f"  ✓ Batch audit: {len(batch_results)} results")

    # Test report generation
    report = engine.generate_report(sample_resumes)
    assert report.total_resumes == 3, "Should count resumes"
    assert len(report.skill_frequency) > 0, "Should have skill frequencies"
    print(f"  ✓ Report generation:")
    print(f"    - Total resumes: {report.total_resumes}")
    print(f"    - Unique skills: {len(report.skill_frequency)}")

    print()


def test_dataset_loading():
    """Test dataset loading."""
    print("Test 5: Dataset Loading")

    try:
        resumes, labels, vocab = load_resume_dataset()
        assert len(resumes) > 0, "Should load resumes"
        assert len(labels) == len(resumes), "Should have labels for all resumes"
        assert len(vocab.tokens) > 0, "Should have vocabulary"

        print(f"  ✓ Loaded {len(resumes)} resumes")
        print(f"  ✓ Categories: {len(set(labels))}")
        print(f"  ✓ Vocabulary: {len(vocab.tokens)} tokens")

    except FileNotFoundError as e:
        print(f"  ⚠ Dataset not found (expected if not downloaded): {e}")

    print()


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Phase 1 Smoke Test")
    print("=" * 60)
    print()

    try:
        test_resume_processor()
        test_skill_vocabulary()
        test_resume_dataclass()
        test_rules_engine()
        test_dataset_loading()

        print("=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
