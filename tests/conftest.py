"""Pytest fixtures for hiring bias tests."""
import pytest
from typing import List, Dict, Any
from src.rules.data import Resume, SkillVocabulary
from src.rules.engine import SkillRulesEngine


@pytest.fixture
def basic_vocabulary() -> SkillVocabulary:
    """Basic skill vocabulary for testing."""
    return SkillVocabulary(
        tokens=["python", "sql", "java", "machine_learning", "aws"],
        categories={
            "programming": ["python", "java"],
            "ml": ["machine_learning"],
            "database": ["sql"],
            "infrastructure": ["aws"]
        }
    )


@pytest.fixture
def sample_resumes() -> List[Resume]:
    """Sample resumes for testing with varied skills and demographics."""
    return [
        Resume(
            skill_tokens=["python", "sql"],
            years_experience=3.0,
            education_level="master",
            domain_background=["finance"],
            demographics={"gender": 0}  # Male
        ),
        Resume(
            skill_tokens=["java", "aws"],
            years_experience=2.0,
            education_level="bachelor",
            domain_background=["tech"],
            demographics={"gender": 1}  # Female
        ),
        Resume(
            skill_tokens=["python", "machine_learning"],
            years_experience=5.0,
            education_level="phd",
            domain_background=["healthcare"],
            demographics={"gender": 0}  # Male
        ),
        Resume(
            skill_tokens=["sql"],
            years_experience=1.0,
            education_level="bachelor",
            domain_background=["retail"],
            demographics={"gender": 1}  # Female
        )
    ]


@pytest.fixture
def hired_rejected_labels() -> List[bool]:
    """Corresponding labels for sample_resumes: [hired, rejected, hired, rejected]."""
    return [True, False, True, False]


@pytest.fixture
def fitted_engine(
    basic_vocabulary: SkillVocabulary,
    sample_resumes: List[Resume],
    hired_rejected_labels: List[bool]
) -> SkillRulesEngine:
    """Pre-fitted SkillRulesEngine ready for audit tests."""
    engine = SkillRulesEngine(basic_vocabulary)
    engine.fit(sample_resumes, hired_rejected_labels)
    return engine


@pytest.fixture
def bias_imbalanced_labels() -> List[bool]:
    """Labels with gender bias: all males hired, all females rejected."""
    return [True, False, True, False]


@pytest.fixture
def bias_scenario_resumes() -> List[Resume]:
    """Resumes with intentional gender imbalance for bias testing."""
    # All males hired, all females rejected - clear bias pattern
    return [
        Resume(
            skill_tokens=["python", "sql"],
            years_experience=3.0,
            education_level="master",
            domain_background=["finance"],
            demographics={"gender": 0}  # Male - hired
        ),
        Resume(
            skill_tokens=["python", "sql"],
            years_experience=3.0,
            education_level="master",
            domain_background=["finance"],
            demographics={"gender": 1}  # Female - rejected (same skills)
        ),
        Resume(
            skill_tokens=["java", "aws"],
            years_experience=2.0,
            education_level="bachelor",
            domain_background=["tech"],
            demographics={"gender": 0}  # Male - hired
        ),
        Resume(
            skill_tokens=["java", "aws"],
            years_experience=2.0,
            education_level="bachelor",
            domain_background=["tech"],
            demographics={"gender": 1}  # Female - rejected (same skills)
        )
    ]
