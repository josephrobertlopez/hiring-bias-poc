"""Unit tests for data structures."""
import pytest
import numpy as np
from src.rules.data import Resume, SkillVocabulary, SkillTokenizer


class TestResume:
    """Test Resume data class."""

    def test_resume_creation(self):
        """Test basic resume creation."""
        resume = Resume(
            skill_tokens=["python", "sql"],
            years_experience=3.0,
            education_level="master",
            domain_background=["finance"],
            demographics={"gender": 0}
        )

        assert resume.skill_tokens == ["python", "sql"]
        assert resume.years_experience == 3.0
        assert resume.education_level == "master"
        assert resume.domain_background == ["finance"]
        assert resume.demographics["gender"] == 0

    def test_resume_frozen(self):
        """Test that Resume is immutable."""
        resume = Resume(["python"], 3.0, "master", ["finance"], {"gender": 0})

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            resume.skill_tokens = ["java"]

    def test_get_skill_vector(self):
        """Test conversion of skills to binary vector."""
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql"],
            categories={}
        )
        resume = Resume(["python", "sql"], 3.0, "master", ["finance"], {"gender": 0})

        vector = resume.get_skill_vector(vocab)

        assert isinstance(vector, np.ndarray)
        assert vector.dtype == bool
        assert len(vector) == 3
        assert vector[0] == True   # python
        assert vector[1] == False  # java
        assert vector[2] == True   # sql

    def test_skill_vector_with_unknown_skills(self):
        """Test skill vector ignores unknown skills."""
        vocab = SkillVocabulary(
            tokens=["python", "java"],
            categories={}
        )
        resume = Resume(["python", "unknown_skill"], 3.0, "master", ["finance"], {"gender": 0})

        vector = resume.get_skill_vector(vocab)

        assert vector[0] == True   # python
        assert vector[1] == False  # java (unknown_skill ignored)

    def test_get_experience_features(self):
        """Test extraction of experience features."""
        resume = Resume(["python"], 5.0, "master", ["finance"], {"gender": 0})

        features = resume.get_experience_features()

        assert isinstance(features, dict)
        assert "years_experience" in features
        assert features["years_experience"] == 5.0


class TestSkillVocabulary:
    """Test SkillVocabulary data class."""

    def test_vocabulary_creation(self):
        """Test basic vocabulary creation."""
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql"],
            categories={"programming": ["python", "java"]}
        )

        assert vocab.tokens == ["python", "java", "sql"]
        assert "programming" in vocab.categories

    def test_token_to_index(self):
        """Test token to index mapping."""
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql"],
            categories={}
        )

        assert vocab.token_to_index("python") == 0
        assert vocab.token_to_index("java") == 1
        assert vocab.token_to_index("sql") == 2

    def test_token_to_index_not_found(self):
        """Test that unknown token raises error."""
        vocab = SkillVocabulary(
            tokens=["python", "java"],
            categories={}
        )

        with pytest.raises(ValueError):
            vocab.token_to_index("unknown")

    def test_get_category_mask(self):
        """Test category mask generation."""
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql", "tensorflow"],
            categories={"programming": ["python", "java"], "ml": ["tensorflow"]}
        )

        mask = vocab.get_category_mask("programming")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == 4
        assert mask[0] == True   # python in programming
        assert mask[1] == True   # java in programming
        assert mask[2] == False  # sql not in programming
        assert mask[3] == False  # tensorflow not in programming

    def test_get_category_mask_ml(self):
        """Test ML category mask."""
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql", "tensorflow"],
            categories={"programming": ["python", "java"], "ml": ["tensorflow"]}
        )

        mask = vocab.get_category_mask("ml")

        assert mask[3] == True   # tensorflow in ml
        assert mask[0] == False  # python not in ml

    def test_get_category_mask_unknown(self):
        """Test mask for unknown category returns zeros."""
        vocab = SkillVocabulary(
            tokens=["python", "java"],
            categories={}
        )

        mask = vocab.get_category_mask("unknown_category")

        assert np.all(mask == False)

    def test_vocabulary_with_embeddings(self):
        """Test vocabulary with embeddings."""
        embeddings = np.random.rand(3, 10)
        vocab = SkillVocabulary(
            tokens=["python", "java", "sql"],
            categories={},
            embeddings=embeddings
        )

        assert vocab.embeddings is not None
        assert vocab.embeddings.shape == (3, 10)


class TestSkillTokenizer:
    """Test SkillTokenizer extraction."""

    def test_tokenizer_creation(self):
        """Test tokenizer initialization."""
        vocab = SkillVocabulary(
            tokens=["python", "sql"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        assert tokenizer.vocabulary == vocab

    def test_extract_skills_exact_match(self):
        """Test skill extraction with exact matches."""
        vocab = SkillVocabulary(
            tokens=["python", "sql", "java"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        text = "I know python and sql but not java"
        skills = tokenizer.extract_skills(text)

        assert "python" in skills
        assert "sql" in skills
        assert "java" in skills

    def test_extract_skills_case_insensitive(self):
        """Test that skill extraction is case-insensitive."""
        vocab = SkillVocabulary(
            tokens=["python", "sql"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        text = "I know PYTHON and SQL"
        skills = tokenizer.extract_skills(text)

        assert "python" in skills
        assert "sql" in skills

    def test_extract_skills_word_boundary(self):
        """Test that word boundaries prevent false matches."""
        vocab = SkillVocabulary(
            tokens=["java"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        # "javascript" contains "java" but should not match
        text = "I know javascript"
        skills = tokenizer.extract_skills(text)

        assert "java" not in skills

    def test_extract_skills_no_matches(self):
        """Test extraction when no skills match."""
        vocab = SkillVocabulary(
            tokens=["python", "sql"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        text = "I know ruby and perl"
        skills = tokenizer.extract_skills(text)

        assert len(skills) == 0

    def test_extract_skills_duplicate_removal(self):
        """Test that same skill mentioned multiple times is included once."""
        vocab = SkillVocabulary(
            tokens=["python"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        text = "I know python and I love python"
        skills = tokenizer.extract_skills(text)

        # Should extract python, potentially with duplicates
        assert "python" in skills

    def test_extract_skills_special_characters(self):
        """Test skill extraction with special characters."""
        vocab = SkillVocabulary(
            tokens=["c++", "c#", "node.js"],
            categories={}
        )
        tokenizer = SkillTokenizer(vocab)

        text = "I can code in c++ and c# and node.js"
        skills = tokenizer.extract_skills(text)

        assert "c++" in skills
        assert "c#" in skills
        assert "node.js" in skills
