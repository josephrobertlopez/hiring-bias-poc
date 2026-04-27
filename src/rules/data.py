from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

import numpy as np


@dataclass(frozen=True)
class Resume:
    """Resume with structured skill tokens and demographic data."""
    skill_tokens: List[str]  # ["python", "machine_learning", "sklearn"]
    years_experience: float
    education_level: str  # "bachelor", "master", "phd", "bootcamp"
    domain_background: List[str]  # ["finance", "healthcare"]
    demographics: Dict[str, Any]  # {"gender": 0, "age_bucket": "25-35"}

    def get_skill_vector(self, vocabulary: 'SkillVocabulary') -> np.ndarray:
        """Convert skill tokens to binary vector."""
        vector = np.zeros(len(vocabulary.tokens), dtype=bool)
        for token in self.skill_tokens:
            if token in vocabulary.tokens:
                index = vocabulary.token_to_index(token)
                vector[index] = True
        return vector

    def get_experience_features(self) -> Dict[str, float]:
        """Extract experience-based features."""
        # Implementation stub
        return {
            'years_experience': self.years_experience,
            # Add more experience-based features here
        }


@dataclass(frozen=True)
class SkillVocabulary:
    """Centralized skill vocabulary with embeddings."""
    tokens: List[str]  # ["python", "sql", "tensorflow", ...]
    categories: Dict[str, List[str]]  # {"programming": ["python", "java"]}
    embeddings: Optional[np.ndarray] = None

    def token_to_index(self, token: str) -> int:
        """Map skill token to vocabulary index."""
        return self.tokens.index(token)

    def get_category_mask(self, category: str) -> np.ndarray:
        """Binary mask for tokens in category."""
        mask = np.zeros(len(self.tokens), dtype=bool)
        for token in self.categories.get(category, []):
            if token in self.tokens:
                index = self.token_to_index(token)
                mask[index] = True
        return mask


class SkillTokenizer:
    """Extract skill tokens from raw resume text."""

    def __init__(self, vocabulary: SkillVocabulary):
        self.vocabulary = vocabulary

    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract skill tokens from resume text.

        Uses word boundaries to avoid false positives (e.g., "java" won't match "javascript").
        """
        found_skills = []
        text_lower = resume_text.lower()

        for token in self.vocabulary.tokens:
            # Use word boundaries to avoid substring matches
            pattern = r'\b' + re.escape(token.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(token)

        return found_skills
