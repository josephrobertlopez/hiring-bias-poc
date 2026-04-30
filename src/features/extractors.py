"""Content-neutral feature extraction from resumes."""

from typing import Dict, List, Any, Set
import numpy as np
from dataclasses import dataclass

from ..rules.data import Resume, SkillVocabulary


@dataclass
class JobRole:
    """Target job role specification for feature extraction."""
    required_skills: Set[str]
    preferred_skills: Set[str]
    min_experience: float
    max_experience: float
    role_keywords: Set[str]
    seniority_level: str  # "junior", "mid", "senior", "lead"


class ContentNeutralExtractor:
    """Extract content-neutral features from resumes."""

    def __init__(self, vocabulary: SkillVocabulary, target_role: JobRole):
        self.vocabulary = vocabulary
        self.target_role = target_role

        # Define seniority ordering
        self.seniority_order = {
            "intern": 0,
            "junior": 1,
            "mid": 2,
            "senior": 3,
            "lead": 4,
            "principal": 5
        }

    def extract_features(self, resume: Resume) -> Dict[str, Any]:
        """Extract content-neutral features from resume."""
        features = {}

        # Skill overlap features
        resume_skills = set(resume.skill_tokens)
        required_overlap = len(resume_skills & self.target_role.required_skills)
        preferred_overlap = len(resume_skills & self.target_role.preferred_skills)
        total_target_skills = len(self.target_role.required_skills | self.target_role.preferred_skills)

        features['required_skill_count'] = required_overlap
        features['preferred_skill_count'] = preferred_overlap
        features['required_skill_ratio'] = required_overlap / max(len(self.target_role.required_skills), 1)
        features['preferred_skill_ratio'] = preferred_overlap / max(len(self.target_role.preferred_skills), 1)

        # Jaccard similarity
        if total_target_skills > 0:
            union_size = len(resume_skills | self.target_role.required_skills | self.target_role.preferred_skills)
            intersection_size = required_overlap + preferred_overlap
            features['skill_overlap_jaccard'] = intersection_size / union_size
        else:
            features['skill_overlap_jaccard'] = 0.0

        # Experience features
        features['years_experience'] = resume.years_experience
        features['experience_in_range'] = float(
            self.target_role.min_experience <= resume.years_experience <= self.target_role.max_experience
        )
        features['years_experience_match'] = max(0.0, 1.0 - abs(
            resume.years_experience - (self.target_role.min_experience + self.target_role.max_experience) / 2
        ) / 10.0)  # Normalized by 10 years

        # Experience bins for rule mining
        features['experience_bin'] = self._bin_experience(resume.years_experience)

        # Education level (categorical, content-neutral)
        features['education_level'] = resume.education_level
        features['education_numeric'] = self._education_to_numeric(resume.education_level)

        # Domain overlap
        resume_domains = set(resume.domain_background)
        features['domain_count'] = len(resume_domains)

        # Role keyword matching (content-neutral)
        role_keyword_count = 0
        for keyword in self.target_role.role_keywords:
            # Check if keyword appears in skills or domains
            if (keyword.lower() in ' '.join(resume.skill_tokens).lower() or
                keyword.lower() in ' '.join(resume.domain_background).lower()):
                role_keyword_count += 1
        features['role_keyword_count'] = role_keyword_count
        features['role_keyword_ratio'] = role_keyword_count / max(len(self.target_role.role_keywords), 1)

        # Seniority match (based on education + experience heuristic)
        inferred_seniority = self._infer_seniority(resume)
        features['seniority_level'] = inferred_seniority
        features['seniority_match'] = float(inferred_seniority == self.target_role.seniority_level)
        features['seniority_numeric'] = self.seniority_order.get(inferred_seniority, 1)

        # Skill diversity
        features['unique_skill_count'] = len(resume_skills)
        features['skill_diversity'] = len(resume_skills) / max(len(self.vocabulary.tokens), 1)

        # Category coverage
        covered_categories = set()
        for skill in resume_skills:
            for category, category_skills in self.vocabulary.categories.items():
                if skill in category_skills:
                    covered_categories.add(category)
        features['category_coverage'] = len(covered_categories)

        return features

    def _bin_experience(self, years: float) -> str:
        """Bin experience into categories for rule mining."""
        if years < 1:
            return "entry_level"
        elif years < 3:
            return "junior"
        elif years < 7:
            return "mid_level"
        elif years < 10:
            return "senior"
        else:
            return "expert"

    def _education_to_numeric(self, education: str) -> int:
        """Convert education level to numeric for ordering."""
        education_map = {
            "bootcamp": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5
        }
        return education_map.get(education.lower(), 3)  # Default to bachelor level

    def _infer_seniority(self, resume: Resume) -> str:
        """Infer seniority level from education and experience (content-neutral)."""
        years = resume.years_experience
        education_numeric = self._education_to_numeric(resume.education_level)

        # Simple heuristic based on experience + education
        if years < 1:
            return "intern"
        elif years < 2:
            return "junior"
        elif years < 5:
            return "mid"
        elif years < 8:
            return "senior"
        else:
            # Advanced degree + significant experience = lead/principal
            if education_numeric >= 4:  # Master/PhD
                return "lead" if years < 12 else "principal"
            else:
                return "lead"

    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        return [
            'experience_bin',
            'education_level',
            'seniority_level'
        ]

    def get_binary_features(self) -> List[str]:
        """Get list of binary feature names."""
        return [
            'experience_in_range',
            'seniority_match'
        ]

    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'required_skill_count',
            'preferred_skill_count',
            'required_skill_ratio',
            'preferred_skill_ratio',
            'skill_overlap_jaccard',
            'years_experience',
            'years_experience_match',
            'education_numeric',
            'domain_count',
            'role_keyword_count',
            'role_keyword_ratio',
            'seniority_numeric',
            'unique_skill_count',
            'skill_diversity',
            'category_coverage'
        ]


def create_default_role(vocabulary: SkillVocabulary) -> JobRole:
    """Create a default software engineering role for testing/demo."""
    programming_skills = set(vocabulary.categories.get('programming', []))

    return JobRole(
        required_skills=programming_skills & {'python', 'sql'},
        preferred_skills=programming_skills & {'javascript', 'react', 'tensorflow'},
        min_experience=2.0,
        max_experience=8.0,
        role_keywords={'software', 'engineer', 'developer', 'programming', 'backend', 'api'},
        seniority_level='mid'
    )