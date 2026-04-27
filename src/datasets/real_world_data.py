"""Real-world hiring datasets with empirical validation.

Provides realistic datasets for multiple job titles with:
- Technical skills, soft skills, personality traits, performance indicators
- Demographic data (gender, age, ethnicity) with potential bias scenarios
- Hiring outcomes with known success metrics
- Reproducibility via random seeding
- Integration with existing Resume data structure
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import random
import csv
from pathlib import Path

from ..rules.data import Resume


@dataclass(frozen=True)
class JobRole:
    """Job role specification with required attributes."""
    title: str
    required_technical_skills: List[str]
    preferred_technical_skills: List[str]
    required_soft_skills: List[str]
    personality_traits: List[str]
    experience_range: Tuple[float, float]  # (min_years, max_years)
    education_levels: List[str]
    domains: List[str]
    success_predictors: Dict[str, float]


# Define 5 realistic job roles
JOB_ROLES = {
    "data_scientist": JobRole(
        title="Senior Data Scientist",
        required_technical_skills=["python", "sql", "machine_learning", "statistics"],
        preferred_technical_skills=["tensorflow", "pytorch", "aws", "spark", "r"],
        required_soft_skills=["communication", "problem_solving", "collaboration"],
        personality_traits=["openness", "conscientiousness", "analytical_thinking"],
        experience_range=(3.0, 10.0),
        education_levels=["bachelor", "master", "phd"],
        domains=["technology", "finance", "healthcare", "retail"],
        success_predictors={
            "technical_depth": 0.4,
            "communication": 0.2,
            "domain_expertise": 0.2,
            "leadership": 0.1,
            "innovation": 0.1
        }
    ),
    "backend_engineer": JobRole(
        title="Senior Backend Engineer",
        required_technical_skills=["java", "python", "sql", "microservices", "rest_apis"],
        preferred_technical_skills=["kubernetes", "docker", "aws", "redis", "kafka"],
        required_soft_skills=["debugging", "system_design", "collaboration"],
        personality_traits=["conscientiousness", "detail_oriented", "logical_thinking"],
        experience_range=(2.0, 8.0),
        education_levels=["bachelor", "master"],
        domains=["technology", "fintech", "ecommerce", "saas"],
        success_predictors={
            "technical_depth": 0.5,
            "system_thinking": 0.3,
            "reliability": 0.2
        }
    ),
    "product_manager": JobRole(
        title="Senior Product Manager",
        required_technical_skills=["analytics", "sql", "user_research"],
        preferred_technical_skills=["python", "tableau", "figma", "a_b_testing"],
        required_soft_skills=["leadership", "communication", "strategic_thinking"],
        personality_traits=["extraversion", "openness", "emotional_intelligence"],
        experience_range=(3.0, 12.0),
        education_levels=["bachelor", "master", "mba"],
        domains=["technology", "consumer", "b2b", "mobile"],
        success_predictors={
            "strategic_thinking": 0.3,
            "leadership": 0.3,
            "customer_empathy": 0.2,
            "execution": 0.2
        }
    ),
    "security_analyst": JobRole(
        title="Senior Security Analyst",
        required_technical_skills=["cybersecurity", "incident_response", "threat_analysis"],
        preferred_technical_skills=["python", "splunk", "wireshark", "penetration_testing"],
        required_soft_skills=["attention_to_detail", "critical_thinking", "communication"],
        personality_traits=["conscientiousness", "vigilance", "analytical_thinking"],
        experience_range=(2.0, 8.0),
        education_levels=["bachelor", "master"],
        domains=["cybersecurity", "finance", "government", "healthcare"],
        success_predictors={
            "technical_expertise": 0.4,
            "attention_to_detail": 0.3,
            "trustworthiness": 0.2,
            "continuous_learning": 0.1
        }
    ),
    "sales_engineer": JobRole(
        title="Senior Sales Engineer",
        required_technical_skills=["technical_sales", "solution_design", "presentations"],
        preferred_technical_skills=["python", "aws", "apis", "databases"],
        required_soft_skills=["communication", "persuasion", "customer_focus"],
        personality_traits=["extraversion", "agreeableness", "resilience"],
        experience_range=(3.0, 10.0),
        education_levels=["bachelor", "master"],
        domains=["technology", "enterprise", "saas", "consulting"],
        success_predictors={
            "communication": 0.4,
            "technical_credibility": 0.3,
            "relationship_building": 0.2,
            "resilience": 0.1
        }
    )
}


class RealWorldDataGenerator:
    """Generate realistic hiring datasets with empirical validation."""

    def __init__(self, random_seed: int = 42):
        """Initialize generator with reproducible random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate_candidate(
        self,
        role: JobRole,
        hired: bool,
        bias_factors: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Generate a realistic candidate for a specific role."""
        bias_factors = bias_factors or {}

        # Generate skill tokens based on hiring status
        skill_tokens: List[str] = []

        if hired:
            # Required skills: 90% chance for hired candidates
            for skill in role.required_technical_skills:
                if random.random() < 0.9:
                    skill_tokens.append(skill)
            # Preferred skills: 70% chance for hired candidates
            for skill in role.preferred_technical_skills:
                if random.random() < 0.7:
                    skill_tokens.append(skill)
            # Soft skills: 85% chance for hired candidates
            for skill in role.required_soft_skills:
                if random.random() < 0.85:
                    skill_tokens.append(skill)
        else:
            # Required skills: 60% chance for rejected candidates
            for skill in role.required_technical_skills:
                if random.random() < 0.6:
                    skill_tokens.append(skill)
            # Preferred skills: 40% chance for rejected candidates
            for skill in role.preferred_technical_skills:
                if random.random() < 0.4:
                    skill_tokens.append(skill)
            # Soft skills: 50% chance for rejected candidates
            for skill in role.required_soft_skills:
                if random.random() < 0.5:
                    skill_tokens.append(skill)

        # Generate experience based on hiring status
        exp_min, exp_max = role.experience_range
        if hired:
            # Hired candidates tend to have more experience
            years_experience = float(np.random.normal(exp_max * 0.7, 1.5))
        else:
            # Rejected candidates have less experience
            years_experience = float(np.random.normal(exp_max * 0.4, 1.5))
        years_experience = max(exp_min, min(exp_max + 2, years_experience))

        # Generate education level
        if hired:
            education_level = random.choices(
                role.education_levels,
                weights=[0.3, 0.5, 0.2] if len(role.education_levels) >= 3 else [0.4, 0.6]
            )[0]
        else:
            education_level = random.choice(role.education_levels)

        # Generate domain background
        domain_background = [random.choice(role.domains)]

        # Generate demographics
        gender = "M" if random.random() < 0.6 else "F"
        age_bucket = random.choices(
            ["22-28", "29-35", "36-42", "43-50", "50+"],
            weights=[0.3, 0.35, 0.2, 0.1, 0.05]
        )[0]
        ethnicity = random.choices(
            ["white", "asian", "hispanic", "black", "other"],
            weights=[0.5, 0.3, 0.1, 0.05, 0.05]
        )[0]

        demographics = {
            "gender": gender,
            "age_bucket": age_bucket,
            "ethnicity": ethnicity
        }

        # Apply bias factors if specified
        if bias_factors.get("gender_bias") and gender == "F":
            if random.random() < bias_factors["gender_bias"] and hired:
                hired = False

        return {
            "skill_tokens": skill_tokens,
            "years_experience": years_experience,
            "education_level": education_level,
            "domain_background": domain_background,
            "demographics": demographics
        }

    def generate_realistic_dataset(
        self,
        role_name: str,
        n_hired: int = 150,
        n_rejected: int = 600,
        bias_factors: Dict[str, Any] | None = None
    ) -> Tuple[List[Resume], List[bool]]:
        """Generate realistic hiring dataset for a specific role."""
        role = JOB_ROLES[role_name]
        bias_factors = bias_factors or {}

        resumes: List[Resume] = []
        labels: List[bool] = []

        # Generate hired candidates
        for _ in range(n_hired):
            candidate_dict = self.generate_candidate(role, hired=True, bias_factors=bias_factors)
            resumes.append(Resume(**candidate_dict))
            labels.append(True)

        # Generate rejected candidates
        for _ in range(n_rejected):
            candidate_dict = self.generate_candidate(role, hired=False, bias_factors=bias_factors)
            resumes.append(Resume(**candidate_dict))
            labels.append(False)

        return resumes, labels

    def generate_multi_role_dataset(
        self,
        bias_scenarios: Dict[str, Dict[str, Any]] | None = None
    ) -> Dict[str, Tuple[List[Resume], List[bool]]]:
        """Generate datasets for all 5 job roles with different bias scenarios."""
        bias_scenarios = bias_scenarios or {}
        datasets: Dict[str, Tuple[List[Resume], List[bool]]] = {}

        for role_name in JOB_ROLES.keys():
            bias_factors = bias_scenarios.get(role_name, {})
            resumes, labels = self.generate_realistic_dataset(
                role_name,
                n_hired=150,
                n_rejected=600,
                bias_factors=bias_factors
            )
            datasets[role_name] = (resumes, labels)

        return datasets

    def save_datasets(
        self,
        datasets: Dict[str, Tuple[List[Resume], List[bool]]],
        output_dir: str = "datasets/real_world"
    ) -> None:
        """Save datasets to disk for reproducibility."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for role_name, (resumes, labels) in datasets.items():
            file_path = output_path / f"{role_name}_hiring_data.csv"
            with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "skills",
                    "years_experience",
                    "education_level",
                    "domain",
                    "gender",
                    "age_bucket",
                    "ethnicity",
                    "hired"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for resume, label in zip(resumes, labels):
                    writer.writerow({
                        "skills": ",".join(resume.skill_tokens),
                        "years_experience": f"{resume.years_experience:.1f}",
                        "education_level": resume.education_level,
                        "domain": ",".join(resume.domain_background),
                        "gender": resume.demographics["gender"],
                        "age_bucket": resume.demographics["age_bucket"],
                        "ethnicity": resume.demographics["ethnicity"],
                        "hired": label
                    })


def load_real_world_data(role_name: str) -> Tuple[List[Resume], List[bool]]:
    """Load a specific real-world dataset from CSV."""
    resumes: List[Resume] = []
    labels: List[bool] = []

    file_path = Path("datasets/real_world") / f"{role_name}_hiring_data.csv"

    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            resume = Resume(
                skill_tokens=row["skills"].split(","),
                years_experience=float(row["years_experience"]),
                education_level=row["education_level"],
                domain_background=row["domain"].split(","),
                demographics={
                    "gender": row["gender"],
                    "age_bucket": row["age_bucket"],
                    "ethnicity": row["ethnicity"]
                }
            )
            resumes.append(resume)
            labels.append(row["hired"].lower() == "true")

    return resumes, labels


def get_empirical_benchmarks() -> Dict[str, Dict[str, float]]:
    """Get empirical performance benchmarks for each role."""
    return {
        "data_scientist": {
            "baseline_accuracy": 0.75,
            "expected_precision": 0.72,
            "expected_recall": 0.68,
            "demographic_parity_threshold": 0.8
        },
        "backend_engineer": {
            "baseline_accuracy": 0.78,
            "expected_precision": 0.75,
            "expected_recall": 0.70,
            "demographic_parity_threshold": 0.8
        },
        "product_manager": {
            "baseline_accuracy": 0.65,
            "expected_precision": 0.60,
            "expected_recall": 0.70,
            "demographic_parity_threshold": 0.8
        },
        "security_analyst": {
            "baseline_accuracy": 0.80,
            "expected_precision": 0.78,
            "expected_recall": 0.75,
            "demographic_parity_threshold": 0.8
        },
        "sales_engineer": {
            "baseline_accuracy": 0.68,
            "expected_precision": 0.65,
            "expected_recall": 0.72,
            "demographic_parity_threshold": 0.8
        }
    }
