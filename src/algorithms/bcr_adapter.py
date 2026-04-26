from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.stats import beta

@dataclass
class ResumeInfo:
    resume_id: str
    skills: List[str]
    experience_years: int
    education_level: str
    protected_attributes: Dict[str, Any]

@dataclass
class SkillBelief:
    skill_name: str
    alpha: float  # Beta distribution parameter
    beta: float   # Beta distribution parameter
    observations: int

class BayesianControlRule:
    def __init__(self, skills: List[str]):
        self.skills = skills
        self.skill_beliefs = {skill: SkillBelief(skill, 1.0, 1.0, 0) for skill in skills}

    def update_belief(self, resume_info: ResumeInfo, outcome: bool):
        for skill in resume_info.skills:
            if skill in self.skill_beliefs:
                belief = self.skill_beliefs[skill]
                if outcome:
                    belief.alpha += 1
                else:
                    belief.beta += 1
                belief.observations += 1

    def get_observation_ranking(self, resumes: List[ResumeInfo]) -> List[Tuple[str, float]]:
        rankings = []
        for resume in resumes:
            for skill in resume.skills:
                if skill in self.skill_beliefs:
                    belief = self.skill_beliefs[skill]
                    variance = (belief.alpha * belief.beta) / ((belief.alpha + belief.beta)**2 * (belief.alpha + belief.beta + 1))
                    rankings.append((resume.resume_id, variance))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_intervention_ranking(self, resumes: List[ResumeInfo]) -> List[Tuple[str, float]]:
        rankings = []
        for resume in resumes:
            utility = 0.0
            for skill in resume.skills:
                if skill in self.skill_beliefs:
                    belief = self.skill_beliefs[skill]
                    mean = belief.alpha / (belief.alpha + belief.beta)
                    # Assuming skill_improvement and complexity_cost are predefined constants
                    skill_improvement = 1.0  # Example value
                    complexity_cost = 0.1    # Example value
                    utility += mean * skill_improvement - complexity_cost
            rankings.append((resume.resume_id, utility))

        if not rankings:
            return []

        max_utility = max(rankings, key=lambda x: x[1])[1]
        min_utility = min(rankings, key=lambda x: x[1])[1]

        # Normalize utilities to [0, 1]
        if max_utility == min_utility:
            # All utilities are equal - return 0.5 for all
            normalized_rankings = [(resume_id, 0.5) for resume_id, _ in rankings]
        else:
            normalized_rankings = [(resume_id, (utility - min_utility) / (max_utility - min_utility)) for resume_id, utility in rankings]

        return sorted(normalized_rankings, key=lambda x: x[1], reverse=True)

    def sample_utility(self, resume_info: ResumeInfo) -> float:
        samples = []
        for skill in resume_info.skills:
            if skill in self.skill_beliefs:
                belief = self.skill_beliefs[skill]
                mean = belief.alpha / (belief.alpha + belief.beta)
                variance = (belief.alpha * belief.beta) / ((belief.alpha + belief.beta)**2 * (belief.alpha + belief.beta + 1))
                sample = np.clip(np.random.normal(mean, np.sqrt(variance)), 0.0, 1.0)
                samples.append(sample)
        return np.mean(samples) if samples else 0.5
