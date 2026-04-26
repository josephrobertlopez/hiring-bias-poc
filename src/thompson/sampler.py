"""
Thompson Sampling for Skill Matching

Exploration/exploitation approach to resume-job matching with uncertainty quantification.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import beta


class SkillThompsonSampler:
    """Thompson sampler for skill-based resume matching"""

    def __init__(self, skills: List[str], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson sampler for skills

        Args:
            skills: List of skill names
            alpha_prior: Beta distribution alpha parameter (successes + 1)
            beta_prior: Beta distribution beta parameter (failures + 1)
        """
        self.skills = skills
        self.skill_to_idx = {skill: i for i, skill in enumerate(skills)}

        # Beta distribution parameters for each skill
        # Start with uniform prior
        self.alphas = np.full(len(skills), alpha_prior)
        self.betas = np.full(len(skills), beta_prior)

        # Track observations
        self.n_observations = 0

    def sample_weights(self) -> np.ndarray:
        """
        Sample skill weights from posterior Beta distributions

        Returns:
            Array of sampled weights for each skill
        """
        weights = np.array([
            beta.rvs(a=alpha, b=beta)
            for alpha, beta in zip(self.alphas, self.betas)
        ])

        # Normalize to sum to 1
        return weights / np.sum(weights)

    def update(self, skill_matches: Dict[str, bool]):
        """
        Update posterior beliefs based on match outcomes

        Args:
            skill_matches: Dict mapping skill names to boolean match outcomes
        """
        for skill, matched in skill_matches.items():
            if skill in self.skill_to_idx:
                idx = self.skill_to_idx[skill]
                if matched:
                    self.alphas[idx] += 1
                else:
                    self.betas[idx] += 1

        self.n_observations += 1

    def get_confidence_intervals(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Get confidence intervals for skill match probabilities

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dict mapping skill names to (lower, upper) confidence bounds
        """
        alpha = 1 - confidence
        intervals = {}

        for i, skill in enumerate(self.skills):
            lower = beta.ppf(alpha/2, a=self.alphas[i], b=self.betas[i])
            upper = beta.ppf(1 - alpha/2, a=self.alphas[i], b=self.betas[i])
            intervals[skill] = (lower, upper)

        return intervals

    def get_skill_scores(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, float]:
        """
        Score resume against job requirements using current posterior

        Args:
            resume_skills: List of skills from resume
            job_skills: List of required skills for job

        Returns:
            Dict with match scores and uncertainty estimates
        """
        # Sample weights from posterior
        weights = self.sample_weights()

        # Compute match score
        resume_skill_set = set(resume_skills)
        job_skill_set = set(job_skills)

        matched_skills = resume_skill_set & job_skill_set
        missing_skills = job_skill_set - resume_skill_set

        # Weighted match score
        match_score = 0.0
        total_weight = 0.0

        for skill in job_skill_set:
            if skill in self.skill_to_idx:
                idx = self.skill_to_idx[skill]
                skill_weight = weights[idx]
                total_weight += skill_weight

                if skill in matched_skills:
                    match_score += skill_weight

        normalized_score = match_score / total_weight if total_weight > 0 else 0.0

        # Compute uncertainty (variance of the score)
        score_variance = self._compute_score_variance(job_skills)

        return {
            'match_score': normalized_score,
            'uncertainty': np.sqrt(score_variance),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'confidence_intervals': self.get_confidence_intervals()
        }

    def _compute_score_variance(self, job_skills: List[str]) -> float:
        """Compute variance in match score due to posterior uncertainty"""
        variances = []

        # Sample multiple times to estimate variance
        n_samples = 100
        scores = []

        for _ in range(n_samples):
            weights = self.sample_weights()
            score = 0.0
            total_weight = 0.0

            for skill in job_skills:
                if skill in self.skill_to_idx:
                    idx = self.skill_to_idx[skill]
                    weight = weights[idx]
                    total_weight += weight
                    # Assume skill is present for variance calculation
                    score += weight

            normalized_score = score / total_weight if total_weight > 0 else 0.0
            scores.append(normalized_score)

        return np.var(scores)

    def get_exploration_priority(self) -> Dict[str, float]:
        """
        Get skills that would benefit most from additional observations

        Returns:
            Dict mapping skill names to exploration priority scores
        """
        priorities = {}

        for i, skill in enumerate(self.skills):
            # Higher variance = higher exploration value
            variance = (self.alphas[i] * self.betas[i]) / \
                      ((self.alphas[i] + self.betas[i])**2 * (self.alphas[i] + self.betas[i] + 1))

            priorities[skill] = variance

        return priorities

    def export_priors(self) -> Dict[str, Dict[str, float]]:
        """Export current posterior parameters for serialization"""
        return {
            skill: {
                'alpha': float(self.alphas[i]),
                'beta': float(self.betas[i]),
                'mean': float(self.alphas[i] / (self.alphas[i] + self.betas[i]))
            }
            for i, skill in enumerate(self.skills)
        }