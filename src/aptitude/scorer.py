"""Per-skill aptitude scoring API.

Public API locked - demo and collateral depend on this interface.
Banking MRM compatible with deterministic scoring and quantified uncertainty.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..rules.data import Resume
from ..features.extractors import JobRole
from ..posteriors.rule_reliability import RulePosterior


@dataclass
class SkillAptitude:
    """Per-skill aptitude score with uncertainty and explanation."""
    skill: str                         # e.g. "python"
    score: float                       # posterior mean, [0, 1]
    uncertainty_interval: Tuple[float, float]   # 95% credible interval
    contributing_rules: List['RuleFiring']      # explanation
    fairness_filter_passed: bool       # did the contributing rules clear the proxy filter


@dataclass
class RuleFiring:
    """Rule contribution to skill aptitude score."""
    rule_id: str
    antecedent: str                    # human-readable
    posterior_mean_reliability: float
    posterior_interval: Tuple[float, float]
    contribution_to_skill: float       # signed contribution


@dataclass
class CandidateScoring:
    """Complete candidate evaluation with per-skill breakdown."""
    aptitudes: Dict[str, SkillAptitude]  # per-skill
    overall_recommendation: str          # "advance" | "review" | "do_not_advance"
    overall_uncertainty: Tuple[float, float]
    decision_id: str                     # for audit ledger
    model_version: str
    timestamp: str


def score_candidate(resume: Resume, role: JobRole) -> CandidateScoring:
    """Score candidate with per-skill aptitude breakdown.

    Banking MRM compatible: deterministic scoring with quantified uncertainty
    and full explanation decomposition.

    Args:
        resume: Candidate resume data
        role: Job role requirements

    Returns:
        CandidateScoring with per-skill aptitudes and overall recommendation
    """
    # Generate deterministic decision ID
    resume_hash = _hash_resume(resume)
    role_id = _hash_role(role)
    model_version = "1.0.0"  # TODO: version from git or config
    decision_id = hashlib.md5(f"{resume_hash}:{role_id}:{model_version}".encode()).hexdigest()[:16]

    # Placeholder implementation - will be completed in Item 3
    aptitudes = {}
    for skill in role.required_skills | role.preferred_skills:
        aptitudes[skill] = SkillAptitude(
            skill=skill,
            score=float('nan'),  # TODO: compute from rule posteriors
            uncertainty_interval=(float('nan'), float('nan')),
            contributing_rules=[],
            fairness_filter_passed=True
        )

    return CandidateScoring(
        aptitudes=aptitudes,
        overall_recommendation="review",  # TODO: compute from skill scores
        overall_uncertainty=(0.0, 1.0),  # TODO: propagate from skill uncertainties
        decision_id=decision_id,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


def _hash_resume(resume: Resume) -> str:
    """Deterministic hash of resume content."""
    content = {
        'skill_tokens': sorted(resume.skill_tokens),
        'years_experience': resume.years_experience,
        'education_level': resume.education_level,
        'domain_background': sorted(resume.domain_background)
        # Note: explicitly excluding demographics for fairness
    }
    return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


def _hash_role(role: JobRole) -> str:
    """Deterministic hash of role requirements."""
    content = {
        'required_skills': sorted(role.required_skills),
        'preferred_skills': sorted(role.preferred_skills),
        'min_experience': role.min_experience,
        'max_experience': role.max_experience,
        'role_keywords': sorted(role.role_keywords),
        'seniority_level': role.seniority_level
    }
    return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]