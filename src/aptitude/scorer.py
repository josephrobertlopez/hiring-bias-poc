"""Per-skill aptitude scoring API.

Public API locked - demo and collateral depend on this interface.
Banking MRM compatible with deterministic scoring and quantified uncertainty.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import numpy as np
from scipy.stats import beta

from ..rules.data import Resume
from ..features.extractors import JobRole
from ..posteriors.rule_reliability import RulePosterior, fit_rule_posteriors
from ..features.rule_miner import AssociationRule


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


def score_candidate(
    resume: Resume,
    role: JobRole,
    rules: List[AssociationRule] = None,
    rule_posteriors: Dict[str, RulePosterior] = None,
    extractor = None
) -> CandidateScoring:
    """Score candidate with per-skill aptitude breakdown.

    Banking MRM compatible: deterministic scoring with quantified uncertainty
    and full explanation decomposition.

    Args:
        resume: Candidate resume data
        role: Job role requirements
        rules: Association rules (optional, for testing)
        rule_posteriors: Fitted rule posteriors (optional, for testing)
        extractor: Feature extractor (optional, for testing)

    Returns:
        CandidateScoring with per-skill aptitudes and overall recommendation
    """
    # Generate deterministic decision ID
    resume_hash = _hash_resume(resume)
    role_id = _hash_role(role)
    model_version = "1.0.0"  # TODO: version from git or config
    decision_id = hashlib.md5(f"{resume_hash}:{role_id}:{model_version}".encode()).hexdigest()[:16]

    # Early return with NaN scores if no rules/posteriors provided
    if not rules or not rule_posteriors or not extractor:
        aptitudes = {}
        for skill in role.required_skills | role.preferred_skills:
            aptitudes[skill] = SkillAptitude(
                skill=skill,
                score=float('nan'),
                uncertainty_interval=(float('nan'), float('nan')),
                contributing_rules=[],
                fairness_filter_passed=True
            )

        return CandidateScoring(
            aptitudes=aptitudes,
            overall_recommendation="review",
            overall_uncertainty=(0.0, 1.0),
            decision_id=decision_id,
            model_version=model_version,
            timestamp=f"deterministic_ts_{decision_id}"
        )

    # Convert resume to transaction for rule evaluation
    transaction = _resume_to_transaction_for_scoring(resume, extractor)

    # Identify firing rules and their contributions
    firing_rules = []
    for rule_idx, rule in enumerate(rules):
        rule_id = f"rule_{rule_idx}"

        # Check if rule fires
        if rule.antecedent.issubset(transaction):
            posterior = rule_posteriors.get(rule_id)
            if posterior:
                rule_firing = RuleFiring(
                    rule_id=rule_id,
                    antecedent=_antecedent_to_text(rule.antecedent),
                    posterior_mean_reliability=posterior.posterior_mean,
                    posterior_interval=posterior.credible_interval_95,
                    contribution_to_skill=posterior.posterior_mean  # Base contribution
                )
                firing_rules.append((rule, rule_firing, posterior))

    # Compute per-skill aptitudes
    aptitudes = {}
    all_skills = role.required_skills | role.preferred_skills

    for skill in all_skills:
        # Find rules that contribute to this skill
        skill_rules = []
        skill_contributions = []
        all_fairness_passed = True

        for rule, rule_firing, posterior in firing_rules:
            # Check if rule mentions this skill (in antecedent or consequent)
            if _rule_mentions_skill(rule, skill):
                skill_rules.append(rule_firing)
                skill_contributions.append(posterior.posterior_mean)
                if not posterior.passed_fairness_filter:
                    all_fairness_passed = False

        # Compute skill aptitude
        if skill_contributions:
            # Simple aggregation: mean of contributing rule reliabilities
            skill_score = np.mean(skill_contributions)

            # Uncertainty propagation: conservative approach
            # Use widest interval among contributing rules
            intervals = [rule_firing.posterior_interval for rule_firing in skill_rules]
            lower_bounds = [interval[0] for interval in intervals]
            upper_bounds = [interval[1] for interval in intervals]
            uncertainty_interval = (min(lower_bounds), max(upper_bounds))
        else:
            # No rules mention this skill
            skill_score = float('nan')
            uncertainty_interval = (float('nan'), float('nan'))

        aptitudes[skill] = SkillAptitude(
            skill=skill,
            score=skill_score,
            uncertainty_interval=uncertainty_interval,
            contributing_rules=skill_rules,
            fairness_filter_passed=all_fairness_passed
        )

    # Compute overall recommendation
    # Use required skills only for overall scoring
    required_scores = []
    required_uncertainties = []

    for skill in role.required_skills:
        if skill in aptitudes and not np.isnan(aptitudes[skill].score):
            required_scores.append(aptitudes[skill].score)
            required_uncertainties.extend(aptitudes[skill].uncertainty_interval)

    if required_scores:
        overall_score = np.mean(required_scores)
        overall_uncertainty = (
            min(required_uncertainties) if required_uncertainties else 0.0,
            max(required_uncertainties) if required_uncertainties else 1.0
        )

        # Deterministic thresholds (configurable)
        if overall_score >= 0.7:
            recommendation = "advance"
        elif overall_score >= 0.3:
            recommendation = "review"
        else:
            recommendation = "do_not_advance"
    else:
        # No required skill scores available
        recommendation = "review"
        overall_uncertainty = (0.0, 1.0)

    return CandidateScoring(
        aptitudes=aptitudes,
        overall_recommendation=recommendation,
        overall_uncertainty=overall_uncertainty,
        decision_id=decision_id,
        model_version=model_version,
        timestamp=f"deterministic_ts_{decision_id}"
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


def _resume_to_transaction_for_scoring(resume: Resume, extractor) -> set:
    """Convert resume to transaction set for rule evaluation."""
    features = extractor.extract_features(resume)
    transaction = set()

    # Add skill tokens
    transaction.update(resume.skill_tokens)

    # Add binned features that rules might reference
    if 'experience_bin' in features:
        transaction.add(features['experience_bin'])
    if 'education_level' in features:
        transaction.add(features['education_level'])
    if 'seniority_level' in features:
        transaction.add(features['seniority_level'])

    # Add domain background
    transaction.update(resume.domain_background)

    return transaction


def _rule_mentions_skill(rule: AssociationRule, skill: str) -> bool:
    """Check if rule mentions a specific skill in antecedent or consequent."""
    all_terms = rule.antecedent | rule.consequent

    # Direct mention
    if skill in all_terms:
        return True

    # Check for skill-related terms (simple heuristic)
    skill_lower = skill.lower()
    for term in all_terms:
        term_lower = term.lower()
        if skill_lower in term_lower or term_lower in skill_lower:
            return True

    return False


def _antecedent_to_text(antecedent: set) -> str:
    """Convert rule antecedent to human-readable text."""
    if not antecedent:
        return "always"

    terms = sorted(list(antecedent))
    if len(terms) == 1:
        return terms[0]
    elif len(terms) == 2:
        return f"{terms[0]} AND {terms[1]}"
    else:
        return f"{', '.join(terms[:-1])} AND {terms[-1]}"