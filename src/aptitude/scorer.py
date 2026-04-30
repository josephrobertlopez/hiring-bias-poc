"""Per-skill aptitude scoring."""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import subprocess
import numpy as np
from scipy.stats import beta

from ..rules.data import Resume
from ..features.extractors import JobRole
from ..posteriors.rule_reliability import RulePosterior, fit_rule_posteriors
from ..features.rule_miner import AssociationRule


@dataclass
class SkillAptitude:
    """Per-skill aptitude score with uncertainty and explanation."""
    skill: str
    score: float
    uncertainty_interval: Tuple[float, float]
    contributing_rules: List['RuleFiring']
    fairness_filter_passed: bool


@dataclass
class RuleFiring:
    """Rule contribution to skill aptitude score."""
    rule_id: str
    antecedent: str
    posterior_mean_reliability: float
    posterior_interval: Tuple[float, float]
    contribution_to_skill: float


@dataclass
class CandidateScoring:
    """Complete candidate evaluation with per-skill breakdown."""
    aptitudes: Dict[str, SkillAptitude]
    overall_recommendation: str
    overall_uncertainty: Tuple[float, float]
    decision_id: str
    model_version: str
    timestamp: str


def score_candidate(
    resume: Resume,
    role: JobRole,
    rules: List[AssociationRule] = None,
    rule_posteriors: Dict[str, RulePosterior] = None,
    extractor = None
) -> CandidateScoring:
    """Score candidate with per-skill aptitude breakdown."""
    # Generate deterministic decision ID
    resume_hash = _hash_resume(resume)
    role_id = _hash_role(role)
    try:
        model_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd='/', stderr=subprocess.DEVNULL).decode().strip()
    except:
        model_version = "1.0.0"
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
            timestamp=datetime.utcnow().isoformat()
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
        timestamp=datetime.utcnow().isoformat()
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

    # Add skill tokens directly (matches rule miner format)
    transaction.update(resume.skill_tokens)

    # Add binned features that rules might reference
    experience_bin = features.get('experience_bin', 'unknown')
    if experience_bin != 'unknown':
        transaction.add(f"experience_{experience_bin}")

    seniority = features.get('seniority_level', 'unknown')
    if seniority != 'unknown':
        transaction.add(f"seniority_{seniority}")

    # Exclude education_level and domain_background to match rule miner changes
    # (these were removed from rule mining to avoid protected attributes)

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