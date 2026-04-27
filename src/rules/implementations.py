"""Rule implementations for SkillRulesEngine.

Complete implementations of 6 rule types:
- CombinationRule: Association rules mining for skill combinations
- ExperienceRule: Skill-specific experience thresholds
- EducationRule: Education level category matching
- DomainRule: Domain background pattern matching
- GapRule: Critical skill gap detection
- BiasRule: Demographic parity fairness metrics
"""

from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np

from .protocols import RuleProtocol
from .data import Resume
from .bias_utils import compute_disparity_index


@dataclass
class AssociationRule:
    """Discovered association rule: antecedent -> consequent."""

    antecedent: Set[str]  # {"python", "sql"}
    consequent: Set[str]  # {"hired"}
    support: float  # P(antecedent ∪ consequent)
    confidence: float  # P(consequent | antecedent)
    lift: float  # confidence / P(consequent)


class CombinationRuleImpl(RuleProtocol):
    """Association rules for skill combinations using apriori algorithm.

    Mines frequent skill pairs and triplets that predict hiring.
    Learns both positive patterns (from hired) and negative patterns (from rejected).
    Scores based on presence of discriminative skills.
    """

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """Initialize combination rule with support/confidence thresholds.

        Args:
            min_support: Minimum support threshold [0-1]
            min_confidence: Minimum confidence threshold [0-1]
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules: List[AssociationRule] = []
        self.positive_skills: Counter = Counter()
        self.negative_skills: Counter = Counter()
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "CombinationRuleImpl":
        """Learn skill patterns from both hired and rejected candidates.

        Tracks which skills appear more frequently in hired vs rejected resumes.
        This enables discriminative scoring based on positive/negative skill presence.

        Args:
            resumes: List of resumes
            labels: List of hiring labels (True = hired)

        Returns:
            Self for method chaining
        """
        hired_resumes = [r for r, label in zip(resumes, labels) if label]
        rejected_resumes = [r for r, label in zip(resumes, labels) if not label]

        # Learn positive skills from hired candidates
        for resume in hired_resumes:
            self.positive_skills.update(resume.skill_tokens)

        # Learn negative skills from rejected candidates
        for resume in rejected_resumes:
            self.negative_skills.update(resume.skill_tokens)

        # Also maintain original association rules for explain()
        total_hired = len(hired_resumes)
        if total_hired == 0:
            self.fitted = True
            return self

        all_skills = set()
        skill_counts = Counter()

        for resume in hired_resumes:
            skills = set(resume.skill_tokens)
            all_skills.update(skills)
            skill_counts.update(skills)

        # Mine skill pairs
        for skill1, skill2 in combinations(sorted(all_skills), 2):
            pair_count = sum(
                1
                for r in hired_resumes
                if skill1 in r.skill_tokens and skill2 in r.skill_tokens
            )

            support = pair_count / total_hired
            if support < self.min_support:
                continue

            skill1_count = skill_counts[skill1]
            if skill1_count == 0:
                continue
            confidence = pair_count / skill1_count

            if confidence < self.min_confidence:
                continue

            skill2_prob = skill_counts[skill2] / total_hired
            if skill2_prob == 0:
                continue
            lift = confidence / skill2_prob

            rule = AssociationRule(
                antecedent={skill1},
                consequent={skill2},
                support=support,
                confidence=confidence,
                lift=lift,
            )
            self.rules.append(rule)

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if resume has any strong skill combinations.

        Args:
            resume: Resume to evaluate

        Returns:
            True if resume matches at least one rule
        """
        resume_skills = set(resume.skill_tokens)
        for rule in self.rules:
            if rule.antecedent.issubset(resume_skills):
                return True
        return False

    def score(self, resume: Resume) -> float:
        """Score resume based on positive vs negative skill presence.

        Compares positive skills (associated with hiring) against negative skills
        (associated with rejection) to produce discriminative scores.

        Args:
            resume: Resume to score

        Returns:
            Score based on positive/negative skill ratio [0-1]
        """
        if not self.fitted:
            return 0.5

        skill_set = set(resume.skill_tokens)
        if not skill_set:
            return 0.5

        # Count positive and negative skills
        positive_count = len(skill_set & set(self.positive_skills.keys()))
        negative_count = len(skill_set & set(self.negative_skills.keys()))
        total_skills = len(skill_set)

        if total_skills == 0:
            return 0.5

        # Score: more positive, fewer negative = higher score
        # Formula: (positive - negative + total) / (2 * total)
        # Centers score around 0.5, ranges from 0 to 1
        raw_score = (positive_count - negative_count + total_skills) / (
            2.0 * total_skills
        )
        return max(0.0, min(1.0, raw_score))

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain which skill combinations triggered.

        Args:
            resume: Resume to explain

        Returns:
            Dict with list of triggered rules and their metrics
        """
        resume_skills = set(resume.skill_tokens)
        triggered_rules = []

        for rule in self.rules:
            if rule.antecedent.issubset(resume_skills):
                triggered_rules.append({
                    "antecedent": sorted(list(rule.antecedent)),
                    "consequent": sorted(list(rule.consequent)),
                    "confidence": float(rule.confidence),
                    "lift": float(rule.lift),
                    "support": float(rule.support),
                })

        return {
            "rule_type": "combination",
            "matched": len(triggered_rules) > 0,
            "triggered_rules": triggered_rules,
            "count": len(triggered_rules),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "combination"


class ExperienceRuleImpl(RuleProtocol):
    """Rules about experience thresholds for skills.

    Learns median experience requirement for each skill from hired candidates.
    """

    def __init__(self):
        """Initialize experience rule."""
        self.skill_experience_thresholds: Dict[str, float] = {}
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "ExperienceRuleImpl":
        """Learn experience thresholds per skill.

        For each skill, compute median experience of hired candidates.

        Args:
            resumes: List of resumes
            labels: List of hiring labels

        Returns:
            Self for method chaining
        """
        hired_resumes = [r for r, label in zip(resumes, labels) if label]

        if not hired_resumes:
            self.fitted = True
            return self

        # Group experience by skill
        skill_experiences: Dict[str, List[float]] = defaultdict(list)

        for resume in hired_resumes:
            for skill in resume.skill_tokens:
                skill_experiences[skill].append(resume.years_experience)

        # Compute median experience per skill
        for skill, experiences in skill_experiences.items():
            self.skill_experience_thresholds[skill] = float(
                np.median(experiences)
            )

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if experience meets thresholds for claimed skills.

        Args:
            resume: Resume to evaluate

        Returns:
            True if all skills meet their thresholds
        """
        if not self.fitted or not self.skill_experience_thresholds:
            return True

        for skill in resume.skill_tokens:
            threshold = self.skill_experience_thresholds.get(skill, 0.0)
            if resume.years_experience < threshold:
                return False

        return True

    def score(self, resume: Resume) -> float:
        """Score based on experience adequacy.

        Args:
            resume: Resume to score

        Returns:
            Average ratio of actual/required experience [0-1]
        """
        if not self.fitted or not self.skill_experience_thresholds:
            return 0.5

        if not resume.skill_tokens:
            return 0.5

        ratios = []
        for skill in resume.skill_tokens:
            threshold = self.skill_experience_thresholds.get(skill, 1.0)
            if threshold > 0:
                ratio = min(resume.years_experience / threshold, 1.0)
                ratios.append(ratio)

        return sum(ratios) / len(ratios) if ratios else 0.5

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain experience gaps/strengths.

        Args:
            resume: Resume to explain

        Returns:
            Dict with per-skill experience analysis
        """
        if not self.fitted:
            return {"status": "not_fitted"}

        skills_analysis = []
        for skill in resume.skill_tokens:
            threshold = self.skill_experience_thresholds.get(skill)
            if threshold is not None:
                gap = resume.years_experience - threshold
                status = "meets" if gap >= 0 else "gap"
                skills_analysis.append({
                    "skill": skill,
                    "required": float(threshold),
                    "actual": float(resume.years_experience),
                    "gap": float(gap),
                    "status": status,
                })

        return {
            "rule_type": "experience",
            "skills": skills_analysis,
            "matches": self.matches(resume),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "experience"


class EducationRuleImpl(RuleProtocol):
    """Rules about education requirements.

    Learns hiring rates for each education level from all candidates.
    Scores based on the hiring rate of the candidate's education level.
    """

    def __init__(self):
        """Initialize education rule."""
        self.education_scores: Dict[str, float] = {}
        self.education_counts: Dict[str, Tuple[int, int]] = {}  # (hired, total)
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "EducationRuleImpl":
        """Learn education level hiring rates from all candidates.

        Computes hiring rate = hired_count / total_count for each education level.
        This enables discriminative scoring based on actual hiring patterns.

        Args:
            resumes: List of resumes
            labels: List of hiring labels (True = hired)

        Returns:
            Self for method chaining
        """
        hired_counts: Counter = Counter()
        total_counts: Counter = Counter()

        for resume, label in zip(resumes, labels):
            edu_level = resume.education_level
            total_counts[edu_level] += 1
            if label:
                hired_counts[edu_level] += 1

        # Compute hiring rate for each education level
        self.education_scores = {}
        self.education_counts = {}
        for edu_level in total_counts:
            total = total_counts[edu_level]
            hired = hired_counts.get(edu_level, 0)
            self.education_counts[edu_level] = (hired, total)
            self.education_scores[edu_level] = float(hired) / total if total > 0 else 0.5

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if resume education level appears in fitted data.

        Args:
            resume: Resume to evaluate

        Returns:
            True if education level found in training data
        """
        if not self.fitted:
            return True

        return resume.education_level in self.education_scores

    def score(self, resume: Resume) -> float:
        """Score based on education level hiring rate.

        Returns the hiring rate for the candidate's education level,
        enabling discrimination between candidates with different education backgrounds.

        Args:
            resume: Resume to score

        Returns:
            Hiring rate for this education level [0-1]
        """
        if not self.fitted:
            return 0.5

        return self.education_scores.get(resume.education_level, 0.5)

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain education level hiring rate analysis.

        Args:
            resume: Resume to explain

        Returns:
            Dict with education level info and hiring rate
        """
        if not self.fitted:
            return {"status": "not_fitted"}

        edu_level = resume.education_level
        hiring_rate = self.education_scores.get(edu_level, 0.5)
        hired, total = self.education_counts.get(edu_level, (0, 0))

        return {
            "rule_type": "education",
            "education_level": edu_level,
            "found_in_training": edu_level in self.education_scores,
            "hired_count": int(hired),
            "total_count": int(total),
            "hiring_rate": float(hiring_rate),
            "matches": self.matches(resume),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "education"


class DomainRuleImpl(RuleProtocol):
    """Rules about domain background patterns.

    Learns hiring rates for each domain background from both hired and rejected candidates.
    Scores based on the hiring rate of the candidate's domains.
    """

    def __init__(self):
        """Initialize domain rule."""
        self.domain_scores: Dict[str, float] = {}
        self.domain_counts: Dict[str, Tuple[int, int]] = {}  # (hired, total)
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "DomainRuleImpl":
        """Learn domain background hiring rates from all candidates.

        Computes hiring rate = hired_count / total_count for each domain.
        This enables discriminative scoring based on actual hiring patterns.

        Args:
            resumes: List of resumes
            labels: List of hiring labels (True = hired)

        Returns:
            Self for method chaining
        """
        hired_counts: Counter = Counter()
        total_counts: Counter = Counter()

        for resume, label in zip(resumes, labels):
            for domain in resume.domain_background:
                total_counts[domain] += 1
                if label:
                    hired_counts[domain] += 1

        # Compute hiring rate for each domain
        self.domain_scores = {}
        self.domain_counts = {}
        for domain in total_counts:
            total = total_counts[domain]
            hired = hired_counts.get(domain, 0)
            self.domain_counts[domain] = (hired, total)
            self.domain_scores[domain] = float(hired) / total if total > 0 else 0.5

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if resume has any domain in fitted data.

        Args:
            resume: Resume to evaluate

        Returns:
            True if at least one domain found in training data
        """
        if not self.fitted:
            return True

        for domain in resume.domain_background:
            if domain in self.domain_scores:
                return True

        return False

    def score(self, resume: Resume) -> float:
        """Score based on domain background hiring rates.

        Averages the hiring rates of all the candidate's domain backgrounds,
        enabling discrimination between candidates with different domain mixes.

        Args:
            resume: Resume to score

        Returns:
            Average hiring rate of resume domains [0-1]
        """
        if not self.fitted:
            return 0.5

        if not resume.domain_background:
            return 0.5

        domain_scores = []
        for domain in resume.domain_background:
            score = self.domain_scores.get(domain, 0.5)
            domain_scores.append(score)

        return float(np.mean(domain_scores)) if domain_scores else 0.5

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain domain background hiring rate analysis.

        Args:
            resume: Resume to explain

        Returns:
            Dict with per-domain hiring rate analysis
        """
        if not self.fitted:
            return {"status": "not_fitted"}

        domain_analysis = []
        for domain in resume.domain_background:
            hiring_rate = self.domain_scores.get(domain, 0.5)
            hired, total = self.domain_counts.get(domain, (0, 0))
            domain_analysis.append({
                "domain": domain,
                "found_in_training": domain in self.domain_scores,
                "hired_count": int(hired),
                "total_count": int(total),
                "hiring_rate": float(hiring_rate),
            })

        return {
            "rule_type": "domain",
            "domains": domain_analysis,
            "matches": self.matches(resume),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "domain"


class GapRuleImpl(RuleProtocol):
    """Rules about skill gaps.

    Identifies critical skills that appear in majority of hired candidates.
    Flags resumes missing critical skills.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize gap rule.

        Args:
            threshold: Minimum frequency to mark skill as critical [0-1]
        """
        self.threshold = threshold
        self.critical_skills: Set[str] = set()
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "GapRuleImpl":
        """Identify critical skills from hired candidates.

        Skills appearing in >threshold of hired candidates are critical.

        Args:
            resumes: List of resumes
            labels: List of hiring labels

        Returns:
            Self for method chaining
        """
        hired_resumes = [r for r, label in zip(resumes, labels) if label]
        total_hired = len(hired_resumes)

        if total_hired == 0:
            self.fitted = True
            return self

        # Count skill occurrences
        skill_counts: Counter = Counter()
        for resume in hired_resumes:
            skill_counts.update(resume.skill_tokens)

        # Find critical skills
        self.critical_skills = {
            skill
            for skill, count in skill_counts.items()
            if count / total_hired > self.threshold
        }

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if resume has all critical skills.

        Args:
            resume: Resume to evaluate

        Returns:
            True if no critical skills are missing
        """
        if not self.fitted or not self.critical_skills:
            return True

        resume_skills = set(resume.skill_tokens)
        missing = self.critical_skills - resume_skills

        return len(missing) == 0

    def score(self, resume: Resume) -> float:
        """Score based on critical skill coverage with penalties for gaps.

        Provides discriminative scoring: candidates missing critical skills score lower.
        When all candidates have all critical skills, returns neutral score to allow
        other rules to discriminate.

        Args:
            resume: Resume to score

        Returns:
            Fraction of critical skills present [0-1]
        """
        if not self.fitted or not self.critical_skills:
            return 0.5

        resume_skills = set(resume.skill_tokens)
        present = len(self.critical_skills & resume_skills)
        total_critical = len(self.critical_skills)

        if total_critical == 0:
            return 0.5

        coverage = float(present) / total_critical

        # Apply penalty for missing skills (make it more discriminative)
        # When missing skills: apply exponential penalty
        # When all skills present: return neutral to let other rules discriminate
        if coverage < 1.0:
            # Exponential penalty for missing skills
            return coverage ** 1.5
        else:
            # All critical skills present - return neutral so other rules can discriminate
            return 0.5

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain missing critical skills.

        Args:
            resume: Resume to explain

        Returns:
            Dict with missing skills list and counts
        """
        if not self.fitted:
            return {"status": "not_fitted"}

        resume_skills = set(resume.skill_tokens)
        missing = self.critical_skills - resume_skills

        return {
            "rule_type": "gap",
            "critical_skills": sorted(list(self.critical_skills)),
            "missing_skills": sorted(list(missing)),
            "missing_count": len(missing),
            "total_critical": len(self.critical_skills),
            "matches": self.matches(resume),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "gap"


class BiasRuleImpl(RuleProtocol):
    """Rules for bias detection using fairness metrics.

    Computes demographic parity and disparity indices across protected attributes.
    """

    def __init__(self, demographic_keys: List[str] = None):
        """Initialize bias rule.

        Args:
            demographic_keys: List of demographic attributes to track
        """
        self.demographic_keys = demographic_keys or []
        self.demographic_groups: Dict[str, Counter] = defaultdict(Counter)
        self.total_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_hired = 0
        self.fitted = False

    def fit(
        self, resumes: List[Resume], labels: List[bool]
    ) -> "BiasRuleImpl":
        """Learn hiring rates per demographic group.

        Track both hired and total counts per demographic value
        to enable accurate fairness metric computation.

        Args:
            resumes: List of resumes
            labels: List of hiring labels (True = hired)

        Returns:
            Self for method chaining
        """
        self.demographic_groups = defaultdict(Counter)
        self.total_counts = defaultdict(Counter)
        self.total_hired = 0

        for resume, label in zip(resumes, labels):
            # Track each demographic attribute for all candidates
            for key in resume.demographics.keys():
                value = resume.demographics.get(key, "unknown")

                # Always count total
                self.total_counts[key][value] += 1

                # Count hired if applicable
                if label:
                    self.demographic_groups[key][value] += 1
                    self.total_hired += 1

        self.fitted = True
        return self

    def matches(self, resume: Resume) -> bool:
        """Check if demographic attributes exist in fitted groups.

        Args:
            resume: Resume to evaluate

        Returns:
            True if resume has demographics in fitted groups
        """
        if not self.fitted or self.total_hired == 0:
            return True

        for key, value in resume.demographics.items():
            if key in self.demographic_groups:
                if value in self.demographic_groups[key]:
                    return True

        return False

    def score(self, resume: Resume) -> float:
        """Score based on hiring rate for demographic group.

        Args:
            resume: Resume to score

        Returns:
            Hiring rate for resume's demographic groups [0-1]
        """
        if not self.fitted or self.total_hired == 0:
            return 0.5

        scores = []
        for key, value in resume.demographics.items():
            if key in self.total_counts:
                total = self.total_counts[key].get(value, 0)
                if total > 0:
                    hired = self.demographic_groups[key].get(value, 0)
                    rate = float(hired) / total
                    scores.append(rate)

        return sum(scores) / len(scores) if scores else 0.5

    def explain(self, resume: Resume) -> Dict[str, Any]:
        """Explain bias metrics per demographic group.

        Args:
            resume: Resume to explain

        Returns:
            Dict with per-demographic hiring rates and disparity info
        """
        if not self.fitted:
            return {"status": "not_fitted"}

        demographic_analysis = []
        for key in self.total_counts.keys():
            value = resume.demographics.get(key, "unknown")

            # Get counts for this demographic value
            total = self.total_counts[key].get(value, 0)
            hired = self.demographic_groups[key].get(value, 0)
            rate = (hired / total) if total > 0 else 0.0

            # Compute disparity index using shared utility
            group_rates = []
            for group_val in self.total_counts[key].keys():
                group_total = self.total_counts[key][group_val]
                group_hired = self.demographic_groups[key].get(group_val, 0)
                if group_total > 0:
                    group_rates.append(group_hired / group_total)

            di_result = compute_disparity_index(group_rates)

            demographic_analysis.append({
                "attribute": key,
                "value": str(value),
                "hiring_rate": float(rate),
                "hired_count": int(hired),
                "total_count": int(total),
                "disparity_index": di_result["disparity_index"],
            })

        return {
            "rule_type": "bias",
            "total_hired": int(self.total_hired),
            "demographics": demographic_analysis,
            "matches": self.matches(resume),
        }

    @property
    def rule_type(self) -> str:
        """Return rule category."""
        return "bias"
