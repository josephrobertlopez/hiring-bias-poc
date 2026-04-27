"""Main SkillRulesEngine facade for resume auditing.

Gate 3: Aggregates all 6 rule types and provides high-level audit methods.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import numpy as np

from .data import Resume, SkillVocabulary, SkillTokenizer
from .implementations import (
    CombinationRuleImpl,
    ExperienceRuleImpl,
    EducationRuleImpl,
    DomainRuleImpl,
    GapRuleImpl,
    BiasRuleImpl,
)
from .bias_utils import compute_disparity_index


@dataclass
class SkillAuditResult:
    """Result of auditing a single resume."""

    resume_id: str
    overall_score: float  # [0-1]
    rule_scores: Dict[str, float]  # {"combination": 0.8, "experience": 0.6, ...}
    skill_patterns: List[str]  # ["[python, sql] → [hired]"]
    skill_gaps: List[str]  # ["tensorflow", "finance_experience"]
    bias_flags: List[str]  # ["demographic parity violation in gender"]
    recommendations: List[str]  # ["add tensorflow", "add finance_experience"]
    explanations: Dict[str, Any]  # Full explanations from each rule


@dataclass
class SkillReport:
    """Aggregate skill analysis across multiple resumes."""

    total_resumes: int
    skill_frequency: Dict[str, float]  # Skill appearance rates [0-1]
    skill_combinations: List[Tuple[Set[str], float]]  # [({"python", "sql"}, 0.15), ...]
    experience_thresholds: Dict[str, float]  # Required experience per skill (years)
    education_patterns: Dict[str, float]  # Education level frequencies [0-1]
    domain_patterns: Dict[str, float]  # Domain background frequencies [0-1]
    bias_metrics: Dict[str, Any]  # Demographic group counts from hiring


class SkillRulesEngine:
    """Main engine for resume skill auditing.

    Aggregates 6 independent rule types:
    - CombinationRule: Association rules for skill combinations
    - ExperienceRule: Experience thresholds per skill
    - EducationRule: Education level hiring patterns
    - DomainRule: Domain background patterns
    - GapRule: Critical skill identification
    - BiasRule: Demographic parity detection
    """

    def __init__(self, vocabulary: SkillVocabulary):
        """Initialize engine with skill vocabulary.

        Args:
            vocabulary: SkillVocabulary instance for tokenization

        Initializes all 6 rule instances in unfitted state.
        """
        self.vocabulary = vocabulary
        self.tokenizer = SkillTokenizer(vocabulary)

        # Initialize all 6 rule types in dictionary
        self.rules = {
            'combination': CombinationRuleImpl(),
            'experience': ExperienceRuleImpl(),
            'education': EducationRuleImpl(),
            'domain': DomainRuleImpl(),
            'gap': GapRuleImpl(),
            'bias': BiasRuleImpl(),
        }

        self.fitted = False
        self._training_labels: Optional[List[bool]] = None
        self._training_resumes: Optional[List[Resume]] = None

    def fit(self, resumes: List[Resume], labels: List[bool]) -> "SkillRulesEngine":
        """Train all rule types on historical data.

        Args:
            resumes: List of Resume instances
            labels: List of hiring decision labels (True = hired)

        Returns:
            Self for method chaining

        Each rule independently learns patterns from the labeled data.
        """
        self.rules['combination'].fit(resumes, labels)
        self.rules['experience'].fit(resumes, labels)
        self.rules['education'].fit(resumes, labels)
        self.rules['domain'].fit(resumes, labels)
        self.rules['gap'].fit(resumes, labels)
        self.rules['bias'].fit(resumes, labels)

        self.fitted = True
        self._training_labels = labels
        self._training_resumes = resumes
        return self

    def audit_resume(self, resume: Resume, resume_id: str = "unknown") -> SkillAuditResult:
        """Audit a single resume against all rule types.

        Args:
            resume: Resume to audit
            resume_id: Identifier for the resume

        Returns:
            SkillAuditResult with scores, patterns, gaps, bias flags, and recommendations

        Runs all 6 rules and aggregates results. Overall score is simple
        average of the 6 rule scores. Bias flags are generated when
        disparity index < 0.8 on any demographic attribute.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before audit_resume()")
        if self._training_resumes is None:
            raise RuntimeError("Training data not available")

        # Run all 6 rules and collect scores
        rule_scores = {
            "combination": self.rules['combination'].score(resume),
            "experience": self.rules['experience'].score(resume),
            "education": self.rules['education'].score(resume),
            "domain": self.rules['domain'].score(resume),
            "gap": self.rules['gap'].score(resume),
            "bias": self.rules['bias'].score(resume),
        }

        # Defensive clamping: ensure all rule scores are in [0-1] range
        rule_scores = {
            rule_name: max(0.0, min(1.0, score))
            for rule_name, score in rule_scores.items()
        }

        # Compute overall score as simple average
        overall_score = float(np.mean(list(rule_scores.values())))

        # Extract skill patterns as formatted strings
        skill_patterns = [
            f"{sorted(list(rule.antecedent))} → {sorted(list(rule.consequent))}"
            for rule in self.rules['combination'].rules
        ]

        # Extract skill gaps: critical skills missing from this resume
        skill_gaps = self.identify_skill_gaps(resume)

        # Generate bias flags from demographic disparity analysis
        bias_flags = []
        for attr_key in self.rules['bias'].total_counts.keys():
            total_counts = self.rules['bias'].total_counts[attr_key]
            hired_counts = self.rules['bias'].demographic_groups[attr_key]

            if total_counts and self.rules['bias'].total_hired > 0:
                # Calculate hiring rates per demographic group
                hiring_rates = []
                for group_val in total_counts.keys():
                    total = total_counts[group_val]
                    hired = hired_counts.get(group_val, 0)
                    if total > 0:
                        rate = hired / total
                        hiring_rates.append(rate)

                # Check for disparity index violation
                # Use more lenient threshold for small samples (< 10 candidates)
                if hiring_rates:
                    threshold = 0.7 if len(self._training_resumes) < 10 else 0.8
                    di_result = compute_disparity_index(hiring_rates, threshold)
                    if di_result["bias_detected"]:
                        bias_flags.append(f"demographic parity violation in {attr_key}")

        # Generate recommendations: add missing critical skills
        recommendations = [f"add {skill}" for skill in skill_gaps]

        # Get explanations from all 6 rules
        explanations = {
            "combination": self.rules['combination'].explain(resume),
            "experience": self.rules['experience'].explain(resume),
            "education": self.rules['education'].explain(resume),
            "domain": self.rules['domain'].explain(resume),
            "gap": self.rules['gap'].explain(resume),
            "bias": self.rules['bias'].explain(resume),
        }

        return SkillAuditResult(
            resume_id=resume_id,
            overall_score=overall_score,
            rule_scores=rule_scores,
            skill_patterns=skill_patterns,
            skill_gaps=skill_gaps,
            bias_flags=bias_flags,
            recommendations=recommendations,
            explanations=explanations,
        )

    def audit_batch(
        self, resumes: List[Resume], resume_ids: Optional[List[str]] = None
    ) -> List[SkillAuditResult]:
        """Audit multiple resumes efficiently.

        Args:
            resumes: List of Resume instances
            resume_ids: Optional list of IDs (defaults to "resume_{i}")

        Returns:
            List of SkillAuditResult objects
        """
        if resume_ids is None:
            resume_ids = [f"resume_{i}" for i in range(len(resumes))]

        return [self.audit_resume(r, rid) for r, rid in zip(resumes, resume_ids)]

    def get_skill_patterns(
        self, min_support: float = 0.1
    ) -> List[Tuple[Set[str], float]]:
        """Get discovered skill combination patterns.

        Args:
            min_support: Minimum support threshold [0-1]

        Returns:
            List of (antecedent_set, support) tuples for patterns
            meeting the minimum support threshold
        """
        return [
            (rule.antecedent, rule.support)
            for rule in self.rules['combination'].rules
            if rule.support >= min_support
        ]

    def identify_skill_gaps(self, resume: Resume) -> List[str]:
        """Identify critical skill gaps for a resume.

        Args:
            resume: Resume to analyze

        Returns:
            Sorted list of critical skills missing from resume
        """
        return sorted(
            list(self.rules['gap'].critical_skills - set(resume.skill_tokens))
        )

    def check_bias(
        self, resumes: List[Resume], protected_attrs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for demographic bias in hiring patterns.

        Args:
            resumes: List of resumes to analyze (used for context)
            protected_attrs: List of demographic attribute dicts

        Returns:
            Dict with disparity indices and bias flags per attribute
        """
        bias_analysis = {}

        for attr_key in self.rules['bias'].total_counts.keys():
            total_counts = self.rules['bias'].total_counts[attr_key]
            hired_counts = self.rules['bias'].demographic_groups[attr_key]

            if not total_counts or self.rules['bias'].total_hired == 0:
                continue

            # Calculate hiring rate per demographic group
            rates = []
            for group_val in total_counts.keys():
                total = total_counts[group_val]
                hired = hired_counts.get(group_val, 0)
                if total > 0:
                    rate = hired / total
                    rates.append(rate)

            if rates:
                di_result = compute_disparity_index(rates, threshold=0.8)
                bias_analysis[attr_key] = {
                    "disparity_index": di_result["disparity_index"],
                    "rates": rates,
                    "flag": di_result["bias_detected"],
                }

        return bias_analysis

    def generate_report(self, resumes: List[Resume]) -> SkillReport:
        """Generate aggregate skill analysis report.

        Args:
            resumes: List of resumes to analyze

        Returns:
            SkillReport with aggregated statistics across batch

        Computes skill frequencies, common combinations, experience
        thresholds, education patterns, domain patterns, and bias metrics.
        """
        total_resumes = len(resumes)

        # Aggregate skill frequencies
        skill_frequency_counts = Counter(
            skill for resume in resumes for skill in resume.skill_tokens
        )
        skill_frequency = {
            skill: freq / total_resumes
            for skill, freq in skill_frequency_counts.items()
        }

        # Get skill combinations from CombinationRule
        skill_combinations = self.get_skill_patterns()

        # Get experience thresholds from ExperienceRule
        experience_thresholds = self.rules['experience'].skill_experience_thresholds

        # Aggregate education patterns
        education_counts = Counter(resume.education_level for resume in resumes)
        education_patterns = {
            level: count / total_resumes for level, count in education_counts.items()
        }

        # Aggregate domain patterns
        domain_counts = Counter(
            domain for resume in resumes for domain in resume.domain_background
        )
        domain_patterns = {
            domain: count / total_resumes for domain, count in domain_counts.items()
        }

        # Bias metrics from demographic groups
        bias_metrics = dict(self.rules['bias'].demographic_groups)

        return SkillReport(
            total_resumes=total_resumes,
            skill_frequency=skill_frequency,
            skill_combinations=skill_combinations,
            experience_thresholds=experience_thresholds,
            education_patterns=education_patterns,
            domain_patterns=domain_patterns,
            bias_metrics=bias_metrics,
        )

    def get_rule_explanations(
        self, resume: Resume
    ) -> Dict[str, Dict[str, Any]]:
        """Get detailed explanations from each rule type.

        Args:
            resume: Resume to explain

        Returns:
            Dict mapping rule type names to their explain() output
        """
        return {
            "combination": self.rules['combination'].explain(resume),
            "experience": self.rules['experience'].explain(resume),
            "education": self.rules['education'].explain(resume),
            "domain": self.rules['domain'].explain(resume),
            "gap": self.rules['gap'].explain(resume),
            "bias": self.rules['bias'].explain(resume),
        }
