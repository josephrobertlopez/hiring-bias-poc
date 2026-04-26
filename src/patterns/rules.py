from typing import List, Dict, Set, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict

@dataclass
class AssociationRule:
    antecedents: frozenset
    consequents: frozenset
    support: float
    confidence: float
    lift: float

@dataclass
class SkillVocabulary:
    all_skills: Set[str]
    technical_skills: Set[str]
    soft_skills: Set[str]

class AssociationRulesMiner:
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.8):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.skill_data = None
        self.outcomes = None
        self.rules = []
        self.total_records = 0
        self.total_hired = 0

    def fit(self, skill_data: List[Dict], outcomes: List[bool]) -> None:
        """Fit mining model from skill data and hiring outcomes"""
        self.skill_data = pd.DataFrame(skill_data)
        self.outcomes = np.array(outcomes, dtype=bool)
        self.total_records = len(outcomes)
        self.total_hired = np.sum(self.outcomes)
        self.p_hired = self.total_hired / self.total_records if self.total_records > 0 else 0.5

    def extract_rules(self) -> List[AssociationRule]:
        """Extract association rules ranked by confidence (descending)"""
        if self.skill_data is None or self.outcomes is None:
            return []

        rules = []

        # Mine single-feature rules
        for col in self.skill_data.columns:
            unique_values = self.skill_data[col].unique()
            for value in unique_values:
                antecedents = frozenset([f"{col}={value}"])
                support = self._calculate_support(antecedents)
                confidence = self._calculate_confidence(antecedents)
                lift = self._calculate_lift(confidence)

                if support >= self.min_support and confidence >= self.min_confidence and lift > 1.0:
                    rule = AssociationRule(
                        antecedents=antecedents,
                        consequents=frozenset(['hired=1']),
                        support=support,
                        confidence=confidence,
                        lift=lift
                    )
                    rules.append(rule)

        # Mine multi-feature rules (limited to 2-feature combinations for efficiency)
        feature_values = []
        for col in self.skill_data.columns:
            for val in self.skill_data[col].unique():
                feature_values.append(f"{col}={val}")

        for combo in combinations(feature_values, 2):
            antecedents = frozenset(combo)
            support = self._calculate_support(antecedents)
            confidence = self._calculate_confidence(antecedents)
            lift = self._calculate_lift(confidence)

            if support >= self.min_support and confidence >= self.min_confidence and lift > 1.0:
                rule = AssociationRule(
                    antecedents=antecedents,
                    consequents=frozenset(['hired=1']),
                    support=support,
                    confidence=confidence,
                    lift=lift
                )
                rules.append(rule)

        # Sort by confidence descending
        self.rules = sorted(rules, key=lambda x: x.confidence, reverse=True)
        return self.rules

    def filter_audit_compliant(self, rules: List[AssociationRule]) -> List[AssociationRule]:
        """Filter rules to audit-compliant threshold"""
        return [
            rule for rule in rules
            if rule.support >= self.min_support
            and rule.confidence >= self.min_confidence
            and rule.lift > 1.0
        ]

    def compute_group_coverage(
        self,
        rules: List[AssociationRule],
        protected_attrs: pd.Series
    ) -> Dict[Any, Dict[int, float]]:
        """Compute coverage of rules by protected group"""
        if self.skill_data is None or self.outcomes is None:
            return {}

        coverage = {}
        groups = protected_attrs.unique()

        for group in groups:
            group_mask = (protected_attrs == group).values
            hired_in_group = np.sum(self.outcomes[group_mask])

            if hired_in_group == 0:
                coverage[group] = {}
                continue

            group_coverage = {}
            for rule_idx, rule in enumerate(rules):
                rule_matches = self._matches_rule(rule.antecedents)
                rule_matches_in_group = rule_matches[group_mask]
                rule_matches_hired = self.outcomes[group_mask] & rule_matches_in_group

                coverage_pct = np.sum(rule_matches_hired) / hired_in_group if hired_in_group > 0 else 0.0
                group_coverage[rule_idx] = coverage_pct

            coverage[group] = group_coverage

        return coverage

    def generate_explanations(self, resume_skills: List[str]) -> List[str]:
        """Generate human-readable explanations for rules"""
        explanations = []

        for rule in self.rules:
            # Format antecedents
            antecedent_parts = list(rule.antecedents)
            antecedent_str = ' AND '.join(antecedent_parts)

            # Format coverage percentage
            coverage_pct = rule.support * 100

            # Format confidence percentage
            confidence_pct = rule.confidence * 100

            # Build explanation
            explanation = (
                f"Rule: {antecedent_str} → hired\n"
                f"  Coverage: {coverage_pct:.1f}% of decisions\n"
                f"  Confidence: {confidence_pct:.1f}%\n"
                f"  Lift: {rule.lift:.2f}x baseline"
            )
            explanations.append(explanation)

        return explanations

    def _calculate_support(self, antecedents: frozenset) -> float:
        """Calculate support: P(antecedents AND hired)"""
        matches = self._matches_rule(antecedents)
        hired_matches = np.sum(self.outcomes[matches])
        return hired_matches / self.total_records if self.total_records > 0 else 0.0

    def _calculate_confidence(self, antecedents: frozenset) -> float:
        """Calculate confidence: P(hired | antecedents)"""
        matches = self._matches_rule(antecedents)
        n_matches = np.sum(matches)

        if n_matches == 0:
            return 0.0

        hired_matches = np.sum(self.outcomes[matches])
        return hired_matches / n_matches

    def _calculate_lift(self, confidence: float) -> float:
        """Calculate lift: confidence / P(hired)"""
        if self.p_hired == 0:
            return 0.0
        return confidence / self.p_hired

    def _matches_rule(self, antecedents: frozenset) -> np.ndarray:
        """Get boolean mask of records matching antecedents"""
        if self.skill_data is None:
            return np.zeros(0, dtype=bool)

        matches = np.ones(len(self.skill_data), dtype=bool)

        for antecedent in antecedents:
            if '=' in antecedent:
                col, val = antecedent.split('=', 1)
                col_matches = (self.skill_data[col].astype(str) == val).values
                matches &= col_matches

        return matches

    def discretize_continuous(
        self,
        data: pd.DataFrame,
        continuous_cols: List[str],
        n_bins: int = 5
    ) -> pd.DataFrame:
        """Discretize continuous features into quintiles/bins"""
        df = data.copy()
        for col in continuous_cols:
            if col in df.columns:
                df[f'{col}_binned'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        return df

    def validate_on_test_set(
        self,
        train_data: List[Dict],
        train_outcomes: List[bool],
        test_data: List[Dict],
        test_outcomes: List[bool]
    ) -> Dict[str, float]:
        """Validate rules on held-out test set"""
        # Fit on train
        self.fit(train_data, train_outcomes)
        self.extract_rules()

        # Evaluate on test
        test_df = pd.DataFrame(test_data)
        test_outcomes = np.array(test_outcomes, dtype=bool)

        metrics = {}
        for rule in self.rules[:5]:  # Evaluate top 5 rules
            rule_matches = self._matches_rule_on_data(rule.antecedents, test_df)
            n_matches = np.sum(rule_matches)

            if n_matches > 0:
                hired_matches = np.sum(test_outcomes[rule_matches])
                test_confidence = hired_matches / n_matches
                test_support = hired_matches / len(test_outcomes)

                metrics[str(rule.antecedents)] = {
                    'train_confidence': rule.confidence,
                    'test_confidence': test_confidence,
                    'degradation': abs(rule.confidence - test_confidence)
                }

        return metrics

    def _matches_rule_on_data(self, antecedents: frozenset, data: pd.DataFrame) -> np.ndarray:
        """Get boolean mask for rule matches on given data"""
        matches = np.ones(len(data), dtype=bool)

        for antecedent in antecedents:
            if '=' in antecedent:
                col, val = antecedent.split('=', 1)
                if col in data.columns:
                    col_matches = (data[col].astype(str) == val).values
                    matches &= col_matches

        return matches
