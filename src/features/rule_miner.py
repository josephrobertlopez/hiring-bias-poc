"""FP-growth rule miner with fairness filtering."""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from itertools import combinations
from collections import Counter, defaultdict
import numpy as np

from ..rules.data import Resume, SkillVocabulary
from .extractors import ContentNeutralExtractor, JobRole


@dataclass
class AssociationRule:
    """Association rule with metrics."""
    antecedent: Set[str]
    consequent: Set[str]
    support: float
    confidence: float
    lift: float
    conviction: float


@dataclass
class RuleMinerConfig:
    """Configuration for rule mining."""
    min_support: float = 0.01
    min_confidence: float = 0.6
    min_lift: float = 1.2
    max_rule_length: int = 4
    top_k: int = 100


class FairnessFilteredRuleMiner:
    """Rule miner with fairness filtering.

    Implements frequent pattern mining over skill sets, binned experience,
    and role targets while filtering out protected attributes and proxies.
    """

    def __init__(self, config: RuleMinerConfig = None):
        self.config = config or RuleMinerConfig()
        self.rules: List[AssociationRule] = []
        self.frequent_itemsets: Dict[int, List[Tuple[Set[str], float]]] = {}

        # Protected attributes and known proxies to filter out
        self.protected_attributes = {
            # Direct protected attributes
            'gender', 'race', 'ethnicity', 'age', 'age_bucket', 'age_band',
            'religion', 'sexual_orientation', 'disability', 'veteran_status',
            'marital_status', 'pregnancy', 'genetic_information',

            # School prestige proxies
            'ivy_league', 'top_tier_school', 'elite_university',
            'school_ranking', 'university_prestige', 'school_tier',

            # Geographic/socioeconomic proxies
            'zip_code', 'zipcode', 'postal_code', 'neighborhood',
            'county', 'city_prestige', 'area_code', 'region_income',
            'school_district', 'home_value', 'commute_distance',

            # Name-based proxies
            'name_ethnicity', 'name_gender', 'name_origin',
            'first_name_pattern', 'last_name_pattern',

            # Other indirect proxies
            'family_status', 'dependents', 'childcare',
            'transportation_method', 'gap_reason'
        }

    def mine_rules(self,
                   resumes: List[Resume],
                   labels: List[bool],
                   extractor: ContentNeutralExtractor) -> List[AssociationRule]:
        """Mine association rules from resumes with fairness filtering."""
        # Convert resumes to transaction format
        transactions = self._resumes_to_transactions(resumes, labels, extractor)

        # Mine frequent itemsets
        self.frequent_itemsets = self._find_frequent_itemsets(transactions)

        # Generate association rules
        self.rules = self._generate_association_rules(transactions)

        # Filter out protected attributes and proxies
        self.rules = self._filter_protected_rules(self.rules)

        # Sort by lift and keep top-K
        self.rules.sort(key=lambda r: r.lift, reverse=True)
        self.rules = self.rules[:self.config.top_k]

        return self.rules

    def _resumes_to_transactions(self,
                                resumes: List[Resume],
                                labels: List[bool],
                                extractor: ContentNeutralExtractor) -> List[Set[str]]:
        transactions = []

        for resume, hired in zip(resumes, labels):
            transaction = set()

            # Add skill tokens directly (no prefix needed - these are what score_candidate matches)
            for skill in resume.skill_tokens:
                if not self._is_protected_attribute(skill):
                    transaction.add(skill)

            # Add binned experience
            features = extractor.extract_features(resume)
            experience_bin = features.get('experience_bin', 'unknown')
            if experience_bin != 'unknown':
                transaction.add(f"experience_{experience_bin}")

            # Add seniority level (job-relevant, not protected)
            seniority = features.get('seniority_level', 'unknown')
            if seniority != 'unknown':
                transaction.add(f"seniority_{seniority}")

            # Exclude education_level and domain_background - these are protected proxies
            # that the fairness filter is designed to remove

            # Add hiring outcome as consequent
            if hired:
                transaction.add("advance")
            else:
                transaction.add("do_not_advance")

            transactions.append(transaction)

        return transactions

    def _find_frequent_itemsets(self, transactions: List[Set[str]]) -> Dict[int, List[Tuple[Set[str], float]]]:
        n_transactions = len(transactions)
        min_support_count = max(1, int(self.config.min_support * n_transactions))

        frequent_itemsets = {}

        # Find frequent 1-itemsets
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        frequent_1 = []
        for item, count in item_counts.items():
            if count >= min_support_count:
                support = count / n_transactions
                frequent_1.append((frozenset([item]), support))

        frequent_itemsets[1] = [(set(itemset), support) for itemset, support in frequent_1]

        # Find frequent k-itemsets for k > 1
        k = 2
        while k <= self.config.max_rule_length and frequent_itemsets.get(k-1):
            candidates = self._generate_candidates(frequent_itemsets[k-1], k)
            frequent_k = []

            for candidate in candidates:
                count = 0
                for transaction in transactions:
                    if candidate.issubset(transaction):
                        count += 1

                if count >= min_support_count:
                    support = count / n_transactions
                    frequent_k.append((candidate, support))

            if frequent_k:
                frequent_itemsets[k] = frequent_k
            k += 1

        return frequent_itemsets

    def _generate_candidates(self, frequent_prev: List[Tuple[Set[str], float]], k: int) -> List[Set[str]]:
        candidates = []
        itemsets = [itemset for itemset, _ in frequent_prev]

        for i, itemset1 in enumerate(itemsets):
            for itemset2 in itemsets[i+1:]:
                # Join condition: itemsets should differ by exactly one item
                union = itemset1 | itemset2
                if len(union) == k:
                    candidates.append(union)

        return candidates

    def _generate_association_rules(self, transactions: List[Set[str]]) -> List[AssociationRule]:
        rules = []
        n_transactions = len(transactions)

        # Generate rules from itemsets of length >= 2
        for length in range(2, max(self.frequent_itemsets.keys()) + 1):
            if length not in self.frequent_itemsets:
                continue

            for itemset, support in self.frequent_itemsets[length]:
                # Generate all possible antecedent/consequent splits
                for r in range(1, len(itemset)):
                    for antecedent in combinations(itemset, r):
                        antecedent = set(antecedent)
                        consequent = itemset - antecedent

                        # Calculate confidence
                        antecedent_support = self._get_support(antecedent, transactions, n_transactions)
                        if antecedent_support == 0:
                            continue

                        confidence = support / antecedent_support

                        # Filter by confidence threshold
                        if confidence < self.config.min_confidence:
                            continue

                        # Calculate lift
                        consequent_support = self._get_support(consequent, transactions, n_transactions)
                        if consequent_support == 0:
                            continue

                        lift = confidence / consequent_support

                        # Filter by lift threshold
                        if lift < self.config.min_lift:
                            continue

                        # Calculate conviction
                        if confidence == 1.0:
                            conviction = float('inf')
                        else:
                            conviction = (1 - consequent_support) / (1 - confidence)

                        rule = AssociationRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support=support,
                            confidence=confidence,
                            lift=lift,
                            conviction=conviction
                        )
                        rules.append(rule)

        return rules

    def _get_support(self, itemset: Set[str], transactions: List[Set[str]], n_transactions: int) -> float:
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / n_transactions

    def _filter_protected_rules(self, rules: List[AssociationRule]) -> List[AssociationRule]:
        filtered_rules = []

        for rule in rules:
            # Check antecedent for protected attributes
            has_protected = False
            all_items = rule.antecedent | rule.consequent

            for item in all_items:
                if self._is_protected_attribute(item):
                    has_protected = True
                    break

            # Filter out rules with outcomes in antecedents (these can't fire on candidates)
            has_outcome_in_antecedent = any(
                outcome in rule.antecedent
                for outcome in ['advance', 'do_not_advance']
            )

            if not has_protected and not has_outcome_in_antecedent:
                filtered_rules.append(rule)

        return filtered_rules

    def _is_protected_attribute(self, item: str) -> bool:
        item_lower = item.lower()

        # Direct match with protected attributes
        if item_lower in self.protected_attributes:
            return True

        # Check for patterns in the item name
        protected_patterns = [
            'gender', 'race', 'ethnicity', 'age', 'religion',
            'zip', 'postal', 'neighborhood', 'county', 'district',
            'prestige', 'tier', 'ranking', 'elite', 'ivy',
            'name_', 'family', 'marital', 'veteran', 'disability'
        ]

        for pattern in protected_patterns:
            if pattern in item_lower:
                return True

        return False

    def get_rule_features(self, resume: Resume, extractor: ContentNeutralExtractor) -> Dict[str, int]:
        """Generate binary features for each rule."""
        features = {}

        # Convert resume to transaction format
        transaction = self._resumes_to_transactions([resume], [True], extractor)[0]
        # Remove outcome from transaction for feature generation
        transaction.discard("outcome_hired")
        transaction.discard("outcome_not_hired")

        for i, rule in enumerate(self.rules):
            # Check if antecedent fires
            rule_fires = rule.antecedent.issubset(transaction)
            features[f"rule_{i}_fires"] = int(rule_fires)

        return features

    def get_rule_explanations(self) -> List[Dict[str, Any]]:
        """Get human-readable explanations for each rule."""
        explanations = []

        for i, rule in enumerate(self.rules):
            explanation = {
                'rule_id': i,
                'antecedent': sorted(list(rule.antecedent)),
                'consequent': sorted(list(rule.consequent)),
                'support': round(rule.support, 3),
                'confidence': round(rule.confidence, 3),
                'lift': round(rule.lift, 3),
                'human_readable': self._rule_to_text(rule)
            }
            explanations.append(explanation)

        return explanations

    def _rule_to_text(self, rule: AssociationRule) -> str:
        antecedent_text = ", ".join(sorted(rule.antecedent))
        consequent_text = ", ".join(sorted(rule.consequent))

        return f"If {antecedent_text} then {consequent_text} (confidence: {rule.confidence:.2f}, lift: {rule.lift:.2f})"