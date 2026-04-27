"""Counterfactual fairness analysis through protected attribute swapping.

Measures |score_original - score_swapped| when gender/race tokens are swapped
to detect discrimination based on protected attributes.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
import copy

from ..rules.data import Resume


@dataclass
class CounterfactualResult:
    """Result of counterfactual fairness analysis."""
    attribute_name: str
    flip_rate_mean: float  # Mean absolute score difference
    flip_rate_p95: float   # 95th percentile score difference
    flip_rate_max: float   # Maximum score difference
    gate_passed: bool      # True if p95 ≤ threshold
    threshold: float       # Threshold for gate
    total_comparisons: int
    details: Dict[str, Any]


@dataclass
class CounterfactualComparison:
    """Single counterfactual comparison."""
    original_resume: Resume
    counterfactual_resume: Resume
    original_score: float
    counterfactual_score: float
    score_difference: float
    swapped_attribute: str
    swapped_values: Tuple[Any, Any]  # (original_value, counterfactual_value)


class CounterfactualAnalyzer:
    """Analyze counterfactual fairness through protected attribute manipulation.

    Creates counterfactual versions of resumes by swapping protected attributes
    (gender, race, etc.) and measures score differences to detect discrimination.
    """

    def __init__(self):
        """Initialize counterfactual analyzer."""
        # Define protected attribute swapping mappings
        self.attribute_swaps = {
            'gender': {
                'male': 'female',
                'female': 'male',
                'm': 'f',
                'f': 'm',
                0: 1,  # Binary encoding
                1: 0,
                'man': 'woman',
                'woman': 'man'
            },
            'race': {
                'white': 'black',
                'black': 'white',
                'asian': 'hispanic',
                'hispanic': 'asian',
                'caucasian': 'african_american',
                'african_american': 'caucasian',
                0: 1,  # Binary encoding
                1: 0
            },
            'ethnicity': {
                'hispanic': 'non_hispanic',
                'non_hispanic': 'hispanic',
                'latino': 'non_latino',
                'non_latino': 'latino',
                0: 1,
                1: 0
            }
        }

        # Gender/race tokens that might appear in skill lists or domains
        self.token_swaps = {
            # Gender-coded skill tokens
            'womens_studies': 'mens_studies',
            'mens_studies': 'womens_studies',
            'gender_studies': 'gender_studies',  # No swap needed

            # Name-based tokens (if they somehow appear)
            'john': 'jane',
            'jane': 'john',
            'michael': 'michelle',
            'michelle': 'michael',
            'david': 'diana',
            'diana': 'david',
            'james': 'jessica',
            'jessica': 'james'
        }

    def analyze_counterfactual_fairness(self,
                                      resumes: List[Resume],
                                      model_predict_fn,
                                      threshold: float = 0.05) -> Dict[str, CounterfactualResult]:
        """Analyze counterfactual fairness across protected attributes.

        Args:
            resumes: List of resumes to analyze
            model_predict_fn: Function that takes resume and returns probability
            threshold: Maximum acceptable p95 flip rate

        Returns:
            Dictionary mapping attribute_name -> CounterfactualResult
        """
        results = {}

        # Analyze each protected attribute
        for attr_name in ['gender', 'race', 'ethnicity']:
            comparisons = self._generate_counterfactual_comparisons(
                resumes, attr_name, model_predict_fn
            )

            if comparisons:
                result = self._calculate_counterfactual_metrics(
                    comparisons, attr_name, threshold
                )
                results[attr_name] = result

        return results

    def _generate_counterfactual_comparisons(self,
                                           resumes: List[Resume],
                                           attribute_name: str,
                                           model_predict_fn) -> List[CounterfactualComparison]:
        """Generate counterfactual comparisons for a specific attribute.

        Args:
            resumes: List of resumes
            attribute_name: Name of protected attribute to swap
            model_predict_fn: Prediction function

        Returns:
            List of counterfactual comparisons
        """
        comparisons = []

        for resume in resumes:
            # Create counterfactual version
            counterfactual_resume = self._create_counterfactual_resume(
                resume, attribute_name
            )

            # Skip if no meaningful counterfactual could be created
            if counterfactual_resume is None:
                continue

            # Get predictions for both versions
            try:
                original_prob = model_predict_fn(resume)
                counterfactual_prob = model_predict_fn(counterfactual_resume)

                # Extract probability (handle different return formats)
                if hasattr(original_prob, 'probability'):
                    original_score = original_prob.probability
                elif isinstance(original_prob, (list, np.ndarray)):
                    original_score = original_prob[1] if len(original_prob) > 1 else original_prob[0]
                else:
                    original_score = float(original_prob)

                if hasattr(counterfactual_prob, 'probability'):
                    counterfactual_score = counterfactual_prob.probability
                elif isinstance(counterfactual_prob, (list, np.ndarray)):
                    counterfactual_score = counterfactual_prob[1] if len(counterfactual_prob) > 1 else counterfactual_prob[0]
                else:
                    counterfactual_score = float(counterfactual_prob)

                score_difference = abs(original_score - counterfactual_score)

                # Get swapped values for reporting
                original_value = resume.demographics.get(attribute_name, 'unknown')
                counterfactual_value = counterfactual_resume.demographics.get(attribute_name, 'unknown')

                comparison = CounterfactualComparison(
                    original_resume=resume,
                    counterfactual_resume=counterfactual_resume,
                    original_score=original_score,
                    counterfactual_score=counterfactual_score,
                    score_difference=score_difference,
                    swapped_attribute=attribute_name,
                    swapped_values=(original_value, counterfactual_value)
                )
                comparisons.append(comparison)

            except Exception:
                # Skip resumes that cause prediction errors
                continue

        return comparisons

    def _create_counterfactual_resume(self,
                                    resume: Resume,
                                    attribute_name: str) -> Optional[Resume]:
        """Create counterfactual version of resume with swapped protected attribute.

        Args:
            resume: Original resume
            attribute_name: Protected attribute to swap

        Returns:
            Counterfactual resume or None if no swap possible
        """
        swapped = False
        new_demographics = resume.demographics.copy()
        new_skills = list(resume.skill_tokens)
        new_domains = list(resume.domain_background)

        # Swap demographic attribute if present
        if attribute_name in new_demographics:
            original_value = new_demographics[attribute_name]
            if attribute_name in self.attribute_swaps:
                swap_map = self.attribute_swaps[attribute_name]
                if original_value in swap_map:
                    new_demographics[attribute_name] = swap_map[original_value]
                    swapped = True

        # Swap any gender/race tokens in skill list
        if attribute_name in ['gender', 'race', 'ethnicity']:
            for i, skill in enumerate(new_skills):
                if skill in self.token_swaps:
                    new_skills[i] = self.token_swaps[skill]
                    swapped = True

            # Swap tokens in domain background
            for i, domain in enumerate(new_domains):
                if domain in self.token_swaps:
                    new_domains[i] = self.token_swaps[domain]
                    swapped = True

        # Return counterfactual only if we made meaningful swaps
        if swapped:
            return Resume(
                skill_tokens=new_skills,
                years_experience=resume.years_experience,
                education_level=resume.education_level,
                domain_background=new_domains,
                demographics=new_demographics
            )
        else:
            return None

    def _calculate_counterfactual_metrics(self,
                                        comparisons: List[CounterfactualComparison],
                                        attribute_name: str,
                                        threshold: float) -> CounterfactualResult:
        """Calculate metrics from counterfactual comparisons.

        Args:
            comparisons: List of comparisons
            attribute_name: Name of attribute
            threshold: Gate threshold

        Returns:
            CounterfactualResult with metrics
        """
        if not comparisons:
            return CounterfactualResult(
                attribute_name=attribute_name,
                flip_rate_mean=0.0,
                flip_rate_p95=0.0,
                flip_rate_max=0.0,
                gate_passed=True,
                threshold=threshold,
                total_comparisons=0,
                details={}
            )

        # Extract score differences
        score_diffs = [comp.score_difference for comp in comparisons]

        # Calculate statistics
        flip_rate_mean = np.mean(score_diffs)
        flip_rate_p95 = np.percentile(score_diffs, 95)
        flip_rate_max = np.max(score_diffs)

        # Check gate (p95 ≤ threshold)
        gate_passed = flip_rate_p95 <= threshold

        # Detailed analysis
        high_impact_comparisons = [
            comp for comp in comparisons if comp.score_difference > threshold
        ]

        details = {
            'score_differences': score_diffs,
            'flip_rate_p50': np.percentile(score_diffs, 50),
            'flip_rate_p90': np.percentile(score_diffs, 90),
            'flip_rate_p99': np.percentile(score_diffs, 99),
            'high_impact_count': len(high_impact_comparisons),
            'high_impact_rate': len(high_impact_comparisons) / len(comparisons),
            'largest_difference': {
                'score_diff': flip_rate_max,
                'original_score': comparisons[np.argmax(score_diffs)].original_score,
                'counterfactual_score': comparisons[np.argmax(score_diffs)].counterfactual_score,
                'swapped_values': comparisons[np.argmax(score_diffs)].swapped_values
            }
        }

        return CounterfactualResult(
            attribute_name=attribute_name,
            flip_rate_mean=flip_rate_mean,
            flip_rate_p95=flip_rate_p95,
            flip_rate_max=flip_rate_max,
            gate_passed=gate_passed,
            threshold=threshold,
            total_comparisons=len(comparisons),
            details=details
        )

    def analyze_specific_attribute(self,
                                 resumes: List[Resume],
                                 attribute_name: str,
                                 model_predict_fn,
                                 threshold: float = 0.05) -> CounterfactualResult:
        """Analyze counterfactual fairness for a specific attribute.

        Args:
            resumes: List of resumes to analyze
            attribute_name: Specific protected attribute ('gender', 'race', etc.)
            model_predict_fn: Prediction function
            threshold: Gate threshold

        Returns:
            CounterfactualResult for the attribute
        """
        comparisons = self._generate_counterfactual_comparisons(
            resumes, attribute_name, model_predict_fn
        )

        return self._calculate_counterfactual_metrics(
            comparisons, attribute_name, threshold
        )

    def get_detailed_comparisons(self,
                               resumes: List[Resume],
                               attribute_name: str,
                               model_predict_fn,
                               top_k: int = 10) -> List[CounterfactualComparison]:
        """Get detailed counterfactual comparisons for analysis.

        Args:
            resumes: List of resumes
            attribute_name: Protected attribute to analyze
            model_predict_fn: Prediction function
            top_k: Number of top differences to return

        Returns:
            List of top-k comparisons sorted by score difference
        """
        comparisons = self._generate_counterfactual_comparisons(
            resumes, attribute_name, model_predict_fn
        )

        # Sort by score difference (largest first)
        comparisons.sort(key=lambda x: x.score_difference, reverse=True)

        return comparisons[:top_k]

    def check_counterfactual_gates(self,
                                 results: Dict[str, CounterfactualResult],
                                 threshold: float = 0.05) -> Dict[str, bool]:
        """Check if counterfactual fairness gates pass for all attributes.

        Args:
            results: Results from analyze_counterfactual_fairness
            threshold: Gate threshold

        Returns:
            Dictionary mapping attribute -> gate_passed
        """
        gate_results = {}
        for attr_name, result in results.items():
            gate_results[attr_name] = result.flip_rate_p95 <= threshold

        return gate_results