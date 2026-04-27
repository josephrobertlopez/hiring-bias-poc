"""Gate 5: Comprehensive pilot validation for SkillRulesEngine + Thompson integration.

Tests entire pipeline end-to-end with 5 realistic hiring scenarios to prove
the system works before production deployment.
"""

from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np

from .data import Resume, SkillVocabulary
from .engine import SkillRulesEngine
from .thompson_classifier import ThompsonRulesClassifier


@dataclass(frozen=True)
class PilotScenario:
    """Test scenario with expected behavior and success criteria."""
    name: str
    resumes: List[Resume]
    labels: List[bool]
    expected_patterns: List[str]
    expected_gaps: List[str]
    expected_bias: bool
    success_criteria: Dict[str, float]


class PilotValidator:
    """Validates entire SkillRulesEngine + Thompson system end-to-end."""

    def __init__(self) -> None:
        """Initialize validator with realistic skill vocabulary."""
        tokens = [
            "python", "sql", "java", "javascript", "r", "scala", "c++",
            "machine_learning", "tensorflow", "pytorch", "sklearn", "pandas", "numpy",
            "statistics", "data_visualization", "deep_learning",
            "finance", "healthcare", "retail", "marketing", "operations",
            "aws", "docker", "kubernetes", "git", "tableau", "excel"
        ]
        categories = {
            "programming": ["python", "sql", "java", "javascript", "r", "scala", "c++"],
            "ml": ["machine_learning", "tensorflow", "pytorch", "sklearn", "pandas", "numpy"],
            "analytics": ["statistics", "data_visualization", "r", "tableau"],
            "domains": ["finance", "healthcare", "retail", "marketing", "operations"],
            "infrastructure": ["aws", "docker", "kubernetes", "git"]
        }
        self.vocabulary = SkillVocabulary(tokens, categories)

    def _create_pilot_scenarios(self) -> List[PilotScenario]:
        """Create 5 realistic hiring scenarios for validation."""
        scenarios: List[PilotScenario] = []

        # Scenario 1: Data Science Role - python+ML required
        resumes_hired = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 0}
            ),
            Resume(
                skill_tokens=["python", "sklearn"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["healthcare"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "machine_learning"],
                years_experience=3.0,
                education_level="bachelor",
                domain_background=["retail"],
                demographics={"gender": 0}
            )
        ]
        resumes_rejected = [
            Resume(
                skill_tokens=["sql"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["java"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["healthcare"],
                demographics={"gender": 0}
            )
        ]
        scenarios.append(PilotScenario(
            name="Data Science Role",
            resumes=resumes_hired + resumes_rejected,
            labels=[True] * 3 + [False] * 2,
            expected_patterns=["python", "machine_learning"],
            expected_gaps=["python"],
            expected_bias=False,  # No actual bias in this balanced data
            success_criteria={"accuracy": 0.8}
        ))

        # Scenario 2: Backend Engineering - experience bias
        resumes_hired = [
            Resume(
                skill_tokens=["java", "python"],
                years_experience=7.0,
                education_level="master",
                domain_background=["operations"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["java", "python"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["retail"],
                demographics={"gender": 0}
            )
        ]
        resumes_rejected = [
            Resume(
                skill_tokens=["java", "python"],
                years_experience=3.0,
                education_level="master",
                domain_background=["operations"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["java", "python"],
                years_experience=1.0,
                education_level="phd",
                domain_background=["retail"],
                demographics={"gender": 0}
            )
        ]
        scenarios.append(PilotScenario(
            name="Backend Engineering (experience bias)",
            resumes=resumes_hired + resumes_rejected,
            labels=[True] * 2 + [False] * 2,
            expected_patterns=["5+ years"],
            expected_gaps=["experience"],
            expected_bias=False,
            success_criteria={"accuracy": 0.75}
        ))

        # Scenario 3: Gender Bias - equal skills, unequal outcomes
        resumes_hired = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 0}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="phd",
                domain_background=["healthcare"],
                demographics={"gender": 0}
            )
        ]
        resumes_rejected = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["healthcare"],
                demographics={"gender": 1}
            )
        ]
        scenarios.append(PilotScenario(
            name="Gender Bias",
            resumes=resumes_hired + resumes_rejected,
            labels=[True] * 2 + [False] * 2,
            expected_patterns=["python", "tensorflow"],
            expected_gaps=[],
            expected_bias=True,
            success_criteria={"bias_detected": True}
        ))

        # Scenario 4: Education Requirements - PhD vs Master
        resumes_hired = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="phd",
                domain_background=["finance"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["healthcare"],
                demographics={"gender": 0}
            )
        ]
        resumes_rejected = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=5.0,
                education_level="master",
                domain_background=["healthcare"],
                demographics={"gender": 0}
            )
        ]
        scenarios.append(PilotScenario(
            name="Education Requirements",
            resumes=resumes_hired + resumes_rejected,
            labels=[True] * 2 + [False] * 2,
            expected_patterns=["phd"],
            expected_gaps=["education"],
            expected_bias=False,
            success_criteria={"accuracy": 0.75}
        ))

        # Scenario 5: Domain Expertise - finance domain required
        resumes_hired = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["finance"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["finance"],
                demographics={"gender": 0}
            )
        ]
        resumes_rejected = [
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=4.0,
                education_level="master",
                domain_background=["retail"],
                demographics={"gender": 1}
            ),
            Resume(
                skill_tokens=["python", "tensorflow"],
                years_experience=5.0,
                education_level="phd",
                domain_background=["retail"],
                demographics={"gender": 0}
            )
        ]
        scenarios.append(PilotScenario(
            name="Domain Expertise",
            resumes=resumes_hired + resumes_rejected,
            labels=[True] * 2 + [False] * 2,
            expected_patterns=["finance"],
            expected_gaps=["finance_experience"],
            expected_bias=False,
            success_criteria={"accuracy": 0.75}
        ))

        return scenarios

    def run_pilot(self) -> Dict[str, Any]:
        """Run all 5 pilot scenarios and validate results.

        Returns:
            Dict with scenarios_passed, scenarios_failed, results list, and overall_success flag.
            Overall success requires >= 4/5 scenarios passing.
        """
        scenarios = self._create_pilot_scenarios()
        results: List[Dict[str, Any]] = []

        for scenario in scenarios:
            try:
                result = self._test_scenario(scenario)
                results.append(result)
            except Exception as e:
                results.append({
                    "scenario": scenario.name,
                    "passed": False,
                    "accuracy": 0.0,
                    "patterns_found": 0,
                    "bias_detected": False,
                    "details": str(e)
                })

        scenarios_passed = sum(1 for r in results if r["passed"])
        return {
            "scenarios_passed": scenarios_passed,
            "scenarios_failed": len(scenarios) - scenarios_passed,
            "results": results,
            "overall_success": scenarios_passed >= 4
        }

    def _test_scenario(self, scenario: PilotScenario) -> Dict[str, Any]:
        """Test a single scenario end-to-end.

        Tests both SkillRulesEngine and ThompsonRulesClassifier, combines results,
        evaluates against success criteria.
        """
        # Test SkillRulesEngine
        engine = SkillRulesEngine(vocabulary=self.vocabulary)
        engine.fit(scenario.resumes, scenario.labels)
        predictions_engine = [
            audit.overall_score > 0.5
            for audit in engine.audit_batch(scenario.resumes)
        ]
        accuracy_engine = np.mean([
            pred == label
            for pred, label in zip(predictions_engine, scenario.labels)
        ])

        # Test ThompsonRulesClassifier
        thompson = ThompsonRulesClassifier(self.vocabulary)
        thompson.fit(scenario.resumes, scenario.labels)
        predictions_thompson = [
            p.prediction
            for p in thompson.predict_batch(scenario.resumes)
        ]
        accuracy_thompson = np.mean([
            pred == label
            for pred, label in zip(predictions_thompson, scenario.labels)
        ])

        # Combine results
        avg_accuracy = (accuracy_engine + accuracy_thompson) / 2
        patterns = engine.get_skill_patterns(min_support=0.2)
        bias_detected = any(
            audit.bias_flags
            for audit in engine.audit_batch(scenario.resumes)
        )

        # Evaluate success criteria
        passed = self._evaluate_success_criteria(
            scenario, avg_accuracy, patterns, bias_detected
        )

        return {
            "scenario": scenario.name,
            "passed": passed,
            "accuracy": avg_accuracy,
            "patterns_found": len(patterns),
            "bias_detected": bias_detected,
            "details": ""
            if passed
            else f"Accuracy: {avg_accuracy:.2f}, Patterns: {len(patterns)}, Bias: {bias_detected}"
        }

    def _evaluate_success_criteria(
        self,
        scenario: PilotScenario,
        accuracy: float,
        patterns: List[Tuple[Set[str], float]],
        bias_detected: bool
    ) -> bool:
        """Evaluate if scenario meets success criteria.

        Args:
            scenario: PilotScenario with criteria
            accuracy: Combined accuracy from both engines
            patterns: List of discovered patterns
            bias_detected: Whether bias was flagged

        Returns:
            True if all applicable criteria are met
        """
        criteria = scenario.success_criteria

        # Check accuracy threshold
        if "accuracy" in criteria and accuracy < criteria["accuracy"]:
            return False

        # Check bias detection matches expectation
        if scenario.expected_bias != bias_detected:
            return False

        # Check expected patterns found
        for pattern in scenario.expected_patterns:
            pattern_found = False
            for discovered_pattern in patterns:
                if pattern.lower() in str(discovered_pattern).lower():
                    pattern_found = True
                    break
            if not pattern_found:
                return False  # Fail if expected pattern not found

        return True


def run_gate5_validation() -> bool:
    """Main entry point for Gate 5 validation.

    Runs all 5 pilot scenarios and prints formatted results.

    Returns:
        True if overall_success (>= 4/5 scenarios pass), False otherwise.
    """
    validator = PilotValidator()
    results = validator.run_pilot()

    print("GATE 5 PILOT VALIDATION")
    print("=" * 60)

    for result in results["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        icon = "✅" if result["passed"] else "❌"
        details = (
            f": {result['details']}"
            if not result["passed"] and result["details"]
            else ""
        )
        print(f"{icon} {status} {result['scenario']}{details}")

    print("=" * 60)
    print(f"Passed: {results['scenarios_passed']}/5")
    print(f"Overall success: {results['overall_success']}")

    return results["overall_success"]


if __name__ == "__main__":
    success = run_gate5_validation()
    exit(0 if success else 1)
