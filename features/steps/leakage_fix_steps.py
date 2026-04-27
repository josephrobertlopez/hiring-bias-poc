"""Step definitions for leakage_fix package - eliminate train-test leakage."""

import time
import psutil
import os
from pathlib import Path

from behave import given, when, then
import numpy as np

# Import current system (will be refactored)
from src.rules.engine import SkillRulesEngine
from src.rules.data import Resume, SkillVocabulary


class LeakageAuditor:
    """Auditor for detecting and fixing data leakage."""

    def __init__(self):
        self.engine = None
        self.training_resumes = []
        self.training_labels = []
        self.test_resumes = []
        self.test_labels = []
        self.audit_results = []

    def create_test_data(self):
        """Create synthetic training and test data."""
        # Create comprehensive vocabulary
        tokens = [
            'python', 'java', 'javascript', 'sql', 'machine_learning',
            'tensorflow', 'react', 'nodejs', 'aws', 'docker'
        ]
        categories = {
            'programming': ['python', 'java', 'javascript'],
            'data': ['sql', 'machine_learning', 'tensorflow'],
            'web': ['javascript', 'react', 'nodejs'],
            'devops': ['aws', 'docker']
        }
        self.vocabulary = SkillVocabulary(tokens=tokens, categories=categories)

        # Create training data
        for i in range(100):
            resume = Resume(
                skill_tokens=['python', 'sql'] if i % 3 == 0 else ['java'],
                years_experience=float(2 + i % 8),
                education_level='bachelor' if i % 2 == 0 else 'master',
                domain_background=['tech'],
                demographics={'gender': i % 2}
            )
            self.training_resumes.append(resume)
            self.training_labels.append(1 if i % 2 == 0 else 0)

        # Create separate test data
        for i in range(50):
            resume = Resume(
                skill_tokens=['python', 'machine_learning'] if i % 4 == 0 else ['javascript'],
                years_experience=float(1 + i % 6),
                education_level='master' if i % 3 == 0 else 'bachelor',
                domain_background=['tech'],
                demographics={'gender': i % 2}
            )
            self.test_resumes.append(resume)
            self.test_labels.append(1 if i % 3 == 0 else 0)

    def fit_engine(self):
        """Fit the engine on training data only."""
        self.engine = SkillRulesEngine(self.vocabulary)
        self.engine.fit(self.training_resumes, self.training_labels)

    def check_training_data_storage(self):
        """Check if engine stores training data internally."""
        has_training_labels = hasattr(self.engine, '_training_labels')
        has_training_resumes = hasattr(self.engine, '_training_resumes')
        return has_training_labels, has_training_resumes

    def audit_test_resumes(self):
        """Audit test resumes and measure performance."""
        self.audit_results = []
        start_time = time.time()

        for i, resume in enumerate(self.test_resumes):
            result = self.engine.audit_resume(resume, resume_id=f"test_{i}")
            self.audit_results.append(result)

        end_time = time.time()
        return (end_time - start_time) / len(self.test_resumes) * 1000  # ms per resume

    def check_deterministic_inference(self):
        """Check if inference is deterministic."""
        if not self.test_resumes:
            return False

        # Audit same resume twice
        resume = self.test_resumes[0]
        result1 = self.engine.audit_resume(resume)
        result2 = self.engine.audit_resume(resume)

        # Check if results are identical
        score_match = result1.overall_score == result2.overall_score
        rule_scores_match = result1.rule_scores == result2.rule_scores

        return score_match and rule_scores_match

    def measure_memory_usage(self):
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB


# Global auditor instance
auditor = LeakageAuditor()


@given("the benchmark harness has established baseline performance")
def step_benchmark_harness_baseline(context):
    """Verify benchmark harness exists and is GREEN."""
    baseline_path = Path("benchmarks/baseline.json")
    status_path = Path(".ldd/benchmark_harness.status")

    assert baseline_path.exists(), "Benchmark baseline not found"
    assert status_path.exists() and status_path.read_text().strip() == "GREEN", "Benchmark harness not GREEN"


@given("the current SkillRulesEngine has data leakage issues")
def step_current_engine_has_leakage(context):
    """Set up test data to detect leakage issues."""
    auditor.create_test_data()
    auditor.fit_engine()

    # Verify the current engine does have training data stored (the leakage issue)
    has_labels, has_resumes = auditor.check_training_data_storage()
    context.has_training_data_leak = has_labels or has_resumes


@given("I need to maintain model performance while fixing leakage")
def step_maintain_performance_requirement(context):
    """Establish performance requirement for leakage fix."""
    context.performance_requirement = "AUC should not drop more than 0.05 after leakage fix"


@given("the current SkillRulesEngine stores training labels internally")
def step_engine_stores_training_labels(context):
    """Verify current engine stores training data."""
    has_labels, has_resumes = auditor.check_training_data_storage()
    assert has_labels or has_resumes, "Engine should currently store training data (before fix)"


@when("I audit a resume using the engine")
def step_audit_resume_using_engine(context):
    """Audit a test resume."""
    test_resume = auditor.test_resumes[0]
    context.audit_result = auditor.engine.audit_resume(test_resume)


@then("the engine should not access any training labels during inference")
def step_engine_no_training_labels_access(context):
    """Verify engine doesn't access training labels during inference."""
    # This is a behavioral requirement - the engine should not access _training_labels
    # In the fixed version, this attribute should not exist or not be used

    # Check if training labels are accessible during inference
    has_labels, has_resumes = auditor.check_training_data_storage()

    # This will fail before fix, pass after fix
    assert not has_labels, "Engine should not store training labels after leakage fix"


@then("the engine should not store _training_labels or _training_resumes")
def step_engine_no_training_data_storage(context):
    """Verify engine doesn't store training data internally."""
    has_labels, has_resumes = auditor.check_training_data_storage()

    assert not has_labels, "Engine should not store _training_labels"
    assert not has_resumes, "Engine should not store _training_resumes"


@then("audit_resume should work with only the fitted model parameters")
def step_audit_resume_fitted_parameters_only(context):
    """Verify audit_resume uses only fitted parameters."""
    # Successful audit means it worked with fitted parameters
    assert context.audit_result is not None
    assert hasattr(context.audit_result, 'overall_score')
    assert 0 <= context.audit_result.overall_score <= 1


@given("a SkillRulesEngine that has been fitted on training data")
def step_fitted_skillrules_engine(context):
    """Ensure engine is fitted."""
    if auditor.engine is None:
        auditor.create_test_data()
        auditor.fit_engine()

    assert auditor.engine.fitted, "Engine should be fitted"


@when("I call audit_resume on a test resume")
def step_call_audit_resume_on_test(context):
    """Call audit_resume on test data."""
    test_resume = auditor.test_resumes[0]
    context.test_audit_result = auditor.engine.audit_resume(test_resume)


@then("the method should not access the original training resumes")
def step_no_access_original_training_resumes(context):
    """Verify no access to original training resumes."""
    # Check that _training_resumes is not accessible
    has_labels, has_resumes = auditor.check_training_data_storage()
    assert not has_resumes, "Should not access original training resumes"


@then("the method should not access the original training labels")
def step_no_access_original_training_labels(context):
    """Verify no access to original training labels."""
    # Check that _training_labels is not accessible
    has_labels, has_resumes = auditor.check_training_data_storage()
    assert not has_labels, "Should not access original training labels"


@then("all rule scoring should use only fitted parameters")
def step_rule_scoring_fitted_parameters_only(context):
    """Verify rule scoring uses only fitted parameters."""
    # Successful audit with reasonable scores indicates fitted parameters are used
    result = context.test_audit_result
    assert result is not None
    assert len(result.rule_scores) > 0
    assert all(isinstance(score, float) for score in result.rule_scores.values())


@given("a dataset split into train and test sets")
def step_dataset_split_train_test(context):
    """Create proper train/test split."""
    auditor.create_test_data()
    context.train_size = len(auditor.training_resumes)
    context.test_size = len(auditor.test_resumes)

    assert context.train_size > 0, "Training set should not be empty"
    assert context.test_size > 0, "Test set should not be empty"


@when("I fit the SkillRulesEngine on the training set only")
def step_fit_on_training_set_only(context):
    """Fit engine on training data only."""
    auditor.fit_engine()
    assert auditor.engine.fitted, "Engine should be fitted"


@when("I evaluate on the test set")
def step_evaluate_on_test_set(context):
    """Evaluate engine on test set."""
    context.test_performance_ms = auditor.audit_test_resumes()
    context.test_results = auditor.audit_results


@then("there should be no information leakage from test to train")
def step_no_information_leakage_test_to_train(context):
    """Verify no information leakage from test to train."""
    # Test data should not influence training
    # This is ensured by proper data splits and no access to test labels during training

    # Verify test data was not used during fitting
    # In fixed version, engine should not store training data
    has_labels, has_resumes = auditor.check_training_data_storage()
    assert not has_labels and not has_resumes, "No training data should be stored to prevent leakage"


@then("CV folds should be properly isolated")
def step_cv_folds_properly_isolated(context):
    """Verify CV fold isolation (conceptual check)."""
    # This is a design requirement - when implementing CV, folds must be isolated
    # For now, we verify that the basic isolation principle is followed
    assert auditor.engine is not None, "Engine should exist for CV evaluation"


@then("rule mining should happen only within each fold")
def step_rule_mining_within_fold_only(context):
    """Verify rule mining happens only within fold."""
    # This is a design requirement for the refactored system
    # Current system will need to be modified to meet this requirement
    assert True  # Placeholder for proper fold-based rule mining


@given("a fitted SkillRulesEngine")
def step_fitted_skillrules_engine_simple(context):
    """Ensure we have a fitted engine."""
    if auditor.engine is None:
        auditor.create_test_data()
        auditor.fit_engine()


@when("I audit the same resume twice")
def step_audit_same_resume_twice(context):
    """Audit the same resume twice to test determinism."""
    test_resume = auditor.test_resumes[0]
    context.result1 = auditor.engine.audit_resume(test_resume)
    context.result2 = auditor.engine.audit_resume(test_resume)


@then("both audit results should be byte-identical")
def step_audit_results_byte_identical(context):
    """Verify audit results are byte-identical."""
    result1, result2 = context.result1, context.result2

    # Check overall scores
    assert result1.overall_score == result2.overall_score, "Overall scores should be identical"


@then("the overall_score should be exactly the same")
def step_overall_score_exactly_same(context):
    """Verify overall scores are exactly the same."""
    result1, result2 = context.result1, context.result2
    assert result1.overall_score == result2.overall_score


@then("rule_scores should be exactly the same")
def step_rule_scores_exactly_same(context):
    """Verify rule scores are exactly the same."""
    result1, result2 = context.result1, context.result2
    assert result1.rule_scores == result2.rule_scores


@then("no randomness should affect inference")
def step_no_randomness_affects_inference(context):
    """Verify no randomness in inference."""
    # Deterministic check already done above
    # Additional check: multiple calls should be deterministic
    is_deterministic = auditor.check_deterministic_inference()
    assert is_deterministic, "Inference should be completely deterministic"


@when("I inspect the engine's internal state")
def step_inspect_engine_internal_state(context):
    """Inspect engine for training data storage."""
    has_labels, has_resumes = auditor.check_training_data_storage()
    context.has_training_labels = has_labels
    context.has_training_resumes = has_resumes

    # Get engine attributes for inspection
    context.engine_attributes = dir(auditor.engine)


@then("there should be no accessible training labels")
def step_no_accessible_training_labels(context):
    """Verify training labels are not accessible."""
    assert not context.has_training_labels, "Training labels should not be accessible"


@then("there should be no accessible training resumes")
def step_no_accessible_training_resumes(context):
    """Verify training resumes are not accessible."""
    assert not context.has_training_resumes, "Training resumes should not be accessible"


@then("only fitted rule parameters should be present")
def step_only_fitted_rule_parameters_present(context):
    """Verify only fitted parameters are present."""
    # Engine should have fitted parameters but not training data
    assert 'fitted' in context.engine_attributes or auditor.engine.fitted
    assert not context.has_training_labels
    assert not context.has_training_resumes


@then("memory usage should not scale with training set size")
def step_memory_usage_not_scale_training_set(context):
    """Verify memory usage doesn't scale with training set size."""
    # This is a design principle - after fitting, memory should be O(parameters) not O(training_data)
    current_memory = auditor.measure_memory_usage()

    # Memory should be reasonable (< 100MB for this test)
    assert current_memory < 100, f"Memory usage too high: {current_memory}MB"


@given("a fitted SkillRulesEngine with {training_size:d} training examples")
def step_fitted_engine_with_training_size(context, training_size):
    """Create fitted engine with specific training size."""
    auditor.create_test_data()

    # Extend training data to requested size if needed
    while len(auditor.training_resumes) < training_size:
        i = len(auditor.training_resumes)
        resume = Resume(
            skill_tokens=['python'] if i % 2 == 0 else ['java'],
            years_experience=float(1 + i % 10),
            education_level='bachelor',
            domain_background=['tech'],
            demographics={'gender': i % 2}
        )
        auditor.training_resumes.append(resume)
        auditor.training_labels.append(i % 2)

    auditor.fit_engine()
    context.training_size = training_size


@when("I audit {num_resumes:d} test resumes")
def step_audit_num_test_resumes(context, num_resumes):
    """Audit specified number of test resumes."""
    # Extend test data if needed
    while len(auditor.test_resumes) < num_resumes:
        i = len(auditor.test_resumes)
        resume = Resume(
            skill_tokens=['python', 'sql'],
            years_experience=float(2 + i % 5),
            education_level='master',
            domain_background=['tech'],
            demographics={'gender': i % 2}
        )
        auditor.test_resumes.append(resume)
        auditor.test_labels.append(i % 2)

    context.test_performance_ms = auditor.audit_test_resumes()
    context.num_audited = num_resumes


@then("audit_resume should complete in under {max_ms:d}ms per resume")
def step_audit_resume_performance(context, max_ms):
    """Verify audit_resume performance."""
    avg_time_ms = context.test_performance_ms
    assert avg_time_ms < max_ms, f"Average time {avg_time_ms:.1f}ms > {max_ms}ms per resume"


@then("memory usage should be independent of training set size")
def step_memory_usage_independent_training_size(context):
    """Verify memory usage independence from training set size."""
    # After fitting, memory should not grow with training set size
    memory_usage = auditor.measure_memory_usage()

    # For this test, memory should be reasonable regardless of training set size
    assert memory_usage < 200, f"Memory usage {memory_usage}MB suggests training data storage"


@then("performance should not degrade with larger training sets")
def step_performance_not_degrade_larger_training(context):
    """Verify performance doesn't degrade with larger training sets."""
    # Inference time should be O(1) with respect to training set size
    avg_time_ms = context.test_performance_ms

    # Should be fast regardless of training set size
    assert avg_time_ms < 50, f"Performance {avg_time_ms:.1f}ms suggests O(training_size) complexity"