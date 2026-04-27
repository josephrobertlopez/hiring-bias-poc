"""Step definitions for benchmark.feature"""

from behave import given, when, then
import json
from benchmark.harness import BenchmarkHarness


@given('the benchmark harness is initialized')
def step_init_harness(context):
    """Initialize the benchmark harness"""
    context.harness = BenchmarkHarness(random_seed=context.seed)


@when('I request the 5-task hiring suite')
def step_load_5_tasks(context):
    """Load the 5-task hiring suite"""
    context.tasks = context.harness.load_5_task_suite()


@then('I should get Software Developer task with gender bias patterns')
def step_check_dev_task(context):
    """Verify Software Developer task exists with gender bias"""
    assert 'software_developer' in context.tasks
    task = context.tasks['software_developer']
    assert 'X' in task, "Task must have feature matrix X"
    assert 'y' in task, "Task must have labels y"
    assert 'protected_attr' in task, "Task must have protected attributes"
    attr_name = task['protected_attr']['name']
    assert attr_name == 'gender', "Software Developer should focus on gender bias"
    assert len(task['X']) > 0, "Task must have samples"


@then('Financial Analyst task with race/education bias patterns')
def step_check_analyst_task(context):
    """Verify Financial Analyst task exists"""
    assert 'financial_analyst' in context.tasks
    task = context.tasks['financial_analyst']
    attr_name = task['protected_attr']['name']
    assert attr_name in ['race', 'education'], "Should have race or education attr"
    assert len(task['X']) > 0, "Task must have samples"


@then('Healthcare Worker task with age bias patterns')
def step_check_healthcare_task(context):
    """Verify Healthcare Worker task exists"""
    assert 'healthcare_worker' in context.tasks
    task = context.tasks['healthcare_worker']
    attr_name = task['protected_attr']['name']
    assert attr_name == 'age', "Healthcare Worker should focus on age bias"
    assert len(task['X']) > 0, "Task must have samples"


@then('Customer Service task with minimal bias baseline')
def step_check_customer_service_task(context):
    """Verify Customer Service task exists"""
    assert 'customer_service' in context.tasks
    task = context.tasks['customer_service']
    assert len(task['X']) > 0, "Task must have samples"


@then('Management Role task with intersectional challenges')
def step_check_management_task(context):
    """Verify Management Role task exists"""
    assert 'management_role' in context.tasks
    task = context.tasks['management_role']
    assert len(task['X']) > 0, "Task must have samples"


@then('each task should have resume features and hiring ground truth')
def step_check_task_structure(context):
    """Verify each task has required structure"""
    for task_name, task in context.tasks.items():
        assert 'X' in task, f"{task_name}: missing feature matrix X"
        assert 'y' in task, f"{task_name}: missing labels y"
        assert 'protected_attr' in task, f"{task_name}: missing protected attributes"
        x_shape = task['X'].shape[0]
        y_len = len(task['y'])
        assert x_shape == y_len, f"{task_name}: X and y size mismatch"


@given('benchmark tasks are loaded')
def step_load_benchmark_tasks(context):
    """Load benchmark tasks"""
    context.harness = BenchmarkHarness(random_seed=context.seed)
    context.tasks = context.harness.load_5_task_suite()


@when('I run baseline measurement on all 5 tasks')
def step_run_baseline_measurement(context):
    """Run baseline measurement on all tasks"""
    context.results = context.harness.measure_baseline(context.tasks)


@then('I should get AUC scores between 0.60-0.70 for random baseline')
def step_check_auc_baseline(context):
    """Verify AUC scores are in reasonable baseline range"""
    assert 'auc_scores' in context.results
    for task_name, auc in context.results['auc_scores'].items():
        assert 0.45 <= auc <= 0.75, f"{task_name} AUC {auc} outside range"


@then('disparate impact ratios showing known bias patterns')
def step_check_di_ratios(context):
    """Verify disparate impact ratios are computed"""
    assert 'disparate_impact' in context.results
    for task_name, di in context.results['disparate_impact'].items():
        assert 0.0 <= di <= 1.0, f"{task_name} DI {di} outside [0,1]"


@then('flip rates measuring demographic stability')
def step_check_flip_rates(context):
    """Verify flip rates are computed"""
    assert 'flip_rates' in context.results
    for task_name, flip in context.results['flip_rates'].items():
        assert 0.0 <= flip <= 1.0, f"{task_name} flip rate {flip} outside [0,1]"


@then('explanation coverage metrics for audit readiness')
def step_check_explanation_coverage(context):
    """Verify explanation coverage is computed"""
    assert 'explanation_coverage' in context.results
    for task_name, coverage in context.results['explanation_coverage'].items():
        assert 0.0 <= coverage <= 1.0, f"{task_name} coverage {coverage} outside [0,1]"


@then('results should be reproducible with fixed seeds')
def step_check_reproducibility(context):
    """Verify results are reproducible"""
    # Run again with same seed
    harness2 = BenchmarkHarness(random_seed=context.seed)
    tasks2 = harness2.load_5_task_suite()
    results2 = harness2.measure_baseline(tasks2)

    # Compare AUC scores
    for task_name in context.results['auc_scores']:
        assert task_name in results2['auc_scores']
        auc1 = context.results['auc_scores'][task_name]
        auc2 = results2['auc_scores'][task_name]
        # Allow small floating point differences
        diff = abs(auc1 - auc2)
        assert diff < 1e-6, f"{task_name} AUC not reproducible: {auc1} vs {auc2}"


@given('baseline scores are recorded')
def step_record_baseline(context):
    """Record baseline scores for comparison"""
    context.harness = BenchmarkHarness(random_seed=context.seed)
    context.tasks = context.harness.load_5_task_suite()
    context.baseline_results = context.harness.measure_baseline(context.tasks)


@when('I define improvement targets')
def step_define_targets(context):
    """Define improvement targets"""
    context.targets = {
        'good': {'auc': 0.75, 'di': 0.80},
        'competitive': {'auc': 0.85, 'di': 0.85},
        'explanation_coverage': 0.80
    }


@then('good performance should be AUC ≥ 0.75 AND DI ≥ 0.80')
def step_check_good_target(context):
    """Verify good performance threshold definition"""
    assert context.targets['good']['auc'] == 0.75
    assert context.targets['good']['di'] == 0.80


@then('competitive performance should be AUC ≥ 0.85 AND DI ≥ 0.85')
def step_check_competitive_target(context):
    """Verify competitive performance threshold definition"""
    assert context.targets['competitive']['auc'] == 0.85
    assert context.targets['competitive']['di'] == 0.85


@then('explanation coverage should be ≥ 80% for practical use')
def step_check_explanation_target(context):
    """Verify explanation coverage target"""
    assert context.targets['explanation_coverage'] == 0.80


@then('all metrics should be JSON-serializable for comparison')
def step_check_json_serializable(context):
    """Verify results can be serialized to JSON"""
    try:
        # Try to serialize the results
        json.dumps(context.baseline_results, default=str)
    except Exception as e:
        raise AssertionError(f"Results not JSON-serializable: {e}") from e


@given('a benchmark suite with fixed random seed 42')
def step_init_with_seed(context):
    """Initialize harness with fixed seed 42"""
    context.harness_1 = BenchmarkHarness(random_seed=42)


@when('I run measurements twice on the same seed')
def step_run_twice(context):
    """Run measurements twice with same seed"""
    tasks_1 = context.harness_1.load_5_task_suite()
    context.results_1 = context.harness_1.measure_baseline(tasks_1)

    # Create new harness with same seed
    harness_2 = BenchmarkHarness(random_seed=42)
    tasks_2 = harness_2.load_5_task_suite()
    context.results_2 = harness_2.measure_baseline(tasks_2)


@then('results should be identical between runs')
def step_check_identical_runs(context):
    """Verify results are identical"""
    assert context.results_1 is not None
    assert context.results_2 is not None


@then('AUC scores should match to 4 decimal places')
def step_check_auc_match(context):
    """Verify AUC scores match"""
    for task_name in context.results_1['auc_scores']:
        auc1 = round(context.results_1['auc_scores'][task_name], 4)
        auc2_raw = context.results_2['auc_scores'][task_name]
        auc2 = round(auc2_raw, 4)
        assert auc1 == auc2, f"{task_name}: {auc1} != {auc2}"


@then('disparate impact ratios should match to 4 decimal places')
def step_check_di_match(context):
    """Verify disparate impact ratios match"""
    for task_name in context.results_1['disparate_impact']:
        di1 = round(context.results_1['disparate_impact'][task_name], 4)
        di2_raw = context.results_2['disparate_impact'][task_name]
        di2 = round(di2_raw, 4)
        assert di1 == di2, f"{task_name}: {di1} != {di2}"


@given('a 5-task benchmark suite is loaded')
def step_load_suite_for_individual(context):
    """Load 5-task benchmark suite"""
    context.harness = BenchmarkHarness(random_seed=context.seed)
    context.tasks = context.harness.load_5_task_suite()


@when('I evaluate each task independently')
def step_evaluate_independently(context):
    """Evaluate each task independently"""
    context.per_task_results = {}
    for task_name, task_data in context.tasks.items():
        result = context.harness.evaluate_task(task_data, task_name)
        context.per_task_results[task_name] = result


@then('I should get per-task metrics for Software Developer')
def step_check_dev_metrics(context):
    """Verify Software Developer metrics"""
    assert 'software_developer' in context.per_task_results
    assert 'auc' in context.per_task_results['software_developer']
    assert 'di' in context.per_task_results['software_developer']


@then('per-task metrics for Financial Analyst')
def step_check_analyst_metrics(context):
    """Verify Financial Analyst metrics"""
    assert 'financial_analyst' in context.per_task_results
    assert 'auc' in context.per_task_results['financial_analyst']


@then('per-task metrics for Healthcare Worker')
def step_check_healthcare_metrics(context):
    """Verify Healthcare Worker metrics"""
    assert 'healthcare_worker' in context.per_task_results
    assert 'auc' in context.per_task_results['healthcare_worker']


@then('per-task metrics for Customer Service')
def step_check_customer_service_metrics(context):
    """Verify Customer Service metrics"""
    assert 'customer_service' in context.per_task_results
    assert 'auc' in context.per_task_results['customer_service']


@then('per-task metrics for Management Role')
def step_check_management_metrics(context):
    """Verify Management Role metrics"""
    assert 'management_role' in context.per_task_results
    assert 'auc' in context.per_task_results['management_role']


@then('each task should report its dominant bias pattern')
def step_check_bias_patterns(context):
    """Verify bias patterns are reported"""
    for task_name, task_result in context.per_task_results.items():
        assert 'dominant_bias' in task_result, f"{task_name}: missing dominant_bias"
        bias = task_result['dominant_bias']
        assert isinstance(bias, str), f"{task_name}: dominant_bias should be string"
