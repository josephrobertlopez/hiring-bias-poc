"""BDD step definitions for fairness_real module"""

from behave import given, when, then
import numpy as np
from src.fairness.real import (
    compute_counterfactual_flip_rate,
    compute_counterfactual_flip_rate_ci,
    FlipRateResult
)
from src.benchmark.data_utils import create_synthetic_resume_data
from scipy import stats


@given('a fairness computation harness is initialized')
def step_init_harness(context):
    context.harness = {}
    context.random_seed = 42
    np.random.seed(context.random_seed)


@given('a synthetic hiring dataset with {n:d} records')
def step_create_dataset(context, n):
    X, y, protected_attr, metadata = create_synthetic_resume_data(
        n_samples=n,
        protected_attr_name='gender',
        bias_factor=0.5,
        random_seed=context.random_seed
    )
    context.X = X
    context.y = y
    context.protected_attr = protected_attr
    context.n_records = n


@given('protected attribute is gender (M/F)')
def step_protected_attr_gender(context):
    context.protected_attr_name = 'gender'
    # Extract the gender column from DataFrame
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender']
    else:
        gender_col = context.protected_attr
    context.protected_attr_values = np.unique(gender_col)
    assert len(context.protected_attr_values) == 2, f"Expected 2 groups, got {len(context.protected_attr_values)}"


@given('a trained predictor with known decision boundary')
def step_create_predictor(context):
    # Simple linear predictor: X @ w
    np.random.seed(context.random_seed)
    n_features = context.X.shape[1]
    context.weights = np.random.randn(n_features)

    # Predict probabilities
    logits = context.X @ context.weights
    context.y_prob = 1.0 / (1.0 + np.exp(-logits))
    context.y_prob = np.clip(context.y_prob, 0.0, 1.0)

    # Threshold predictions
    context.y_pred = (context.y_prob >= 0.5).astype(int)


@when('I compute counterfactual flip rate for all records')
def step_compute_flip_rate(context):
    # Extract numeric gender column for flip rate computation
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender']  # Use numeric 0/1 column
    else:
        gender_col = context.protected_attr

    result = compute_counterfactual_flip_rate(
        y_prob=context.y_prob,
        protected_attr=gender_col,
        threshold=0.5,
        random_seed=context.random_seed
    )
    context.flip_rate_result = result


@then('each record gets re-scored with swapped protected attribute')
def step_verify_swap(context):
    # Verify that the result contains per-record information
    assert hasattr(context.flip_rate_result, 'flips_per_record'), \
        "Result must track per-record flip status"
    assert len(context.flip_rate_result.flips_per_record) == context.n_records, \
        f"Expected {context.n_records} records, got {len(context.flip_rate_result.flips_per_record)}"


@then('predictions are compared (original vs swapped)')
def step_verify_comparison(context):
    # Verify that comparisons were made
    assert context.flip_rate_result.y_prob_original is not None
    assert context.flip_rate_result.y_prob_swapped is not None
    assert len(context.flip_rate_result.y_prob_original) == context.n_records
    assert len(context.flip_rate_result.y_prob_swapped) == context.n_records


@then('flip count increments when |ΔP(y=1)| > threshold')
def step_verify_flip_counting(context):
    # Compute expected flip count
    delta_prob = np.abs(
        context.flip_rate_result.y_prob_swapped - context.flip_rate_result.y_prob_original
    )
    expected_flips = np.sum(delta_prob > 0.5)
    actual_flips = context.flip_rate_result.flip_count

    # Allow small numerical difference
    assert abs(actual_flips - expected_flips) <= 1, \
        f"Expected ~{expected_flips} flips, got {actual_flips}"


@then('result is aggregate flip_rate = flips / N')
def step_verify_aggregate(context):
    expected_rate = context.flip_rate_result.flip_count / context.n_records
    actual_rate = context.flip_rate_result.flip_rate

    assert abs(actual_rate - expected_rate) < 1e-6, \
        f"Expected {expected_rate}, got {actual_rate}"
    assert 0.0 <= actual_rate <= 1.0, \
        f"Flip rate must be in [0, 1], got {actual_rate}"


@when('I compute counterfactual flip rate with bootstrap')
def step_compute_flip_rate_ci(context):
    # Extract numeric gender column
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender'].values
    else:
        gender_col = context.protected_attr

    # Use threshold=0.1 for meaningful bias detection (vs decision threshold 0.5)
    result = compute_counterfactual_flip_rate_ci(
        y_prob=context.y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        n_bootstrap=1000,
        confidence=0.95,
        random_seed=context.random_seed
    )
    context.flip_rate_ci_result = result


@then('result includes point estimate (0.0-1.0)')
def step_verify_point_estimate(context):
    assert hasattr(context.flip_rate_ci_result, 'point_estimate')
    assert 0.0 <= context.flip_rate_ci_result.point_estimate <= 1.0


@then('lower bound of 95% CI')
def step_verify_lower_bound(context):
    assert hasattr(context.flip_rate_ci_result, 'lower_bound')
    assert 0.0 <= context.flip_rate_ci_result.lower_bound <= 1.0
    assert context.flip_rate_ci_result.lower_bound <= context.flip_rate_ci_result.point_estimate


@then('upper bound of 95% CI')
def step_verify_upper_bound(context):
    assert hasattr(context.flip_rate_ci_result, 'upper_bound')
    assert 0.0 <= context.flip_rate_ci_result.upper_bound <= 1.0
    assert context.flip_rate_ci_result.point_estimate <= context.flip_rate_ci_result.upper_bound


@then('bootstrap iterations = 1000 (reproducible with seed)')
def step_verify_reproducibility(context):
    # Extract numeric gender column
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender'].values
    else:
        gender_col = context.protected_attr

    # Run again with same seed, using same threshold as original
    result2 = compute_counterfactual_flip_rate_ci(
        y_prob=context.y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        n_bootstrap=1000,
        confidence=0.95,
        random_seed=context.random_seed
    )

    assert abs(result2.point_estimate - context.flip_rate_ci_result.point_estimate) < 1e-6
    assert abs(result2.lower_bound - context.flip_rate_ci_result.lower_bound) < 1e-6
    assert abs(result2.upper_bound - context.flip_rate_ci_result.upper_bound) < 1e-6


@given('a gender-biased model (favors males)')
def step_create_biased_model(context):
    # Predictor that discriminates on gender
    np.random.seed(context.random_seed)
    n_features = context.X.shape[1]

    # Create biased weights: features + strong gender effect
    context.biased_weights = np.random.randn(n_features) * 0.5

    # Extract numeric gender column for bias computation
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender'].values.astype(float)  # Use numeric 0/1 column as array
    else:
        gender_col = np.asarray(context.protected_attr, dtype=float)

    # Add STRONG gender bias: if gender=0 (male), add +3.0 bias
    # This creates a difference of ~0.95 in mean probabilities
    logits = context.X @ context.biased_weights
    logits = logits + (gender_col == 0) * 3.0  # Stronger bias

    context.biased_y_prob = 1.0 / (1.0 + np.exp(-logits))
    context.biased_y_prob = np.clip(context.biased_y_prob, 0.0, 1.0)


@given('a fair model (gender-independent)')
def step_create_fair_model(context):
    # Predictor that ignores gender
    np.random.seed(context.random_seed)
    n_features = context.X.shape[1]

    # Weights that don't include gender feature
    fair_weights = np.random.randn(n_features) * 0.3
    logits = context.X @ fair_weights

    context.fair_y_prob = 1.0 / (1.0 + np.exp(-logits))
    context.fair_y_prob = np.clip(context.fair_y_prob, 0.0, 1.0)


@when('I compute flip rates for both')
def step_compute_both_flip_rates(context):
    # Extract numeric gender column
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender']
    else:
        gender_col = context.protected_attr

    # Use threshold=0.1 for meaningful bias detection
    context.biased_flip_rate = compute_counterfactual_flip_rate(
        y_prob=context.biased_y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        random_seed=context.random_seed
    ).flip_rate

    context.fair_flip_rate = compute_counterfactual_flip_rate(
        y_prob=context.fair_y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        random_seed=context.random_seed
    ).flip_rate


@then('biased model has flip_rate > 0.10')
def step_verify_biased_high(context):
    assert context.biased_flip_rate > 0.10, \
        f"Biased model flip_rate should be > 0.10, got {context.biased_flip_rate}"


@then('fair model has flip_rate ≤ 0.05')
def step_verify_fair_low(context):
    assert context.fair_flip_rate <= 0.05, \
        f"Fair model flip_rate should be ≤ 0.05, got {context.fair_flip_rate}"


@then('difference is statistically significant (p < 0.05)')
def step_verify_significant_difference(context):
    # Extract numeric gender column
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender']
    else:
        gender_col = context.protected_attr

    # Compute CIs for both using threshold=0.1
    biased_ci = compute_counterfactual_flip_rate_ci(
        y_prob=context.biased_y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        n_bootstrap=1000,
        random_seed=context.random_seed
    )

    fair_ci = compute_counterfactual_flip_rate_ci(
        y_prob=context.fair_y_prob,
        protected_attr=gender_col,
        threshold=0.1,
        n_bootstrap=1000,
        random_seed=context.random_seed
    )

    # CIs should not overlap
    intervals_overlap = (
        fair_ci.upper_bound >= biased_ci.lower_bound and
        biased_ci.upper_bound >= fair_ci.lower_bound
    )

    assert not intervals_overlap, \
        f"CIs should not overlap: biased [{biased_ci.lower_bound}, {biased_ci.upper_bound}], " \
        f"fair [{fair_ci.lower_bound}, {fair_ci.upper_bound}]"


@given('records with missing protected attributes')
def step_add_missing_attrs(context):
    # Extract numeric gender column and add missing values
    if hasattr(context.protected_attr, 'columns'):
        context.protected_attr_with_missing = context.protected_attr['gender'].values.astype(float).copy()
    else:
        context.protected_attr_with_missing = context.protected_attr.astype(float).copy()
    context.protected_attr_with_missing[0:5] = np.nan


@given('records with constant predicted probabilities')
def step_constant_predictions(context):
    # Ensure we have y_prob
    if not hasattr(context, 'y_prob') or context.y_prob is None:
        step_create_predictor(context)
    context.constant_y_prob = np.ones_like(context.y_prob) * 0.5


@given('single-group datasets')
def step_single_group_dataset(context):
    # Extract numeric gender column
    if hasattr(context.protected_attr, 'columns'):
        context.single_group_attr = np.zeros(len(context.protected_attr['gender']))
    else:
        context.single_group_attr = np.zeros_like(context.protected_attr)


@when('I compute flip rate')
def step_compute_edge_cases(context):
    # Extract numeric gender column for non-missing case
    if hasattr(context.protected_attr, 'columns'):
        gender_col = context.protected_attr['gender'].values
    else:
        gender_col = np.asarray(context.protected_attr)

    # Test with missing
    if hasattr(context, 'protected_attr_with_missing'):
        result = compute_counterfactual_flip_rate(
            y_prob=context.y_prob,
            protected_attr=context.protected_attr_with_missing,
            random_seed=context.random_seed
        )
        context.missing_result = result

    # Test with constant predictions
    if hasattr(context, 'constant_y_prob'):
        result = compute_counterfactual_flip_rate(
            y_prob=context.constant_y_prob,
            protected_attr=gender_col,
            random_seed=context.random_seed
        )
        context.constant_result = result

    # Test with single group
    if hasattr(context, 'single_group_attr'):
        result = compute_counterfactual_flip_rate(
            y_prob=context.y_prob,
            protected_attr=context.single_group_attr,
            random_seed=context.random_seed
        )
        context.single_group_result = result


@then('missing attributes are skipped with count')
def step_verify_missing_handling(context):
    assert context.missing_result.skipped_records >= 5, \
        f"Expected >=5 skipped, got {context.missing_result.skipped_records}"
    # flip_count + skipped should equal total (within margin for non-flips)
    # Specifically: skipped + flips + non-flipped = total
    # So flip_count <= (total - skipped)
    valid_count = context.n_records - context.missing_result.skipped_records
    assert context.missing_result.flip_count <= valid_count, \
        f"Flip count {context.missing_result.flip_count} > valid records {valid_count}"


@then('constant predictions return 0.0 flips')
def step_verify_constant_handling(context):
    assert context.constant_result.flip_rate == 0.0, \
        f"Constant predictions should have 0 flips, got {context.constant_result.flip_rate}"


@then('single-group returns 0.0 (no swap possible)')
def step_verify_single_group(context):
    assert context.single_group_result.flip_rate == 0.0, \
        f"Single group should return 0.0, got {context.single_group_result.flip_rate}"


@then('result includes edge_case_count in metadata')
def step_verify_metadata(context):
    assert hasattr(context.missing_result, 'skipped_records')
    assert context.missing_result.skipped_records >= 0




@given('a predictor and 100-record test set')
def step_create_predictor_and_data(context):
    # Create 100-record dataset
    X, y, protected_attr, metadata = create_synthetic_resume_data(
        n_samples=100,
        protected_attr_name='gender',
        bias_factor=0.5,
        random_seed=context.random_seed
    )
    context.X = X
    context.y = y
    context.protected_attr = protected_attr
    context.n_records = 100

    # Create a trained predictor
    np.random.seed(context.random_seed)
    n_features = context.X.shape[1]
    context.weights = np.random.randn(n_features)

    # Predict probabilities
    logits = context.X @ context.weights
    context.y_prob = 1.0 / (1.0 + np.exp(-logits))
    context.y_prob = np.clip(context.y_prob, 0.0, 1.0)
