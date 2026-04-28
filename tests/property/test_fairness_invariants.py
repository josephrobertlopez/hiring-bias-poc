"""Property-based tests for fairness invariants using hypothesis."""
import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from src.rules.bias_utils import compute_disparity_index


class TestFairnessInvariants:
    """Property-based tests for mathematical fairness invariants."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_disparity_index_in_bounds(self, hiring_rates):
        """Property: DI must always be in [0, 1]."""
        assume(all(0.0 <= rate <= 1.0 for rate in hiring_rates))

        result = compute_disparity_index(hiring_rates)
        di = result["disparity_index"]

        assert 0.0 <= di <= 1.0

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_equal_rates_perfect_parity(self, rate):
        """Property: Equal hiring rates yield DI = 1.0."""
        rates = [rate, rate, rate]
        result = compute_disparity_index(rates)

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_min_max_correctness(self, hiring_rates):
        """Property: min_rate and max_rate are correct."""
        assume(len(hiring_rates) >= 2)

        result = compute_disparity_index(hiring_rates)

        assert result["min_rate"] == min(hiring_rates)
        assert result["max_rate"] == max(hiring_rates)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_di_formula_correctness(self, hiring_rates):
        """Property: DI = min_rate / max_rate (when max > 0)."""
        assume(len(hiring_rates) >= 2)
        assume(max(hiring_rates) > 0)

        result = compute_disparity_index(hiring_rates)

        min_rate = min(hiring_rates)
        max_rate = max(hiring_rates)
        expected_di = min_rate / max_rate

        assert abs(result["disparity_index"] - expected_di) < 1e-10

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_bias_threshold_consistency(self, hiring_rates):
        """Property: bias_detected iff DI < 0.8."""
        assume(len(hiring_rates) >= 2)

        result = compute_disparity_index(hiring_rates, threshold=0.8)

        expected_bias = result["disparity_index"] < 0.8
        assert result["bias_detected"] == expected_bias

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_two_group_di_monotonic(self, rate_a, rate_b):
        """Property: As one rate increases, DI increases (or stays same)."""
        # Avoid floating point precision issues near zero
        assume(rate_a >= 1e-6 and rate_b >= 1e-6)
        assume(abs(rate_a - rate_b) >= 1e-6)

        # Ensure the increase doesn't flip the min/max relationship
        if rate_a < rate_b:
            # If rate_a is smaller, ensure increase doesn't make it larger than rate_b
            increase = min(0.01, rate_b - rate_a - 1e-6)
            assume(increase > 0)  # Skip if no valid increase possible

            result1 = compute_disparity_index([rate_a, rate_b])

            rate_a_increased = rate_a + increase
            result2 = compute_disparity_index([rate_a_increased, rate_b])

            # Increasing min should increase or maintain DI when min/max order preserved
            assert result2["disparity_index"] >= result1["disparity_index"]

    def test_extreme_bias_detection(self):
        """Property: 0% vs 100% always detected as bias."""
        result = compute_disparity_index([0.0, 1.0])

        assert result["disparity_index"] == 0.0
        assert result["bias_detected"]

    def test_just_under_threshold_detected(self):
        """Property: DI at 0.799999 detected as bias (< 0.8)."""
        result = compute_disparity_index([0.799999, 1.0])

        assert result["bias_detected"]

    def test_at_threshold_not_detected(self):
        """Property: DI at 0.8 NOT detected as bias (>= 0.8)."""
        result = compute_disparity_index([0.8, 1.0])

        assert not result["bias_detected"]

    def test_above_threshold_not_detected(self):
        """Property: DI > 0.8 NOT detected as bias."""
        result = compute_disparity_index([0.85, 1.0])

        assert not result["bias_detected"]


class TestDisparityIndexMonotonicity:
    """Tests for monotonicity properties."""

    @given(
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=5)
    )
    def test_scale_invariance(self, rates):
        """Property: DI unchanged when all rates scaled by constant."""
        assume(len(rates) >= 2)
        result1 = compute_disparity_index(rates)

        # Scale all rates by 0.5
        scaled_rates = [min(r * 0.5, 1.0) for r in rates]
        result2 = compute_disparity_index(scaled_rates)

        # DI should be approximately the same
        # (may differ slightly due to clamping at 1.0)
        if max(scaled_rates) < 1.0:
            assert abs(result1["disparity_index"] - result2["disparity_index"]) < 0.01

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_symmetric_in_rates(self, hiring_rates):
        """Property: DI depends only on min/max, not on order."""
        assume(len(hiring_rates) >= 2)

        result1 = compute_disparity_index(hiring_rates)

        # Reverse and shuffle
        shuffled = list(reversed(hiring_rates))
        result2 = compute_disparity_index(shuffled)

        assert result1["disparity_index"] == result2["disparity_index"]

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_all_zero_rates_perfect_parity(self, unused):
        """Property: All zero rates yield no bias."""
        result = compute_disparity_index([0.0, 0.0, 0.0])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]


class TestDisparityIndexEdgeCases:
    """Edge case properties."""

    def test_single_rate_no_comparison(self):
        """Property: Single rate cannot show bias (need at least 2 groups)."""
        result = compute_disparity_index([0.5])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    def test_empty_rates_no_bias(self):
        """Property: Empty list yields no bias."""
        result = compute_disparity_index([])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_duplicate_rates_perfect_parity(self, rate):
        """Property: Duplicates always yield perfect parity."""
        result = compute_disparity_index([rate, rate, rate, rate])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_return_dict_always_has_keys(self, hiring_rates):
        """Property: Return dict always has all required keys."""
        result = compute_disparity_index(hiring_rates)

        required_keys = {"disparity_index", "bias_detected", "min_rate", "max_rate"}
        assert set(result.keys()) == required_keys

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    def test_all_values_numeric(self, hiring_rates):
        """Property: All result values are proper numeric types."""
        result = compute_disparity_index(hiring_rates)

        assert isinstance(result["disparity_index"], (int, float))
        assert isinstance(result["bias_detected"], bool)
        assert isinstance(result["min_rate"], (int, float))
        assert isinstance(result["max_rate"], (int, float))


class TestCustomThresholdProperties:
    """Properties with custom thresholds."""

    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=5),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_threshold_behavior(self, hiring_rates, threshold):
        """Property: bias_detected iff DI < threshold."""
        assume(len(hiring_rates) >= 2)

        result = compute_disparity_index(hiring_rates, threshold=threshold)

        expected_bias = result["disparity_index"] < threshold
        assert result["bias_detected"] == expected_bias

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_threshold_zero_all_bias(self, rate):
        """Property: threshold=0 means bias detected when DI < 0, which never happens."""
        rates = [rate, 1.0]
        result = compute_disparity_index(rates, threshold=0.0)

        # Since DI is always >= 0, and threshold=0 means DI < 0 for bias,
        # bias should never be detected with threshold=0
        assert not result["bias_detected"]

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_threshold_one_no_bias(self, rate):
        """Property: threshold=1.0 means bias detected when DI < 1.0."""
        rates = [rate, 1.0]
        result = compute_disparity_index(rates, threshold=1.0)

        # Bias detected when DI < 1.0, which happens when rate < 1.0
        expected_bias = (rate < 1.0)
        assert result["bias_detected"] == expected_bias
