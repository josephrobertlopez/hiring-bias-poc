"""Unit tests for bias utilities."""
import pytest
from src.rules.bias_utils import compute_disparity_index


class TestComputeDisparityIndex:
    """Test disparity index computation."""

    def test_perfect_parity(self):
        """Test that equal rates yield DI = 1.0."""
        result = compute_disparity_index([0.5, 0.5, 0.5])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    def test_extreme_disparity(self):
        """Test that 100% vs 0% yields DI = 0.0."""
        result = compute_disparity_index([1.0, 0.0])

        assert result["disparity_index"] == 0.0
        assert result["bias_detected"]

    def test_threshold_boundary_below(self):
        """Test that DI just below 0.8 threshold triggers bias detection."""
        result = compute_disparity_index([0.79, 1.0])

        assert result["disparity_index"] == 0.79
        assert result["bias_detected"]

    def test_threshold_boundary_above(self):
        """Test that DI at or above 0.8 threshold doesn't trigger bias."""
        result = compute_disparity_index([0.80, 1.0])

        assert result["disparity_index"] == 0.80
        assert not result["bias_detected"]

    def test_min_max_rates(self):
        """Test that min and max rates are correctly identified."""
        result = compute_disparity_index([0.3, 0.7, 0.5])

        assert result["min_rate"] == 0.3
        assert result["max_rate"] == 0.7

    def test_all_zeros(self):
        """Test handling of all zero hiring rates."""
        result = compute_disparity_index([0.0, 0.0, 0.0])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    def test_single_rate(self):
        """Test handling single hiring rate (not enough for comparison)."""
        result = compute_disparity_index([0.5])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    def test_empty_list(self):
        """Test handling empty hiring rate list."""
        result = compute_disparity_index([])

        assert result["disparity_index"] == 1.0
        assert not result["bias_detected"]

    def test_custom_threshold(self):
        """Test using custom threshold."""
        result = compute_disparity_index([0.85, 1.0], threshold=0.9)

        # DI = 0.85, custom threshold = 0.9, so should trigger bias
        assert result["disparity_index"] == 0.85
        assert result["bias_detected"]

    def test_four_fifth_rule(self):
        """Test the standard EEOC 4/5 rule."""
        # 80% hiring rate for group A, 100% for group B
        # DI = 80% / 100% = 0.8, threshold = 0.8 (standard)
        result = compute_disparity_index([0.8, 1.0])

        assert result["disparity_index"] == 0.8
        # At threshold, should NOT be detected as bias (>= threshold)
        assert not result["bias_detected"]

    def test_multiple_groups(self):
        """Test DI with many demographic groups."""
        # Varying hiring rates across multiple groups
        result = compute_disparity_index([0.2, 0.4, 0.6, 0.8, 1.0])

        # DI = min / max = 0.2 / 1.0 = 0.2
        assert result["disparity_index"] == 0.2
        assert result["bias_detected"]

    def test_return_type(self):
        """Test that return value has correct structure."""
        result = compute_disparity_index([0.5, 0.8])

        assert isinstance(result, dict)
        assert set(result.keys()) == {"disparity_index", "bias_detected", "min_rate", "max_rate"}
        assert isinstance(result["disparity_index"], float)
        assert isinstance(result["bias_detected"], bool)
        assert isinstance(result["min_rate"], float)
        assert isinstance(result["max_rate"], float)

    def test_float_precision(self):
        """Test that float values are handled with precision."""
        result = compute_disparity_index([0.333333, 0.666667])

        # DI should be approximately 0.5
        assert abs(result["disparity_index"] - 0.5) < 0.001
        assert result["bias_detected"]  # 0.5 < 0.8
