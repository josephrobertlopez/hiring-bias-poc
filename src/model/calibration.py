"""Isotonic calibration and Expected Calibration Error (ECE) calculation."""

from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import warnings


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece_before: float  # Expected Calibration Error before calibration
    ece_after: float   # Expected Calibration Error after calibration
    brier_score_before: float  # Brier score before calibration
    brier_score_after: float   # Brier score after calibration
    reliability_diagram: Dict[str, List[float]]  # For plotting
    n_calibration_samples: int


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    calibrated_probabilities: np.ndarray
    metrics: CalibrationMetrics
    isotonic_regressor: IsotonicRegression


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability calibration.

    Applies isotonic regression on a held-out fold to improve
    probability calibration and reduce Expected Calibration Error (ECE).
    """

    def __init__(self, n_bins: int = 10, random_state: int = 42):
        """Initialize calibrator.

        Args:
            n_bins: Number of bins for ECE calculation
            random_state: Random seed for train/calibration split
        """
        self.n_bins = n_bins
        self.random_state = random_state
        self.isotonic_regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.fitted = False

    def fit_and_calibrate(self,
                         raw_probabilities: np.ndarray,
                         true_labels: np.ndarray,
                         validation_size: float = 0.3) -> CalibrationResult:
        """Fit isotonic calibrator and calibrate probabilities.

        Args:
            raw_probabilities: Uncalibrated probabilities from model
            true_labels: True binary labels
            validation_size: Fraction of data to use for calibration

        Returns:
            CalibrationResult with calibrated probabilities and metrics
        """
        # Split data into model predictions and calibration set
        probs_train, probs_cal, labels_train, labels_cal = train_test_split(
            raw_probabilities, true_labels,
            test_size=validation_size,
            random_state=self.random_state,
            stratify=true_labels
        )

        # Calculate ECE before calibration
        ece_before = self.calculate_ece(probs_cal, labels_cal)
        brier_before = self.calculate_brier_score(probs_cal, labels_cal)

        # Fit isotonic regressor on training probabilities
        self.isotonic_regressor.fit(probs_train, labels_train)
        self.fitted = True

        # Calibrate validation probabilities
        calibrated_probs = self.isotonic_regressor.transform(probs_cal)

        # Calculate ECE after calibration
        ece_after = self.calculate_ece(calibrated_probs, labels_cal)
        brier_after = self.calculate_brier_score(calibrated_probs, labels_cal)

        # Generate reliability diagram data
        reliability_diagram = self._generate_reliability_diagram(
            probs_cal, calibrated_probs, labels_cal
        )

        metrics = CalibrationMetrics(
            ece_before=ece_before,
            ece_after=ece_after,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            reliability_diagram=reliability_diagram,
            n_calibration_samples=len(labels_cal)
        )

        return CalibrationResult(
            calibrated_probabilities=calibrated_probs,
            metrics=metrics,
            isotonic_regressor=self.isotonic_regressor
        )

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply fitted calibrator to new probabilities.

        Args:
            probabilities: Raw probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")

        return self.isotonic_regressor.transform(probabilities)

    def calculate_ece(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE).

        Args:
            probabilities: Predicted probabilities [0, 1]
            true_labels: True binary labels {0, 1}

        Returns:
            ECE value (lower is better, 0 = perfect calibration)
        """
        if len(probabilities) == 0:
            return 0.0

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = true_labels[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = probabilities[in_bin].mean()
                # ECE contribution from this bin
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def calculate_brier_score(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Brier score (mean squared error of probabilities).

        Args:
            probabilities: Predicted probabilities [0, 1]
            true_labels: True binary labels {0, 1}

        Returns:
            Brier score (lower is better, 0 = perfect)
        """
        return np.mean((probabilities - true_labels) ** 2)

    def _generate_reliability_diagram(self,
                                    raw_probs: np.ndarray,
                                    calibrated_probs: np.ndarray,
                                    true_labels: np.ndarray) -> Dict[str, List[float]]:
        """Generate data for reliability diagram plotting.

        Args:
            raw_probs: Raw probabilities before calibration
            calibrated_probs: Calibrated probabilities
            true_labels: True labels

        Returns:
            Dictionary with diagram data
        """
        diagram_data = {
            'bin_centers': [],
            'raw_accuracies': [],
            'calibrated_accuracies': [],
            'bin_counts': [],
            'perfect_calibration': []
        }

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            # Find samples in this bin (using raw probabilities for binning)
            in_bin = (raw_probs > bin_lower) & (raw_probs <= bin_upper)

            if in_bin.sum() > 0:
                # Raw accuracy
                raw_accuracy = true_labels[in_bin].mean()

                # Calibrated accuracy (bin by calibrated probabilities)
                cal_in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
                if cal_in_bin.sum() > 0:
                    cal_accuracy = true_labels[cal_in_bin].mean()
                else:
                    cal_accuracy = raw_accuracy  # Fallback

                diagram_data['bin_centers'].append(bin_centers[i])
                diagram_data['raw_accuracies'].append(raw_accuracy)
                diagram_data['calibrated_accuracies'].append(cal_accuracy)
                diagram_data['bin_counts'].append(int(in_bin.sum()))
                diagram_data['perfect_calibration'].append(bin_centers[i])

        return diagram_data

    def evaluate_calibration_quality(self,
                                   probabilities: np.ndarray,
                                   true_labels: np.ndarray) -> Dict[str, float]:
        """Comprehensive calibration quality evaluation.

        Args:
            probabilities: Predicted probabilities
            true_labels: True labels

        Returns:
            Dictionary of calibration metrics
        """
        ece = self.calculate_ece(probabilities, true_labels)
        brier_score = self.calculate_brier_score(probabilities, true_labels)

        # Maximum Calibration Error (MCE) - worst bin
        mce = self._calculate_mce(probabilities, true_labels)

        # Average Calibration Error (ACE) - average of absolute deviations
        ace = self._calculate_ace(probabilities, true_labels)

        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'average_calibration_error': ace,
            'brier_score': brier_score,
            'log_loss': self._calculate_log_loss(probabilities, true_labels)
        }

    def _calculate_mce(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        max_error = 0

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                confidence_in_bin = probabilities[in_bin].mean()
                error = abs(confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)

        return max_error

    def _calculate_ace(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Average Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        errors = []

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                confidence_in_bin = probabilities[in_bin].mean()
                error = abs(confidence_in_bin - accuracy_in_bin)
                errors.append(error)

        return np.mean(errors) if errors else 0.0

    def _calculate_log_loss(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate logarithmic loss (cross-entropy)."""
        # Clip probabilities to avoid log(0)
        epsilon = 1e-15
        clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)

        return -np.mean(
            true_labels * np.log(clipped_probs) +
            (1 - true_labels) * np.log(1 - clipped_probs)
        )


def apply_full_calibration_pipeline(raw_probabilities: np.ndarray,
                                  true_labels: np.ndarray,
                                  validation_size: float = 0.3,
                                  n_bins: int = 10,
                                  random_state: int = 42) -> Tuple[np.ndarray, CalibrationMetrics]:
    """Apply complete calibration pipeline with evaluation.

    Args:
        raw_probabilities: Uncalibrated model probabilities
        true_labels: True binary labels
        validation_size: Fraction for calibration fold
        n_bins: Number of bins for ECE calculation
        random_state: Random seed

    Returns:
        (calibrated_probabilities, metrics)
    """
    calibrator = IsotonicCalibrator(n_bins=n_bins, random_state=random_state)
    result = calibrator.fit_and_calibrate(raw_probabilities, true_labels, validation_size)

    return result.calibrated_probabilities, result.metrics


def check_calibration_gates(ece: float,
                           brier_score: float,
                           ece_threshold: float = 0.05,
                           brier_threshold: float = 0.25) -> Dict[str, bool]:
    """Check if calibration meets quality gates.

    Args:
        ece: Expected Calibration Error
        brier_score: Brier score
        ece_threshold: Maximum acceptable ECE
        brier_threshold: Maximum acceptable Brier score

    Returns:
        Dictionary of gate results
    """
    return {
        'ece_gate_passed': ece <= ece_threshold,
        'brier_gate_passed': brier_score <= brier_threshold,
        'overall_calibration_passed': ece <= ece_threshold and brier_score <= brier_threshold
    }