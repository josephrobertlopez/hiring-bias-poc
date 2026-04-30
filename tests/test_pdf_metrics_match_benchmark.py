"""Test that PDF fairness metrics match benchmark.json exactly.

Prevents regression where someone hardcodes fabricated metric values.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.demo.components.pdf_renderer import generate_fairness_audit_pdf


def test_pdf_metrics_match_benchmark():
    """Test fairness audit PDF contains only real benchmark.json values."""
    # Load benchmark.json
    benchmark_path = Path("benchmark.json")
    with open(benchmark_path) as f:
        bench = json.load(f)

    # Generate fairness audit PDF
    pdf_buffer = generate_fairness_audit_pdf("All Decisions")

    # Extract PDF text via pdftotext
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_buffer.getvalue())
        temp_pdf.flush()

        result = subprocess.run(
            ["pdftotext", temp_pdf.name, "-"],
            capture_output=True,
            text=True,
            check=True
        )
        pdf_text = result.stdout

    # Extract expected values from benchmark.json
    expected_gender_di = bench['fairness_metrics']['gender']['disparate_impact']['value']
    expected_race_di = bench['fairness_metrics']['race']['disparate_impact']['value']
    expected_gender_eo = bench['fairness_metrics']['gender']['equalized_odds_gap']['value']
    expected_race_eo = bench['fairness_metrics']['race']['equalized_odds_gap']['value']
    expected_gender_ece = bench['fairness_metrics']['gender']['calibration_ece']['value']
    expected_race_ece = bench['fairness_metrics']['race']['calibration_ece']['value']

    # Test 1: Assert fabricated values do NOT appear
    fabricated_values = ["0.912", "0.867", "0.798"]
    for fabricated in fabricated_values:
        assert fabricated not in pdf_text, f"Fabricated value {fabricated} found in PDF"

    # Test 2: Assert fabricated rows do NOT appear
    fabricated_rows = ["Female × Asian", "Male × Black", "Age:"]
    for fabricated_row in fabricated_rows:
        assert fabricated_row not in pdf_text, f"Fabricated row '{fabricated_row}' found in PDF"

    # Test 3: Assert real benchmark values DO appear with correct precision
    # Format expected values to 3 decimal places (same as PDF)
    expected_values = {
        f"{expected_gender_di:.3f}": "Gender disparate impact",
        f"{expected_race_di:.3f}": "Race disparate impact",
        f"{expected_gender_eo:.3f}": "Gender equalized odds gap",
        f"{expected_race_eo:.3f}": "Race equalized odds gap",
        f"{expected_gender_ece:.3f}": "Gender calibration ECE",
        f"{expected_race_ece:.3f}": "Race calibration ECE"
    }

    values_found = 0
    for expected_value, description in expected_values.items():
        if expected_value in pdf_text:
            values_found += 1
        else:
            pytest.fail(f"Expected {description} value {expected_value} not found in PDF")

    # Test 4: Assert we found at least 3 metric values (requirement to prevent empty PDF)
    assert values_found >= 3, f"Only found {values_found} metric values, expected at least 3"

    # Test 5: Assert intersectional analysis disclaimer is present
    assert "Intersectional analysis: not computed in PoC v1" in pdf_text, \
        "Missing intersectional analysis disclaimer"

    # Test 6: Assert proper failure status for failed metrics
    # From benchmark.json, equalized_odds_gap and calibration_ece should show FAIL
    gender_eo_passed = bench['fairness_metrics']['gender']['equalized_odds_gap']['passed']
    race_eo_passed = bench['fairness_metrics']['race']['equalized_odds_gap']['passed']
    gender_ece_passed = bench['fairness_metrics']['gender']['calibration_ece']['passed']
    race_ece_passed = bench['fairness_metrics']['race']['calibration_ece']['passed']

    if not (gender_eo_passed and race_eo_passed):
        # Should show FAIL for equalized odds if either gender or race failed
        fail_pattern = r"Equalized Odds Gap.*❌ FAIL|Equalized Odds Gap.*FAIL"
        assert re.search(fail_pattern, pdf_text, re.DOTALL), \
            "Equalized Odds Gap should show FAIL status but doesn't"

    if not (gender_ece_passed and race_ece_passed):
        # Should show FAIL for calibration ECE if either gender or race failed
        fail_pattern = r"Calibration ECE.*❌ FAIL|Calibration ECE.*FAIL"
        assert re.search(fail_pattern, pdf_text, re.DOTALL), \
            "Calibration ECE should show FAIL status but doesn't"


def test_benchmark_json_exists():
    """Sanity check that benchmark.json exists and has expected structure."""
    benchmark_path = Path("benchmark.json")
    assert benchmark_path.exists(), "benchmark.json not found"

    with open(benchmark_path) as f:
        bench = json.load(f)

    # Verify expected structure
    assert 'fairness_metrics' in bench
    assert 'gender' in bench['fairness_metrics']
    assert 'race' in bench['fairness_metrics']

    for attr in ['gender', 'race']:
        metrics = bench['fairness_metrics'][attr]
        for metric in ['disparate_impact', 'equalized_odds_gap', 'calibration_ece']:
            assert metric in metrics, f"Missing {metric} for {attr}"
            assert 'value' in metrics[metric], f"Missing value for {attr}.{metric}"
            assert 'passed' in metrics[metric], f"Missing passed for {attr}.{metric}"