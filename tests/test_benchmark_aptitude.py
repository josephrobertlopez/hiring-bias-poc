"""Tests for aptitude integration in benchmark pipeline.

Verifies that benchmark runs with and without aptitude scoring,
and that aptitude_summary is well-formed when enabled.
"""

import json
import tempfile
import os
import pytest
from src.benchmarks.kaggle_eval import KaggleBenchmarkEvaluator


def test_benchmark_without_aptitude():
    """Benchmark runs normally without aptitude scoring."""
    evaluator = KaggleBenchmarkEvaluator(random_state=42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        # Run benchmark without aptitude
        result = evaluator.run_benchmark(
            dataset_path=None,  # Use synthetic dataset
            output_path=output_path,
            with_aptitude=False
        )

        # Verify result structure without aptitude
        assert result.aptitude_summary is None
        assert hasattr(result, 'model_performance')
        assert hasattr(result, 'fairness_metrics')

        # Verify saved file doesn't have aptitude_summary or it's null
        with open(output_path, 'r') as f:
            saved_result = json.load(f)

        assert saved_result.get('aptitude_summary') is None

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_benchmark_with_aptitude():
    """Benchmark includes aptitude scoring when requested."""
    evaluator = KaggleBenchmarkEvaluator(random_state=42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        # Run benchmark with aptitude
        result = evaluator.run_benchmark(
            dataset_path=None,  # Use synthetic dataset
            output_path=output_path,
            with_aptitude=True
        )

        # Verify aptitude_summary exists and is well-formed
        assert result.aptitude_summary is not None
        aptitude = result.aptitude_summary

        # Check required fields
        assert 'n_scored_candidates' in aptitude
        assert 'n_skills_covered' in aptitude
        assert 'per_skill_stats' in aptitude
        assert 'recommendation_distribution' in aptitude
        assert 'mean_overall_uncertainty_width' in aptitude
        assert 'model_version' in aptitude

        # Verify data types and ranges
        assert isinstance(aptitude['n_scored_candidates'], int)
        assert aptitude['n_scored_candidates'] > 0
        assert isinstance(aptitude['n_skills_covered'], int)
        assert aptitude['n_skills_covered'] > 0
        assert isinstance(aptitude['per_skill_stats'], dict)
        assert isinstance(aptitude['recommendation_distribution'], dict)
        assert isinstance(aptitude['mean_overall_uncertainty_width'], float)
        assert 0.0 <= aptitude['mean_overall_uncertainty_width'] <= 1.0

        # Verify per-skill stats structure
        for skill, stats in aptitude['per_skill_stats'].items():
            assert 'mean_score' in stats
            assert 'std_score' in stats
            assert 'mean_uncertainty_width' in stats
            assert 'n_candidates' in stats
            assert 0.0 <= stats['mean_score'] <= 1.0
            assert stats['std_score'] >= 0.0
            assert stats['mean_uncertainty_width'] >= 0.0
            assert stats['n_candidates'] > 0

        # Verify recommendation distribution sums to 1.0 (approximately)
        total_prob = sum(aptitude['recommendation_distribution'].values())
        assert abs(total_prob - 1.0) < 0.01

        # Verify saved file has aptitude_summary
        with open(output_path, 'r') as f:
            saved_result = json.load(f)

        assert saved_result['aptitude_summary'] is not None
        assert saved_result['aptitude_summary']['n_scored_candidates'] > 0

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_aptitude_deterministic():
    """Aptitude scoring produces deterministic results."""
    evaluator1 = KaggleBenchmarkEvaluator(random_state=42)
    evaluator2 = KaggleBenchmarkEvaluator(random_state=42)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        output_path1 = f1.name
        output_path2 = f2.name

    try:
        # Run same benchmark twice with same seed
        result1 = evaluator1.run_benchmark(
            dataset_path=None,
            output_path=output_path1,
            with_aptitude=True
        )

        result2 = evaluator2.run_benchmark(
            dataset_path=None,
            output_path=output_path2,
            with_aptitude=True
        )

        # Verify deterministic aptitude results
        apt1 = result1.aptitude_summary
        apt2 = result2.aptitude_summary

        assert apt1['n_scored_candidates'] == apt2['n_scored_candidates']
        assert apt1['n_skills_covered'] == apt2['n_skills_covered']
        assert abs(apt1['mean_overall_uncertainty_width'] - apt2['mean_overall_uncertainty_width']) < 1e-6

        # Compare per-skill stats
        assert set(apt1['per_skill_stats'].keys()) == set(apt2['per_skill_stats'].keys())
        for skill in apt1['per_skill_stats']:
            stats1 = apt1['per_skill_stats'][skill]
            stats2 = apt2['per_skill_stats'][skill]
            assert abs(stats1['mean_score'] - stats2['mean_score']) < 1e-6
            assert abs(stats1['mean_uncertainty_width'] - stats2['mean_uncertainty_width']) < 1e-6

    finally:
        for path in [output_path1, output_path2]:
            if os.path.exists(path):
                os.unlink(path)


def test_aptitude_small_dataset():
    """Aptitude scoring works with very small datasets."""
    # This tests edge case where we might have fewer test samples than the default 100
    evaluator = KaggleBenchmarkEvaluator(random_state=42)

    # Mock a very small dataset by overriding the synthetic dataset generation
    original_method = evaluator._create_synthetic_dataset

    def small_dataset():
        # Generate only 20 samples total (4 test samples after 80/20 split)
        info, resumes, labels, demographics = original_method()
        n_small = 20
        return (
            {**info, 'n_samples': n_small},
            resumes[:n_small],
            labels[:n_small],
            {k: v[:n_small] for k, v in demographics.items()}
        )

    evaluator._create_synthetic_dataset = small_dataset

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        result = evaluator.run_benchmark(
            dataset_path=None,
            output_path=output_path,
            with_aptitude=True
        )

        # Should still have aptitude summary, just with fewer candidates
        assert result.aptitude_summary is not None
        assert result.aptitude_summary['n_scored_candidates'] <= 4  # 20% of 20
        assert result.aptitude_summary['n_skills_covered'] > 0

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
        # Restore original method
        evaluator._create_synthetic_dataset = original_method