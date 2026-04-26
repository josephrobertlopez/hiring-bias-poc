"""Benchmark harness for running practical bias detection evaluations"""

from typing import Dict, Any
import numpy as np
from .tasks import create_5_task_suite
from .metrics import compute_metrics


class BenchmarkHarness:
    """Main benchmark harness for hiring bias detection POC"""

    def __init__(self, random_seed: int = 42):
        """
        Initialize the benchmark harness

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def load_5_task_suite(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the 5-task hiring benchmark suite

        Returns:
            Dict mapping task names to task data
                Each task: X (features), y (labels), protected_attr (info)
        """
        tasks = create_5_task_suite(random_seed=self.random_seed)
        return tasks

    def measure_baseline(self, tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure baseline metrics on all tasks

        Args:
            tasks: Dict of task data from load_5_task_suite()

        Returns:
            Dict with aggregated metrics:
                - auc_scores: AUC per task
                - disparate_impact: DI per task
                - flip_rates: Flip rate per task
                - explanation_coverage: Coverage per task
                - avg_metrics: Aggregate stats
        """
        results = {
            'auc_scores': {},
            'disparate_impact': {},
            'flip_rates': {},
            'explanation_coverage': {},
            'per_task': {}
        }

        for task_name, task_data in tasks.items():
            X = task_data['X']
            y = task_data['y']
            protected_attr = task_data['protected_attr_values']
            attr_name = task_data['protected_attr']['name']

            # Compute metrics for this task
            metrics = compute_metrics(
                X, y, protected_attr,
                protected_attr_name=attr_name,
                random_seed=self.random_seed + hash(task_name) % 1000
            )

            # Store individual metrics
            results['auc_scores'][task_name] = metrics['auc']
            results['disparate_impact'][task_name] = metrics['disparate_impact']
            results['flip_rates'][task_name] = metrics['flip_rate']
            results['explanation_coverage'][task_name] = metrics['explanation_coverage']
            results['per_task'][task_name] = metrics

        # Compute aggregate statistics
        auc_scores = list(results['auc_scores'].values())
        di_scores = list(results['disparate_impact'].values())
        flip_scores = list(results['flip_rates'].values())
        cov_scores = list(results['explanation_coverage'].values())

        results['avg_metrics'] = {
            'avg_auc': float(np.mean(auc_scores)),
            'avg_disparate_impact': float(np.mean(di_scores)),
            'avg_flip_rate': float(np.mean(flip_scores)),
            'avg_explanation_coverage': float(np.mean(cov_scores))
        }

        return results

    def evaluate_task(
        self, task_data: Dict[str, Any], task_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single task independently

        Args:
            task_data: Task data dict
            task_name: Name of task (for bias pattern mapping)

        Returns:
            Dict with task metrics and dominant bias pattern
        """
        X = task_data['X']
        y = task_data['y']
        protected_attr = task_data['protected_attr_values']
        attr_name = task_data['protected_attr']['name']

        metrics = compute_metrics(
            X, y, protected_attr,
            protected_attr_name=attr_name,
            random_seed=self.random_seed + hash(task_name) % 1000
        )

        # Identify dominant bias pattern
        dominant_bias = self._identify_dominant_bias(
            task_name,
            metrics,
            attr_name
        )

        result = {
            'task_name': task_name,
            'auc': metrics['auc'],
            'di': metrics['disparate_impact'],
            'flip_rate': metrics['flip_rate'],
            'explanation_coverage': metrics['explanation_coverage'],
            'dominant_bias': dominant_bias,
            'feature_importance': metrics['feature_importance'],
            'protected_attr': attr_name
        }

        return result

    def _identify_dominant_bias(
        self,
        task_name: str,
        metrics: Dict[str, float],
        attr_name: str
    ) -> str:
        """
        Identify the dominant bias pattern in a task

        Args:
            task_name: Name of the task
            metrics: Computed metrics
            attr_name: Protected attribute name

        Returns:
            String describing the dominant bias
        """
        di = metrics['disparate_impact']

        # Determine bias severity
        if di < 0.6:
            bias_level = 'severe'
        elif di < 0.8:
            bias_level = 'moderate'
        else:
            bias_level = 'mild'

        # Map task to expected bias
        bias_map = {
            'software_developer': f'{bias_level} gender bias',
            'financial_analyst': f'{bias_level} education/race bias',
            'healthcare_worker': f'{bias_level} age bias',
            'customer_service': f'{bias_level} minimal bias',
            'management_role': f'{bias_level} intersectional bias'
        }

        return bias_map.get(task_name, f'{bias_level} {attr_name} bias')


def run_benchmark_suite(random_seed: int = 42) -> Dict[str, Any]:
    """
    Run the complete benchmark suite in one call

    Args:
        random_seed: Random seed for reproducibility

    Returns:
        Complete benchmark results
    """
    harness = BenchmarkHarness(random_seed=random_seed)
    tasks = harness.load_5_task_suite()
    results = harness.measure_baseline(tasks)
    return results
