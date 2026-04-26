"""5-Task synthetic benchmark suite for hiring bias detection"""

from typing import Dict, Any
from .data_utils import create_synthetic_resume_data, stratified_sample
import numpy as np


def create_task_data(
    task_name: str,
    n_samples: int = 200,
    bias_factor: float = 0.3,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a single task dataset

    Args:
        task_name: Name of task (must be one of the 5 standard tasks)
        n_samples: Number of samples to generate
        bias_factor: Strength of bias injection (0.0-1.0)
        random_seed: Random seed for reproducibility

    Returns:
        Dict with X, y, protected_attr, protected_attr_info
    """
    np.random.seed(random_seed)

    # Map task names to protected attributes
    task_config = {
        'software_developer': {
            'protected_attr': 'gender',
            'n_features': 15,
            'bias_factor': bias_factor,
            'description': 'Gender bias in software development hiring'
        },
        'financial_analyst': {
            'protected_attr': 'education',
            'n_features': 12,
            'bias_factor': bias_factor + 0.1,
            'description': 'Education and race bias in finance'
        },
        'healthcare_worker': {
            'protected_attr': 'age',
            'n_features': 13,
            'bias_factor': bias_factor + 0.15,
            'description': 'Age bias in healthcare hiring'
        },
        'customer_service': {
            'protected_attr': 'gender',
            'n_features': 10,
            'bias_factor': bias_factor * 0.5,
            'description': 'Minimal bias baseline (customer service)'
        },
        'management_role': {
            'protected_attr': 'race',
            'n_features': 16,
            'bias_factor': bias_factor + 0.2,
            'description': 'Intersectional bias in management roles'
        }
    }

    if task_name not in task_config:
        raise ValueError(f"Unknown task: {task_name}")

    config = task_config[task_name]

    # Generate synthetic data
    X, y, protected_df, attr_info = create_synthetic_resume_data(
        n_samples=n_samples,
        n_features=config['n_features'],
        protected_attr_name=config['protected_attr'],
        bias_factor=config['bias_factor'],
        random_seed=random_seed
    )

    # Ensure stratification
    protected_attr = protected_df[config['protected_attr']].values
    X, y, protected_attr = stratified_sample(X, y, protected_attr, min_per_group=10)

    return {
        'X': X,
        'y': y,
        'protected_attr': attr_info,
        'protected_attr_values': protected_attr,
        'description': config['description'],
        'task_name': task_name
    }


def create_5_task_suite(random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Create the complete 5-task benchmark suite

    Args:
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping task names to task data
    """
    tasks = {}
    task_names = [
        'software_developer',
        'financial_analyst',
        'healthcare_worker',
        'customer_service',
        'management_role'
    ]

    for i, task_name in enumerate(task_names):
        # Use different seed for each task but derived from main seed
        task_seed = random_seed + i
        tasks[task_name] = create_task_data(task_name, random_seed=task_seed)

    return tasks
