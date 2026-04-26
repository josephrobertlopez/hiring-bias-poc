"""Data generation utilities for synthetic hiring tasks"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


def create_synthetic_resume_data(
    n_samples: int,
    n_features: int = 15,
    protected_attr_name: str = 'gender',
    bias_factor: float = 0.3,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Create synthetic resume scoring dataset with injected bias patterns

    Args:
        n_samples: Number of resumes to generate
        n_features: Number of features per resume
        protected_attr_name: Name of protected attribute (gender, race, age, education)
        bias_factor: Strength of bias injection (0.0-1.0)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y, protected_attrs_df, attr_info)
    """
    np.random.seed(random_seed)

    # Generate base features (uniformly distributed)
    X = np.random.uniform(0, 1, size=(n_samples, n_features))

    # Create protected attribute
    if protected_attr_name == 'gender':
        # Binary: male=0, female=1
        protected_attr = np.random.binomial(1, 0.5, n_samples)
        attr_map = {0: 'male', 1: 'female'}
        groups = [0, 1]
    elif protected_attr_name == 'race':
        # Multiclass: white=0, black=1, hispanic=2, asian=3
        protected_attr = np.random.choice([0, 1, 2, 3], n_samples)
        attr_map = {0: 'white', 1: 'black', 2: 'hispanic', 3: 'asian'}
        groups = [0, 1, 2, 3]
    elif protected_attr_name == 'age':
        # Binary: young=0 (<35), old=1 (>=35)
        ages = np.random.uniform(25, 65, n_samples)
        protected_attr = (ages >= 35).astype(int)
        attr_map = {0: 'young', 1: 'old'}
        groups = [0, 1]
    elif protected_attr_name == 'education':
        # Ordinal: high_school=0, bachelors=1, masters=2, phd=3
        protected_attr = np.random.choice([0, 1, 2, 3], n_samples)
        attr_map = {0: 'high_school', 1: 'bachelors', 2: 'masters', 3: 'phd'}
        groups = [0, 1, 2, 3]
    else:
        # Default to binary
        protected_attr = np.random.binomial(1, 0.5, n_samples)
        attr_map = {0: 'group_0', 1: 'group_1'}
        groups = [0, 1]

    # Generate labels with bias injection
    # Start with base probability from feature sum
    base_prob = (X.mean(axis=1) + 0.3) / 1.3  # Normalize to ~[0, 1]

    # Inject bias: privileged group gets higher baseline
    for group in groups[1:]:
        mask = protected_attr == group
        base_prob[mask] = base_prob[mask] * (1 - bias_factor)

    # Apply bias to first group (privileged)
    mask = protected_attr == groups[0]
    base_prob[mask] = base_prob[mask] * (1 + bias_factor * 0.3)

    # Clip to valid probability range
    base_prob = np.clip(base_prob, 0.1, 0.9)

    # Generate labels from biased probabilities
    y = np.random.binomial(1, base_prob).astype(int)

    # Create DataFrame with protected attributes
    protected_df = pd.DataFrame({
        protected_attr_name: protected_attr,
        f'{protected_attr_name}_str': [attr_map.get(x, str(x)) for x in protected_attr]
    })

    attr_info = {
        'name': protected_attr_name,
        'map': attr_map,
        'groups': groups,
        'bias_factor': bias_factor
    }

    return X, y, protected_df, attr_info


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    protected_attr: np.ndarray,
    min_per_group: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure stratified distribution across protected groups

    Args:
        X: Feature matrix
        y: Labels
        protected_attr: Protected attribute values
        min_per_group: Minimum samples per group

    Returns:
        Filtered (X, y, protected_attr)
    """
    groups = np.unique(protected_attr)
    valid_indices = []

    for group in groups:
        mask = protected_attr == group
        indices = np.where(mask)[0]
        if len(indices) >= min_per_group:
            valid_indices.extend(indices)

    valid_indices = np.array(valid_indices)
    return X[valid_indices], y[valid_indices], protected_attr[valid_indices]
