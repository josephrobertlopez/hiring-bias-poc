"""Shared BDD step definitions used across multiple modules"""

from behave import given
import numpy as np


@given('random seed is fixed to {seed:d}')
def step_set_seed_shared(context, seed):
    """Set random seed for reproducibility - shared across all modules"""
    context.random_seed = seed
    np.random.seed(seed)
