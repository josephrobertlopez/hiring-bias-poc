"""BDD step definitions for audit_harness module

NOTE: Audit harness scenarios have been removed pending honest implementation.
All previous scenarios used np.random.uniform() as mock "algorithm predictions"
which is pure statistical theater — testing whether RNG produces values in a range,
not testing real algorithm behavior.

These steps remain to support basic harness initialization only.
Real algorithm comparison requires actual trained models, not random noise.
"""

from behave import given, when, then
import numpy as np


@given('audit harness is initialized')
def step_init_harness(context):
    context.harness_ready = True
    context.random_seed = 42


@given('random seed is fixed to 42')
def step_set_seed(context):
    np.random.seed(42)
    context.random_seed = 42
