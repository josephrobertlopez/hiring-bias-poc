"""Behave environment setup for benchmark tests"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def before_all(context):
    """Initialize test context before all scenarios"""
    context.seed = 42
