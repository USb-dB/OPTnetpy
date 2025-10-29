"""
Test suite for Optical Network Simulator

This module contains unit tests for all components of the optical network simulator.
Run tests using: python -m pytest tests/
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

__version__ = "1.0.0"