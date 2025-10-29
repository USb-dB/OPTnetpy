"""
Algorithms for routing, spectrum allocation, and machine learning

This module provides various algorithms for:
- Path finding and routing (Dijkstra, K-Shortest Paths)
- Spectrum allocation strategies (First-Fit, Best-Fit, Random-Fit)
- Machine learning for traffic prediction and proactive allocation
"""

from .routing import DijkstraRouter, KSPRouter, EnhancedDijkstraRouter
from .spectrum_allocation import FirstFit, RandomFit, BestFit, SpectrumAllocationManager
from .ml_predictors import TrafficPredictor, MLPredictiveAllocator

__all__ = [
    'DijkstraRouter',
    'KSPRouter',
    'EnhancedDijkstraRouter',
    'FirstFit',
    'RandomFit',
    'BestFit',
    'SpectrumAllocationManager',
    'TrafficPredictor',
    'MLPredictiveAllocator'
]

__version__ = "1.0.0"