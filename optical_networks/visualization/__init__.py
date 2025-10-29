"""
Visualization tools for optical networks and spectrum allocation

This module provides comprehensive visualization capabilities:
- Network topology plotting with path highlighting
- Spectrum usage maps and allocation visualization
- Performance metrics plotting over time
"""

from .network_plot import NetworkVisualizer
from .spectrum_visualizer import SpectrumVisualizer

__all__ = ['NetworkVisualizer', 'SpectrumVisualizer']

__version__ = "1.0.0"

# Convenience functions
def create_network_visualizer(graph):
    """Create a network visualizer for the given graph"""
    return NetworkVisualizer(graph)

def create_spectrum_visualizer(spectrum_band):
    """Create a spectrum visualizer for the given spectrum band"""
    return SpectrumVisualizer(spectrum_band)