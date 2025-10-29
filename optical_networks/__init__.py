"""
Optical Network Simulator with Dynamic Spectrum Allocation and WDM Support

A comprehensive simulation platform for optical networks and interconnects
featuring Dynamic Spectrum Allocation, Wavelength Division Multiplexing (WDM),
and Machine Learning capabilities.

Author: Prantik Basu
Version: 1.0.0
"""

from .core.spectrum import SpectrumSlot, SpectrumBand
from .core.graph import RedundantGraph
from .core.wdm_channels import WDMChannel, WDMChannelManager

from .algorithms.routing import DijkstraRouter, KSPRouter
from .algorithms.spectrum_allocation import FirstFit, RandomFit, BestFit, SpectrumAllocationManager
from .algorithms.ml_predictors import TrafficPredictor, MLPredictiveAllocator

from .visualization.network_plot import NetworkVisualizer
from .visualization.spectrum_visualizer import SpectrumVisualizer

from .utils.metrics import NetworkMetrics, ConnectionMetrics
from .utils.config import NetworkConfig, SimulationConfig, ConfigManager

from .simulation.engine import OpticalNetworkSimulator
from .simulation.events import Event, EventType, EventManager

__version__ = "1.0.0"
__author__ = "Prantik Basu"

__all__ = [
    # Core components
    'SpectrumSlot',
    'SpectrumBand',
    'RedundantGraph',
    'WDMChannel',
    'WDMChannelManager',

    # Algorithms
    'DijkstraRouter',
    'KSPRouter',
    'FirstFit',
    'RandomFit',
    'BestFit',
    'SpectrumAllocationManager',
    'TrafficPredictor',
    'MLPredictiveAllocator',

    # Visualization
    'NetworkVisualizer',
    'SpectrumVisualizer',

    # Utilities
    'NetworkMetrics',
    'ConnectionMetrics',
    'NetworkConfig',
    'SimulationConfig',
    'ConfigManager',

    # Simulation
    'OpticalNetworkSimulator',
    'Event',
    'EventType',
    'EventManager'
]

# Package metadata
__description__ = "A simulation platform for optical networks with Dynamic Spectrum Allocation and WDM"
__keywords__ = ["optical networks", "spectrum allocation", "WDM", "simulation", "machine learning"]
__license__ = "MIT"
__url__ = "https://github.com/prantikb/optical-network-simulator"

# Import commonly used functions for easier access
def create_simulator(nodes: int = 10, channels: int = 40, total_bandwidth: float = 200e9, k: int = 2):
    """Convenience function to create a simulator instance"""
    return OpticalNetworkSimulator(nodes=nodes, channels=channels, total_bandwidth=total_bandwidth, k=k)

def create_network_config(nodes: int = 10, channels: int = 40, total_bandwidth: float = 200e9):
    """Convenience function to create network configuration"""
    return NetworkConfig(num_nodes=nodes, num_channels=channels, total_bandwidth=total_bandwidth)

def create_simulation_config(duration: int = 1000, use_ml: bool = True):
    """Convenience function to create simulation configuration"""
    return SimulationConfig(duration=duration, use_ml_predictor=use_ml)