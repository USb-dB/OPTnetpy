"""
Utility functions and configuration management

This module provides:
- Performance metrics collection and analysis
- Configuration management for networks and simulations
- Helper functions and data structures
"""

from .metrics import NetworkMetrics, ConnectionMetrics
from .config import NetworkConfig, SimulationConfig, ConfigManager

__all__ = ['NetworkMetrics', 'ConnectionMetrics', 'NetworkConfig', 'SimulationConfig', 'ConfigManager']

__version__ = "1.0.0"

# Default configurations
DEFAULT_NETWORK_CONFIG = NetworkConfig()
DEFAULT_SIMULATION_CONFIG = SimulationConfig()