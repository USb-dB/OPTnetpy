"""
Simulation engine and event management

This module provides:
- Main simulation engine for optical networks
- Event-driven simulation management
- Traffic generation and simulation control
"""

from .engine import OpticalNetworkSimulator
from .events import Event, EventType, EventManager

__all__ = ['OpticalNetworkSimulator', 'Event', 'EventType', 'EventManager']

__version__ = "1.0.0"

# Common event types for easy access
CONNECTION_REQUEST = EventType.CONNECTION_REQUEST
CONNECTION_RELEASE = EventType.CONNECTION_RELEASE
SPECTRUM_ALLOCATION = EventType.SPECTRUM_ALLOCATION
SPECTRUM_RELEASE = EventType.SPECTRUM_RELEASE
METRICS_UPDATE = EventType.METRICS_UPDATE
DEFRAGMENTATION = EventType.DEFRAGMENTATION
ML_RETRAINING = EventType.ML_RETRAINING