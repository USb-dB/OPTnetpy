"""
Core components for optical network simulation

This module contains the fundamental building blocks for optical network simulation:
- Spectrum management (slots and bands)
- Network graph topology with redundancy
- WDM channel management
"""

from .spectrum import SpectrumSlot, SpectrumBand
from .graph import RedundantGraph
from .wdm_channels import WDMChannel, WDMChannelManager

__all__ = [
    'SpectrumSlot',
    'SpectrumBand',
    'RedundantGraph',
    'WDMChannel',
    'WDMChannelManager'
]

__version__ = "1.0.0"