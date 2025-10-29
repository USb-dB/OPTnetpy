# Optical Network Simulator

A comprehensive simulation platform for optical networks and interconnects featuring Dynamic Spectrum Allocation, Wavelength Division Multiplexing (WDM), and Machine Learning capabilities.

## Features

- **Dynamic Spectrum Allocation**: Flexible spectrum slot management with fragmentation metrics
- **WDM Support**: Multiple wavelength channels with physical layer parameters
- **Redundant Path Routing**: Kruskal-based MST with redundant edges for fault tolerance
- **Machine Learning Integration**: Traffic prediction and proactive spectrum allocation
- **Multiple Algorithms**: First-Fit, Best-Fit, Random-Fit spectrum allocation
- **Advanced Visualization**: Network topology, spectrum usage, and performance metrics
- **Event-driven Simulation**: Realistic network operation simulation
- **Comprehensive Metrics**: Utilization, fragmentation, blocking probability tracking

## Installation

### From Source
```bash
git clone https://github.com/USb-dB/OPTnetpy
cd OPTnetpy
pip install -r requirements.txt
pip install -e .
