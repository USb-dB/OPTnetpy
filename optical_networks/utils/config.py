from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json


@dataclass
class NetworkConfig:
    """Network configuration parameters"""
    num_nodes: int = 10
    num_channels: int = 40
    total_bandwidth: float = 200e9  # Hz
    redundancy_factor: int = 2
    slot_width: float = 5e9  # Hz per slot

    # Physical layer parameters
    max_transmission_distance: float = 1000.0  # km
    attenuation_coefficient: float = 0.2  # dB/km
    amplifier_spacing: float = 80.0  # km

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'num_nodes': self.num_nodes,
            'num_channels': self.num_channels,
            'total_bandwidth': self.total_bandwidth,
            'redundancy_factor': self.redundancy_factor,
            'slot_width': self.slot_width,
            'max_transmission_distance': self.max_transmission_distance,
            'attenuation_coefficient': self.attenuation_coefficient,
            'amplifier_spacing': self.amplifier_spacing
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'NetworkConfig':
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    duration: int = 1000
    traffic_intensity: float = 0.7
    connection_holding_time: int = 10
    request_arrival_rate: float = 0.1

    # ML configuration
    use_ml_predictor: bool = True
    prediction_horizon: int = 24
    retrain_interval: int = 100

    # Routing configuration
    routing_algorithm: str = 'dijkstra'
    num_alternative_paths: int = 3
    max_path_hops: int = 10

    # Spectrum allocation configuration
    allocation_strategy: str = 'first-fit'
    defragmentation_interval: int = 1000

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'duration': self.duration,
            'traffic_intensity': self.traffic_intensity,
            'connection_holding_time': self.connection_holding_time,
            'request_arrival_rate': self.request_arrival_rate,
            'use_ml_predictor': self.use_ml_predictor,
            'prediction_horizon': self.prediction_horizon,
            'retrain_interval': self.retrain_interval,
            'routing_algorithm': self.routing_algorithm,
            'num_alternative_paths': self.num_alternative_paths,
            'max_path_hops': self.max_path_hops,
            'allocation_strategy': self.allocation_strategy,
            'defragmentation_interval': self.defragmentation_interval
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulationConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class ConfigManager:
    """Configuration manager for loading/saving configurations"""

    @staticmethod
    def save_config(config: Any, filename: str):
        """Save configuration to JSON file"""
        with open(filename, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    @staticmethod
    def load_network_config(filename: str) -> NetworkConfig:
        """Load network configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return NetworkConfig.from_dict(config_dict)

    @staticmethod
    def load_simulation_config(filename: str) -> SimulationConfig:
        """Load simulation configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return SimulationConfig.from_dict(config_dict)