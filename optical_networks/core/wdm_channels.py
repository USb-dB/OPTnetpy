from typing import List, Dict, Optional
import numpy as np
from scipy import special


class WDMChannel:
    """WDM Channel representation with physical layer parameters"""

    def __init__(self, channel_id: int, center_frequency: float, bandwidth: float):
        self.channel_id = channel_id
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.is_allocated = False
        self.power = 0.0  # dBm
        self.osnr = 0.0  # Optical Signal-to-Noise Ratio
        self.ber = 0.0  # Bit Error Rate
        self.q_factor = 0.0
        self.distance = 0.0  # Transmission distance in km

    def calculate_ber(self, osnr: float, modulation: str = "QPSK") -> float:
        """Calculate Bit Error Rate based on OSNR and modulation format"""
        modulation_penalty = {
            "BPSK": 0,
            "QPSK": 3,
            "8QAM": 6,
            "16QAM": 9,
            "64QAM": 12
        }

        penalty = modulation_penalty.get(modulation, 3)
        effective_osnr = osnr - penalty

        # Simplified BER calculation for optical systems
        # Using Q-function approximation: BER ≈ 0.5 * erfc(sqrt(OSNR/2))
        if effective_osnr <= 0:
            return 0.5  # Worst case

        # More realistic BER calculation for optical communications
        ber = 0.5 * special.erfc(np.sqrt(effective_osnr / 2))
        return max(ber, 1e-12)  # Avoid extremely small values

    def calculate_q_factor(self, ber: float) -> float:
        """Calculate Q-factor from BER"""
        if ber <= 0 or ber >= 0.5:
            return 0.0

        # Q = sqrt(2) * erfinv(1 - 2*BER)
        try:
            q_factor = np.sqrt(2) * special.erfcinv(2 * ber)
            return float(q_factor)
        except (ValueError, FloatingPointError):
            return 0.0

    def calculate_osnr_from_power(self, distance: float, attenuation: float = 0.2) -> float:
        """Calculate OSNR based on transmission distance and power"""
        # Simplified OSNR calculation
        # OSNR = P_signal - P_noise - Loss
        loss = attenuation * distance  # dB
        noise_figure = 5.0  # dB, typical for optical amplifiers

        self.osnr = self.power - loss - noise_figure
        self.distance = distance

        # Calculate BER based on the new OSNR
        self.ber = self.calculate_ber(self.osnr)
        self.q_factor = self.calculate_q_factor(self.ber)

        return self.osnr


class WDMChannelManager:
    """Manager for WDM channels with frequency grid support"""

    def __init__(self, start_frequency: float = 191.3e12,  # ~1565 nm
                 end_frequency: float = 196.1e12,  # ~1528 nm
                 channel_spacing: float = 50e9,  # 50 GHz spacing
                 channel_bandwidth: float = 37.5e9):  # 37.5 GHz bandwidth

        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.channel_spacing = channel_spacing
        self.channel_bandwidth = channel_bandwidth
        self.channels = self._initialize_channels()

    def _initialize_channels(self) -> List[WDMChannel]:
        """Initialize WDM channels based on ITU-T grid"""
        channels = []
        channel_id = 0
        current_freq = self.start_frequency

        while current_freq <= self.end_frequency:
            channels.append(WDMChannel(
                channel_id=channel_id,
                center_frequency=current_freq,
                bandwidth=self.channel_bandwidth
            ))
            channel_id += 1
            current_freq += self.channel_spacing

        return channels

    def find_available_channels(self, num_channels: int = 1) -> List[WDMChannel]:
        """Find contiguous available WDM channels"""
        available = []
        for channel in self.channels:
            if not channel.is_allocated:
                available.append(channel)
                if len(available) == num_channels:
                    return available
            else:
                available = []

        return []

    def allocate_channels(self, channel_ids: List[int], power: float = 0.0, distance: float = 100.0):
        """Allocate specific WDM channels with power and distance"""
        for channel_id in channel_ids:
            if 0 <= channel_id < len(self.channels):
                channel = self.channels[channel_id]
                channel.is_allocated = True
                channel.power = power
                # Calculate OSNR and BER for this channel
                channel.calculate_osnr_from_power(distance)

    def release_channels(self, channel_ids: List[int]):
        """Release allocated WDM channels"""
        for channel_id in channel_ids:
            if 0 <= channel_id < len(self.channels):
                channel = self.channels[channel_id]
                channel.is_allocated = False
                channel.power = 0.0
                channel.osnr = 0.0
                channel.ber = 0.0
                channel.q_factor = 0.0

    def get_channel_utilization(self) -> float:
        """Get overall channel utilization"""
        allocated = sum(1 for channel in self.channels if channel.is_allocated)
        return (allocated / len(self.channels)) * 100 if self.channels else 0

    def get_system_performance(self) -> Dict:
        """Get comprehensive system performance metrics"""
        allocated_channels = [ch for ch in self.channels if ch.is_allocated]

        if not allocated_channels:
            return {
                'total_channels': len(self.channels),
                'allocated_channels': 0,
                'utilization': 0.0,
                'avg_osnr': 0.0,
                'avg_ber': 0.0,
                'avg_q_factor': 0.0
            }

        avg_osnr = np.mean([ch.osnr for ch in allocated_channels])
        avg_ber = np.mean([ch.ber for ch in allocated_channels])
        avg_q_factor = np.mean([ch.q_factor for ch in allocated_channels])

        return {
            'total_channels': len(self.channels),
            'allocated_channels': len(allocated_channels),
            'utilization': self.get_channel_utilization(),
            'avg_osnr': avg_osnr,
            'avg_ber': avg_ber,
            'avg_q_factor': avg_q_factor
        }

    def find_best_channels(self, num_channels: int = 1, min_osnr: float = 15.0) -> List[WDMChannel]:
        """Find best available channels based on estimated performance"""
        available_channels = [ch for ch in self.channels if not ch.is_allocated]

        # For available channels, estimate performance
        for channel in available_channels:
            # Set a nominal power and distance for estimation
            channel.power = 0.0  # dBm
            channel.calculate_osnr_from_power(distance=100.0)

        # Sort by estimated OSNR (higher is better)
        available_channels.sort(key=lambda ch: ch.osnr, reverse=True)

        # Filter by minimum OSNR requirement
        suitable_channels = [ch for ch in available_channels if ch.osnr >= min_osnr]

        return suitable_channels[:num_channels]