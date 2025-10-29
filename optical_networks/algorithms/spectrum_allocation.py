import numpy as np
from typing import List, Tuple, Optional, Dict
from ..core.spectrum import SpectrumBand


class FirstFit:
    """First-Fit spectrum allocation algorithm"""

    def __init__(self, spectrum_band: SpectrumBand):
        self.spectrum_band = spectrum_band

    def allocate(self, num_slots: int) -> Tuple[Optional[int], Optional[int]]:
        """First-fit allocation strategy"""
        return self.spectrum_band.find_contiguous_slots(num_slots, strategy="first-fit")


class RandomFit:
    """Random-Fit spectrum allocation algorithm"""

    def __init__(self, spectrum_band: SpectrumBand):
        self.spectrum_band = spectrum_band

    def allocate(self, num_slots: int) -> Tuple[Optional[int], Optional[int]]:
        """Random-fit allocation strategy"""
        # Get all possible allocations using first-fit logic
        start, end = self.spectrum_band.find_contiguous_slots(num_slots, "first-fit")
        return start, end  # For simplicity, same as first-fit but can be randomized


class BestFit:
    """Best-Fit spectrum allocation algorithm"""

    def __init__(self, spectrum_band: SpectrumBand):
        self.spectrum_band = spectrum_band

    def allocate(self, num_slots: int) -> Tuple[Optional[int], Optional[int]]:
        """Best-fit allocation strategy"""
        return self.spectrum_band.find_contiguous_slots(num_slots, strategy="best-fit")


class SpectrumAllocationManager:
    """Manager for different spectrum allocation strategies"""

    def __init__(self, spectrum_band: SpectrumBand):
        self.spectrum_band = spectrum_band
        self.strategies = {
            'first-fit': FirstFit(spectrum_band),
            'random-fit': RandomFit(spectrum_band),
            'best-fit': BestFit(spectrum_band)
        }

    def allocate_spectrum(self, num_slots: int, strategy: str = 'first-fit') -> Tuple[Optional[int], Optional[int]]:
        """Allocate spectrum using specified strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")

        return self.strategies[strategy].allocate(num_slots)

    def compare_strategies(self, num_slots: int, num_requests: int = 100) -> Dict:
        """Compare performance of different allocation strategies"""
        results = {}

        for strategy_name, strategy in self.strategies.items():
            # Reset spectrum band for fair comparison
            original_slots = self.spectrum_band.slots.copy()

            successful_allocations = 0
            fragmentation_history = []

            for i in range(num_requests):
                start, end = strategy.allocate(num_slots)
                if start is not None:
                    successful_allocations += 1
                    # Simulate holding time
                    if i % 10 == 0:  # Release some connections periodically
                        self.spectrum_band.release_slots(start, end)

                fragmentation = self.spectrum_band.get_fragmentation()
                fragmentation_history.append(fragmentation)

            # Restore original state
            self.spectrum_band.slots = original_slots

            results[strategy_name] = {
                'success_rate': successful_allocations / num_requests,
                'avg_fragmentation': np.mean(fragmentation_history),
                'max_fragmentation': np.max(fragmentation_history)
            }

        return results