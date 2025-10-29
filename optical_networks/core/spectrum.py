import numpy as np
from typing import List, Tuple, Optional


class SpectrumSlot:
    """Enhanced spectrum slot with WDM support"""

    def __init__(self, start_frequency: float, end_frequency: float, channel_id: int = None):
        self.start_frequency = round(start_frequency, 2)
        self.end_frequency = round(end_frequency, 2)
        self.center_frequency = round((start_frequency + end_frequency) / 2, 2)
        self.channel_id = channel_id
        self.is_allocated = False
        self.signal_power = 0.0  # dBm
        self.noise_power = 0.0  # dBm
        self.connection_id = None

    @property
    def bandwidth(self) -> float:
        return self.end_frequency - self.start_frequency

    @property
    def snr(self) -> float:
        """Calculate Signal-to-Noise Ratio"""
        if self.noise_power == 0:
            return float('inf')
        return self.signal_power - self.noise_power

    def allocate(self, connection_id: str, signal_power: float = 0.0):
        self.is_allocated = True
        self.connection_id = connection_id
        self.signal_power = signal_power

    def release(self):
        self.is_allocated = False
        self.connection_id = None
        self.signal_power = 0.0
        self.noise_power = 0.0


class SpectrumBand:
    """Enhanced spectrum band management with WDM capabilities"""

    def __init__(self, num_slots: int, total_bandwidth: float, band_id: str = "C-band"):
        self.num_slots = num_slots
        self.total_bandwidth = total_bandwidth
        self.band_id = band_id
        self.slot_width = total_bandwidth / num_slots
        self.slots = self._initialize_slots()

    def _initialize_slots(self) -> List[SpectrumSlot]:
        """Initialize spectrum slots with WDM channel numbering"""
        slots = []
        for i in range(self.num_slots):
            start_freq = i * self.slot_width
            end_freq = (i + 1) * self.slot_width
            slots.append(SpectrumSlot(start_freq, end_freq, channel_id=i))
        return slots

    def find_contiguous_slots(self, num_slots: int, strategy: str = "first-fit") -> Tuple[Optional[int], Optional[int]]:
        """Find contiguous slots using different strategies"""
        if strategy == "first-fit":
            return self._first_fit(num_slots)
        elif strategy == "best-fit":
            return self._best_fit(num_slots)
        else:
            return self._first_fit(num_slots)

    def _first_fit(self, num_slots: int) -> Tuple[Optional[int], Optional[int]]:
        """First-fit spectrum allocation"""
        contiguous = 0
        start_index = -1

        for i, slot in enumerate(self.slots):
            if not slot.is_allocated:
                if contiguous == 0:
                    start_index = i
                contiguous += 1
                if contiguous == num_slots:
                    return start_index, start_index + num_slots - 1
            else:
                contiguous = 0
                start_index = -1

        return None, None

    def _best_fit(self, num_slots: int) -> Tuple[Optional[int], Optional[int]]:
        """Best-fit spectrum allocation - find the smallest available block that fits"""
        best_start = -1
        best_size = float('inf')
        current_start = -1
        current_count = 0

        for i, slot in enumerate(self.slots):
            if not slot.is_allocated:
                if current_count == 0:
                    current_start = i
                current_count += 1
            else:
                if current_count >= num_slots and current_count < best_size:
                    best_start = current_start
                    best_size = current_count
                current_count = 0

        # Check the last segment
        if current_count >= num_slots and current_count < best_size:
            best_start = current_start
            best_size = current_count

        if best_start != -1:
            return best_start, best_start + num_slots - 1
        return None, None

    def allocate_slots(self, start: int, end: int, connection_id: str, signal_power: float = 0.0):
        """Allocate a range of slots with power management"""
        for i in range(start, end + 1):
            self.slots[i].allocate(connection_id, signal_power)

    def release_slots(self, start: int, end: int):
        """Release a range of slots"""
        for i in range(start, end + 1):
            self.slots[i].release()

    def get_utilization(self) -> float:
        """Calculate spectrum utilization percentage"""
        allocated = sum(1 for slot in self.slots if slot.is_allocated)
        return (allocated / self.num_slots) * 100

    def get_fragmentation(self) -> float:
        """Calculate spectrum fragmentation metric"""
        free_blocks = []
        current_block = 0

        for slot in self.slots:
            if not slot.is_allocated:
                current_block += 1
            else:
                if current_block > 0:
                    free_blocks.append(current_block)
                    current_block = 0

        if current_block > 0:
            free_blocks.append(current_block)

        if not free_blocks:
            return 0.0

        max_block = max(free_blocks)
        total_free = sum(free_blocks)

        if total_free == 0:
            return 100.0

        return (1 - (max_block / total_free)) * 100