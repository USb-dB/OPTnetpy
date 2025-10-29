import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import matplotlib.patches as patches


class SpectrumVisualizer:
    """Visualization tools for spectrum allocation"""

    def __init__(self, spectrum_band):
        self.spectrum_band = spectrum_band

    def plot_spectrum_usage(self, figsize: tuple = (12, 6)):
        """Create a spectrum usage map"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        slots = self.spectrum_band.slots
        num_slots = len(slots)

        # Create spectrum usage map
        usage_map = np.zeros(num_slots)
        for i, slot in enumerate(slots):
            usage_map[i] = 1 if slot.is_allocated else 0

        # Plot spectrum usage as colored bars
        colors = ['red' if used else 'green' for used in usage_map]
        ax1.bar(range(num_slots), [1] * num_slots, color=colors, width=1.0)
        ax1.set_xlabel('Slot Index')
        ax1.set_ylabel('Allocation Status')
        ax1.set_title('Spectrum Slot Allocation Map')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Create legend
        allocated_patch = patches.Patch(color='red', label='Allocated')
        free_patch = patches.Patch(color='green', label='Free')
        ax1.legend(handles=[allocated_patch, free_patch])

        # Plot frequency information
        frequencies = [slot.center_frequency for slot in slots]
        ax2.plot(range(num_slots), frequencies, 'b-', marker='o', markersize=3)
        ax2.set_xlabel('Slot Index')
        ax2.set_ylabel('Center Frequency (Hz)')
        ax2.set_title('Slot Frequency Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_utilization_history(self, utilization_history: List[float],
                                 fragmentation_history: List[float]):
        """Plot utilization and fragmentation over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        time_steps = range(len(utilization_history))

        # Plot utilization
        ax1.plot(time_steps, utilization_history, 'b-', linewidth=2, label='Utilization')
        ax1.fill_between(time_steps, utilization_history, alpha=0.3)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('Spectrum Utilization Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot fragmentation
        ax2.plot(time_steps, fragmentation_history, 'r-', linewidth=2, label='Fragmentation')
        ax2.fill_between(time_steps, fragmentation_history, alpha=0.3)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Fragmentation (%)')
        ax2.set_title('Spectrum Fragmentation Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_contiguous_blocks(self):
        """Visualize contiguous free blocks in spectrum"""
        slots = self.spectrum_band.slots
        num_slots = len(slots)

        # Find contiguous free blocks
        free_blocks = []
        current_block = 0
        block_start = 0

        for i, slot in enumerate(slots):
            if not slot.is_allocated:
                if current_block == 0:
                    block_start = i
                current_block += 1
            else:
                if current_block > 0:
                    free_blocks.append((block_start, current_block))
                    current_block = 0

        if current_block > 0:
            free_blocks.append((block_start, current_block))

        # Plot free blocks
        fig, ax = plt.subplots(figsize=(12, 4))

        for block_start, block_size in free_blocks:
            ax.barh(0, block_size, left=block_start, height=0.5,
                    color='green', alpha=0.7, edgecolor='black')

        # Mark allocated slots
        for i, slot in enumerate(slots):
            if slot.is_allocated:
                ax.barh(0, 1, left=i, height=0.5, color='red', alpha=0.7, edgecolor='black')

        ax.set_xlabel('Slot Index')
        ax.set_title('Contiguous Free Blocks in Spectrum')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

        # Add block size annotations
        for block_start, block_size in free_blocks:
            if block_size > 0:
                ax.text(block_start + block_size / 2, 0, f'{block_size}',
                        ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return free_blocks