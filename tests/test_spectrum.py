import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optical_networks.core.spectrum import SpectrumSlot, SpectrumBand


class TestSpectrum(unittest.TestCase):

    def test_spectrum_slot_creation(self):
        slot = SpectrumSlot(191.0e12, 191.1e12, channel_id=1)
        self.assertEqual(slot.start_frequency, 191.0e12)
        self.assertEqual(slot.end_frequency, 191.1e12)
        self.assertEqual(slot.bandwidth, 0.1e12)
        self.assertFalse(slot.is_allocated)

    def test_spectrum_band_initialization(self):
        band = SpectrumBand(10, 100e9)
        self.assertEqual(len(band.slots), 10)
        self.assertEqual(band.slot_width, 10e9)

    def test_contiguous_slot_allocation(self):
        band = SpectrumBand(10, 100e9)
        start, end = band.find_contiguous_slots(3)
        self.assertEqual(start, 0)
        self.assertEqual(end, 2)

    def test_spectrum_utilization(self):
        band = SpectrumBand(10, 100e9)
        band.allocate_slots(0, 4, "test_conn")
        utilization = band.get_utilization()
        self.assertEqual(utilization, 50.0)  # 5 out of 10 slots

    def test_spectrum_fragmentation(self):
        band = SpectrumBand(10, 100e9)
        # Allocate non-contiguous slots to create fragmentation
        band.allocate_slots(0, 0, "conn1")
        band.allocate_slots(2, 2, "conn2")
        band.allocate_slots(4, 4, "conn3")

        fragmentation = band.get_fragmentation()
        self.assertGreater(fragmentation, 0)


if __name__ == '__main__':
    unittest.main()