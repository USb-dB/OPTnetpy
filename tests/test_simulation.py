import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optical_networks.simulation.engine import OpticalNetworkSimulator
from optical_networks.simulation.events import Event, EventType, EventManager


class TestSimulation(unittest.TestCase):

    def test_simulator_initialization(self):
        """Test simulator initialization"""
        simulator = OpticalNetworkSimulator(nodes=6, channels=20)
        self.assertEqual(simulator.nodes, 6)
        self.assertEqual(simulator.channels, 20)
        self.assertIsNotNone(simulator.graph)

    def test_event_manager(self):
        """Test event management"""
        event_manager = EventManager()

        # Schedule an event
        event = Event(EventType.CONNECTION_REQUEST, 10.0)
        event_manager.schedule_event(event)

        self.assertEqual(len(event_manager.event_queue), 1)

    def test_traffic_generation(self):
        """Test traffic generation"""
        simulator = OpticalNetworkSimulator(nodes=5, channels=15)
        requests = simulator.generate_traffic(duration=100)

        self.assertGreater(len(requests), 0)
        for request in requests:
            self.assertEqual(len(request), 5)  # src, dest, bw, duration, arrival

    def test_ml_predictor_integration(self):
        """Test ML predictor integration"""
        simulator = OpticalNetworkSimulator(nodes=4, channels=10)
        self.assertIsNotNone(simulator.ml_allocator)

    def test_simulation_metrics(self):
        """Test simulation metrics collection"""
        simulator = OpticalNetworkSimulator(nodes=3, channels=8)

        # Run a very short simulation
        simulator.run_simulation(duration=10, show_progress=False)

        self.assertGreater(len(simulator.simulation_history), 0)

        final_metrics = simulator.simulation_history[-1]
        self.assertIn('blocking_probability', final_metrics)


if __name__ == '__main__':
    unittest.main()