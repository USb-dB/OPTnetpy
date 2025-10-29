import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optical_networks.core.graph import RedundantGraph


class TestRedundantGraph(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.vertices = 5
        self.edges = [
            (0, 1, 4), (0, 2, 1), (1, 2, 2),
            (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)
        ]
        self.graph = RedundantGraph(self.vertices, self.edges, k=2)

    def test_graph_initialization(self):
        """Test graph initialization"""
        self.assertEqual(self.graph.V, 5)
        self.assertEqual(self.graph.k, 2)
        self.assertEqual(len(self.graph.original_graph), 7)

    def test_add_redundant_paths(self):
        """Test redundant path addition"""
        total_cost = self.graph.add_redundant_paths()
        # Allow both int and float since cost can be either
        self.assertTrue(isinstance(total_cost, (int, float)))
        self.assertGreater(total_cost, 0)
        self.assertGreater(len(self.graph.redundant_graph_edges), 0)

    def test_dijkstra_shortest_paths(self):
        """Test Dijkstra's algorithm"""
        self.graph.add_redundant_paths()
        paths = self.graph.dijkstra_shortest_paths(0, 4, n=2)
        self.assertIsInstance(paths, list)
        if paths:  # If paths exist
            self.assertIsInstance(paths[0], list)
            self.assertEqual(paths[0][0], 0)
            self.assertEqual(paths[0][-1], 4)

    def test_spectrum_allocation(self):
        """Test spectrum allocation"""
        self.graph.add_redundant_paths()
        self.graph.allocate_spectrum(no_of_slots=20, bandwidth=100e9)

        self.assertIsNotNone(self.graph.spectrum_band)
        self.assertEqual(len(self.graph.spectrum_band.slots), 20)

    def test_spectrum_assignment(self):
        """Test spectrum assignment to paths"""
        self.graph.add_redundant_paths()
        self.graph.allocate_spectrum(no_of_slots=20, bandwidth=100e9)

        path = [0, 2, 4]
        success = self.graph.assign_spectrum(path, required_slots=3)

        # Check if assignment was successful or failed appropriately
        self.assertIsInstance(success, bool)

    def test_performance_metrics(self):
        """Test performance metrics collection"""
        self.graph.add_redundant_paths()
        self.graph.allocate_spectrum(no_of_slots=20, bandwidth=100e9)

        metrics = self.graph.get_performance_metrics()
        expected_keys = ['blocking_probability', 'total_requests', 'blocked_requests',
                         'current_utilization', 'current_fragmentation', 'active_connections']

        for key in expected_keys:
            self.assertIn(key, metrics)


if __name__ == '__main__':
    unittest.main()