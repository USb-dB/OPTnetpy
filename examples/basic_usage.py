"""
Basic usage example for Optical Network Simulator
Author: Prantik Basu
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optical_networks import OpticalNetworkSimulator, RedundantGraph, WDMChannelManager
from optical_networks.algorithms.ml_predictors import MLPredictiveAllocator


def main():
    print("Optical Network Simulator - Basic Usage Example")
    print("=" * 50)

    # Initialize network
    nodes = 8
    initial_edges = [
        (0, 1, 3), (0, 2, 8), (0, 3, 5), (1, 2, 7),
        (1, 3, 9), (2, 3, 8), (2, 4, 6), (3, 4, 4),
        (3, 5, 3), (4, 5, 7), (4, 6, 8), (5, 6, 4),
        (5, 7, 5), (6, 7, 2)
    ]

    # Create redundant graph
    graph = RedundantGraph(vertices=nodes, edges=initial_edges, k=2)
    graph.add_redundant_paths()

    # Initialize spectrum
    graph.allocate_spectrum(no_of_slots=50, bandwidth=200e9)  # 200 GHz total

    # Create WDM channel manager
    wdm_manager = WDMChannelManager()

    # Create ML predictor
    ml_allocator = MLPredictiveAllocator(graph.spectrum_band)

    # Simulate connection requests
    requests = [
        (0, 4, 40e9, 10),  # src, dest, bandwidth, duration
        (1, 5, 25e9, 8),
        (2, 6, 60e9, 15),
        (3, 7, 30e9, 5)
    ]

    print("Handling connection requests...")
    for i, (src, dest, bandwidth, duration) in enumerate(requests):
        print(f"Request {i + 1}: {src} -> {dest}, BW: {bandwidth / 1e9} GHz, Duration: {duration}")

        # Find paths
        paths = graph.dijkstra_shortest_paths(src, dest, n=2)

        if paths:
            path = paths[0]
            required_slots = int(bandwidth / graph.spectrum_band.slot_width)

            # Allocate spectrum
            success = graph.assign_spectrum(path, required_slots)
            if success:
                print(f"  ✓ Spectrum allocated for path: {path}")

                # Update ML predictor
                ml_allocator.update_traffic_history(i, bandwidth)
            else:
                print(f"  ✗ Failed to allocate spectrum for path: {path}")
        else:
            print(f"  ✗ No path found from {src} to {dest}")

    # Print network statistics
    print("\nNetwork Statistics:")
    print(f"Total graph cost: {sum(edge[2] for edge in graph.redundant_graph_edges)}")
    print(f"Spectrum utilization: {graph.spectrum_band.get_utilization():.2f}%")
    print(f"Spectrum fragmentation: {graph.spectrum_band.get_fragmentation():.2f}%")
    print(f"WDM channel utilization: {wdm_manager.get_channel_utilization():.2f}%")

    # Demonstrate ML prediction
    print("\nML Prediction Demo:")
    predicted_slots = ml_allocator.predict_spectrum_demand(current_time=10)
    print(f"Predicted spectrum demand: {predicted_slots} slots")

    # Visualize
    print("\nGenerating visualization...")
    graph.plot_graphs(src=0, dest=4, n=2, required_slots=5)


if __name__ == "__main__":
    main()