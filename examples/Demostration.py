"""
Fixed Comprehensive Demonstration of Optical Network Simulator
Author: Prantik Basu
"""

import numpy as np
from optical_networks import OpticalNetworkSimulator
from optical_networks.core import WDMChannelManager
from optical_networks.algorithms import SpectrumAllocationManager, EnhancedDijkstraRouter
from optical_networks.visualization import NetworkVisualizer, SpectrumVisualizer

def comprehensive_example():
    print("Optical Network Simulator - Fixed Comprehensive Example")
    print("=" * 60)

    # Initialize simulator with custom parameters
    simulator = OpticalNetworkSimulator(
        nodes=8,
        channels=40,
        total_bandwidth=200e9,
        k=2
    )

    # Run simulation
    print("Running simulation...")
    simulator.run_simulation(duration=100, show_progress=True)

    # Display results
    print("\nSimulation Results:")
    print("-" * 30)
    final_metrics = simulator.simulation_history[-1] if simulator.simulation_history else {}
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    # Plot performance metrics
    print("\nGenerating performance plots...")
    simulator.plot_performance_metrics()

    # Demonstrate WDM capabilities
    print("\nWDM System Demonstration:")
    print("-" * 30)
    wdm_system = WDMChannelManager()
    print(f"WDM System: {len(wdm_system.channels)} channels")
    print(f"Frequency Range: {wdm_system.start_frequency/1e12:.1f} - {wdm_system.end_frequency/1e12:.1f} THz")

    # Test spectrum allocation strategies
    print("\nSpectrum Allocation Strategy Comparison:")
    print("-" * 40)
    allocation_manager = SpectrumAllocationManager(simulator.graph.spectrum_band)
    results = allocation_manager.compare_strategies(num_slots=3, num_requests=20)

    for strategy, metrics in results.items():
        print(f"{strategy.upper():<12}: Success Rate: {metrics['success_rate']:.3f}, "
              f"Avg Fragmentation: {metrics['avg_fragmentation']:.2f}%")

    # Demonstrate robust routing
    print("\nEnhanced Routing Demonstration:")
    print("-" * 35)
    enhanced_router = EnhancedDijkstraRouter(simulator.graph)
    diverse_paths = enhanced_router.find_all_shortest_paths(0, 4, max_paths=3)

    print(f"Found {len(diverse_paths)} diverse paths from node 0 to 4:")
    for i, path in enumerate(diverse_paths):
        path_cost = simulator.graph.get_path_cost(path)
        print(f"  Path {i+1}: {path}, Cost: {path_cost}")

    # Visualization examples
    print("\nGenerating Visualizations...")
    print("-" * 30)

    # Network topology visualization
    visualizer = NetworkVisualizer(simulator.graph)
    visualizer.plot_network_topology(highlight_paths=diverse_paths, show_spectrum=True)

    # Spectrum visualization
    if simulator.graph.spectrum_band:
        spectrum_viz = SpectrumVisualizer(simulator.graph.spectrum_band)
        spectrum_viz.plot_spectrum_usage()

        # Show contiguous blocks
        free_blocks = spectrum_viz.plot_contiguous_blocks()
        print(f"Found {len(free_blocks)} contiguous free blocks")

    # Connection request analysis
    traffic_requests = simulator.generate_traffic(duration=100)
    visualizer.plot_connection_requests(traffic_requests)

    print("\nExample completed successfully!")

def basic_routing_example():
    """Basic routing example with error handling"""
    print("\nBasic Routing Example:")
    print("=" * 30)

    # Create a simple network
    from optical_networks.core.graph import RedundantGraph

    nodes = 6
    edges = [
        (0, 1, 2), (0, 2, 4), (1, 2, 1),
        (1, 3, 7), (2, 3, 3), (2, 4, 5),
        (3, 4, 2), (3, 5, 6), (4, 5, 4)
    ]

    graph = RedundantGraph(nodes, edges, k=2)
    graph.add_redundant_paths()
    graph.allocate_spectrum(no_of_slots=20, bandwidth=100e9)

    # Test routing
    router = EnhancedDijkstraRouter(graph)

    # Test valid path
    print("Testing path from 0 to 5:")
    path = router.find_shortest_path(0, 5)
    if path:
        cost = graph.get_path_cost(path)
        print(f"Found path: {path}, Cost: {cost}")
    else:
        print("No path found")

    # Test invalid path
    print("\nTesting invalid path (non-existent node):")
    path = router.find_shortest_path(0, 10)  # Node 10 doesn't exist
    if not path:
        print("Correctly handled - no path found for non-existent node")

    # Test multiple paths
    print("\nFinding multiple paths from 0 to 5:")
    paths = router.find_all_shortest_paths(0, 5, max_paths=3)
    for i, path in enumerate(paths):
        cost = graph.get_path_cost(path)
        print(f"Path {i+1}: {path}, Cost: {cost}")

def ml_prediction_example():
    """ML prediction example"""
    print("\nML Traffic Prediction Example:")
    print("=" * 35)

    from optical_networks.algorithms.ml_predictors import TrafficPredictor

    # Create predictor
    predictor = TrafficPredictor()

    # Generate sample historical data
    historical_data = []
    for hour in range(100):
        # Simulate daily traffic pattern
        base_traffic = 50e9 + 30e9 * np.sin(2 * np.pi * (hour % 24) / 24)
        noise = np.random.normal(0, 5e9)
        traffic = max(10e9, base_traffic + noise)
        historical_data.append((hour, traffic))

    # Train the model
    predictor.train(historical_data)

    # Make predictions
    print("Making traffic predictions for next 6 hours:")
    for hour in range(100, 106):
        historical_dict = {h: t for h, t in historical_data}
        prediction = predictor.predict(hour, historical_dict)
        print(f"Hour {hour}: Predicted traffic = {prediction/1e9:.1f} Gbps")

    print("\nML example completed!")

if __name__ == "__main__":
    basic_routing_example()
    ml_prediction_example()
    comprehensive_example()