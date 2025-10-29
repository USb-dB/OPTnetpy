import time
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from ..core.graph import RedundantGraph
from ..core.wdm_channels import WDMChannelManager
from ..algorithms.ml_predictors import MLPredictiveAllocator


class OpticalNetworkSimulator:
    """Main simulation engine for optical networks"""

    def __init__(self, nodes: int = 10, channels: int = 40,
                 total_bandwidth: float = 200e9, k: int = 2):
        self.nodes = nodes
        self.channels = channels
        self.total_bandwidth = total_bandwidth
        self.k = k
        self.current_time = 0
        self.simulation_history = []

        # Initialize components
        self.graph = self._initialize_graph()
        self.wdm_manager = WDMChannelManager()
        self.ml_allocator = MLPredictiveAllocator(self.graph.spectrum_band)

    def _initialize_graph(self) -> RedundantGraph:
        """Initialize network graph with realistic topology"""
        # Create a more realistic network topology
        edges = []
        # Create ring backbone
        for i in range(self.nodes):
            edges.append((i, (i + 1) % self.nodes, np.random.randint(1, 10)))

        # Add some random cross connections
        for _ in range(self.nodes // 2):
            i, j = np.random.choice(self.nodes, 2, replace=False)
            if (i, j, 0) not in edges and (j, i, 0) not in edges:
                edges.append((i, j, np.random.randint(1, 15)))

        graph = RedundantGraph(self.nodes, edges, self.k)
        graph.add_redundant_paths()
        graph.allocate_spectrum(no_of_slots=self.channels, bandwidth=self.total_bandwidth)

        return graph

    def generate_traffic(self, duration: int) -> List[Tuple]:
        """Generate realistic traffic patterns"""
        requests = []
        time_between_requests = duration / (self.nodes * 2)

        for i in range(int(self.nodes * 2)):
            src, dest = np.random.choice(self.nodes, 2, replace=False)
            bandwidth = np.random.choice([10e9, 40e9, 100e9], p=[0.6, 0.3, 0.1])
            req_duration = np.random.randint(1, 20)
            arrival_time = i * time_between_requests

            requests.append((src, dest, bandwidth, req_duration, arrival_time))

        return requests

    def run_simulation(self, duration: int = 1000, show_progress: bool = True):
        """Run the main simulation"""
        print(f"Starting Optical Network Simulation")
        print(f"Duration: {duration} time units")
        print(f"Network: {self.nodes} nodes, {self.channels} channels")
        print(f"Total bandwidth: {self.total_bandwidth / 1e9} GHz")
        print("=" * 50)

        # Generate traffic
        traffic_requests = self.generate_traffic(duration)

        # Simulation loop
        if show_progress:
            pbar = tqdm(total=duration, desc="Simulation Progress")

        for current_time in range(duration):
            self.current_time = current_time

            # Process requests arriving at this time
            current_requests = [
                (src, dest, bw, dur) for src, dest, bw, dur, arrival in traffic_requests
                if arrival <= current_time
            ]

            if current_requests:
                self.graph.handle_connection_requests(current_requests)

            # Decrement durations
            self.graph.decrement_durations()

            # Update ML predictor
            total_traffic = sum(bw for _, _, bw, _ in current_requests) if current_requests else 0
            self.ml_allocator.update_traffic_history(current_time, total_traffic)

            # Record metrics
            metrics = self.graph.get_performance_metrics()
            metrics['time'] = current_time
            self.simulation_history.append(metrics)

            if show_progress:
                pbar.update(1)
                pbar.set_postfix({
                    'Util': f"{metrics['current_utilization']:.1f}%",
                    'Blocking': f"{metrics['blocking_probability']:.3f}"
                })

        if show_progress:
            pbar.close()

        self._print_simulation_summary()

    def _print_simulation_summary(self):
        """Print comprehensive simulation results"""
        print("\n" + "=" * 50)
        print("SIMULATION SUMMARY")
        print("=" * 50)

        final_metrics = self.simulation_history[-1] if self.simulation_history else {}

        print(f"Total Simulation Time: {self.current_time} units")
        print(f"Total Connection Requests: {final_metrics.get('total_requests', 0)}")
        print(f"Blocked Requests: {final_metrics.get('blocked_requests', 0)}")
        print(f"Final Blocking Probability: {final_metrics.get('blocking_probability', 0):.4f}")
        print(f"Final Spectrum Utilization: {final_metrics.get('current_utilization', 0):.2f}%")
        print(f"Final Spectrum Fragmentation: {final_metrics.get('current_fragmentation', 0):.2f}%")
        print(f"Active Connections: {final_metrics.get('active_connections', 0)}")

        # Calculate average utilization
        avg_utilization = np.mean([m.get('current_utilization', 0) for m in self.simulation_history])
        print(f"Average Spectrum Utilization: {avg_utilization:.2f}%")

    def plot_performance_metrics(self):
        """Plot simulation performance metrics over time"""
        import matplotlib.pyplot as plt

        times = [m['time'] for m in self.simulation_history]
        utilization = [m['current_utilization'] for m in self.simulation_history]
        fragmentation = [m['current_fragmentation'] for m in self.simulation_history]
        blocking = [m['blocking_probability'] for m in self.simulation_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Utilization plot
        ax1.plot(times, utilization, 'b-', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Spectrum Utilization (%)')
        ax1.set_title('Spectrum Utilization Over Time')
        ax1.grid(True, alpha=0.3)

        # Fragmentation plot
        ax2.plot(times, fragmentation, 'r-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fragmentation (%)')
        ax2.set_title('Spectrum Fragmentation Over Time')
        ax2.grid(True, alpha=0.3)

        # Blocking probability plot
        ax3.plot(times, blocking, 'g-', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Blocking Probability')
        ax3.set_title('Connection Blocking Probability Over Time')
        ax3.grid(True, alpha=0.3)

        # Active connections
        active_conns = [m['active_connections'] for m in self.simulation_history]
        ax4.plot(times, active_conns, 'purple', linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Active Connections')
        ax4.set_title('Active Connections Over Time')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()