import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as mpatches


class NetworkVisualizer:
    """Advanced network visualization with spectrum information"""

    def __init__(self, graph):
        self.graph = graph
        self.fig = None
        self.ax = None

    def plot_network_topology(self, highlight_paths: List[List[int]] = None,
                              show_spectrum: bool = True, figsize: Tuple = (15, 10)):
        """Plot network topology with optional path highlighting"""
        self.fig, self.ax = plt.subplots(figsize=figsize)

        G = nx.Graph()
        for u, v, w in self.graph.redundant_graph_edges:
            G.add_edge(u, v, weight=w)

        if self.graph.pos is None:
            self.graph.pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, self.graph.pos, node_color='lightblue',
                               node_size=800, alpha=0.9, ax=self.ax)

        # Draw edges
        nx.draw_networkx_edges(G, self.graph.pos, alpha=0.6, edge_color='gray',
                               width=2, ax=self.ax)

        # Draw labels
        nx.draw_networkx_labels(G, self.graph.pos, font_size=12,
                                font_weight='bold', ax=self.ax)

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.graph.pos, edge_labels=edge_labels,
                                     font_size=8, ax=self.ax)

        # Highlight paths if provided
        if highlight_paths:
            colors = plt.cm.Set3(np.linspace(0, 1, len(highlight_paths)))

            for i, path in enumerate(highlight_paths):
                path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                nx.draw_networkx_edges(G, self.graph.pos, edgelist=path_edges,
                                       width=4, alpha=0.8, edge_color=colors[i:i + 1],
                                       style='dashed', ax=self.ax)

                # Calculate path metrics
                path_cost = sum(G[u][v]['weight'] for u, v in path_edges)
                path_hop = len(path) - 1

                print(f"Path {i + 1}: {path}, Cost: {path_cost}, Hops: {path_hop}")

        # Add spectrum information if available and requested
        if show_spectrum and hasattr(self.graph, 'spectrum_band') and self.graph.spectrum_band:
            self._add_spectrum_info()

        self.ax.set_title("Optical Network Topology", fontsize=16, fontweight='bold')
        self.ax.axis('off')
        plt.tight_layout()

    def _add_spectrum_info(self):
        """Add spectrum utilization information to the plot"""
        spectrum_band = self.graph.spectrum_band
        metrics = self.graph.get_performance_metrics()

        info_text = (
            f"Spectrum Utilization: {metrics['current_utilization']:.1f}%\n"
            f"Fragmentation: {metrics['current_fragmentation']:.1f}%\n"
            f"Active Connections: {metrics['active_connections']}\n"
            f"Blocking Probability: {metrics['blocking_probability']:.3f}\n"
            f"Total Slots: {spectrum_band.num_slots}\n"
            f"Slot Width: {spectrum_band.slot_width / 1e9:.1f} GHz"
        )

        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

    def plot_connection_requests(self, requests: List[Tuple], time_window: int = 50):
        """Plot connection requests over time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot request arrival pattern
        times = [req[4] for req in requests if len(req) > 4]  # arrival times
        bandwidths = [req[2] / 1e9 for req in requests]  # bandwidth in GHz

        ax1.hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Arrival Time')
        ax1.set_ylabel('Number of Requests')
        ax1.set_title('Connection Request Arrival Pattern')
        ax1.grid(True, alpha=0.3)

        # Plot bandwidth distribution
        ax2.hist(bandwidths, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Bandwidth (GHz)')
        ax2.set_ylabel('Number of Requests')
        ax2.set_title('Bandwidth Requirement Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_plot(self, filename: str, dpi: int = 300):
        """Save the current plot to file"""
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved as {filename}")