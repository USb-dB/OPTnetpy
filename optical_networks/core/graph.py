import networkx as nx
import heapq
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict, Optional
from .spectrum import SpectrumBand


class RedundantGraph:
    """Enhanced redundant graph with spectrum management and WDM support"""

    def __init__(self, vertices: int, edges: List[Tuple], k: int = 2):
        self.V = vertices
        self.k = k
        self.original_graph = edges.copy()
        self.graph = edges.copy()
        self.pos = None
        self.redundant_graph_edges = []
        self.spectrum_band = None
        self.path_spectrum_map = {}
        self.link_spectrum_usage = {}
        self.connection_requests = []
        self.performance_metrics = {
            'blocked_requests': 0,
            'total_requests': 0,
            'spectrum_utilization': [],
            'fragmentation': []
        }

    def find(self, parent: List[int], i: int) -> int:
        """Find with path compression"""
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent: List[int], rank: List[int], x: int, y: int):
        """Union by rank"""
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1

    def get_networkx_graph(self) -> nx.Graph:
        """Get NetworkX graph representation"""
        G = nx.Graph()
        for u, v, w in self.redundant_graph_edges:
            G.add_edge(u, v, weight=w)
        return G

    def validate_path(self, path: List[int]) -> bool:
        """Validate if a path exists in the graph"""
        G = self.get_networkx_graph()
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False
        return True

    def get_path_cost(self, path: List[int]) -> float:
        """Calculate total cost of a path"""
        G = self.get_networkx_graph()
        cost = 0
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                cost += G[path[i]][path[i + 1]]['weight']
            else:
                return float('inf')
        return cost

    def add_redundant_paths(self) -> float:
        """Build MST and add redundant paths with enhanced logging"""
        # Sort edges by weight
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = list(range(self.V))
        rank = [0] * self.V
        result = []
        e = 0

        # Kruskal's algorithm for MST
        for u, v, w in self.graph:
            if e < self.k * (self.V - 1):
                x = self.find(parent, u)
                y = self.find(parent, v)

                if e < (self.V - 1) and x != y:
                    result.append([u, v, w])
                    self.union(parent, rank, x, y)
                    e += 1
                elif e >= (self.V - 1):
                    # Add redundant edges
                    result.append([u, v, w])
                    e += 1
            else:
                break

        total_cost = float(sum(weight for _, _, weight in result))  # Ensure float
        self.redundant_graph_edges = result

        print(f"Redundant Graph Construction Complete:")
        print(f"  - Total edges: {len(result)}")
        print(f"  - Total cost: {total_cost}")
        print(f"  - Redundancy factor: {self.k}")

        return total_cost

    def dijkstra_shortest_paths(self, src: int, dest: int, n: int = 1) -> List[List[int]]:
        """Find n shortest paths using Dijkstra's algorithm"""
        graph = nx.Graph()
        for u, v, w in self.redundant_graph_edges:
            graph.add_edge(u, v, weight=w)

        pq = [(0, src, [src])]  # (cost, current_node, path)
        visited = set()
        paths = []

        while pq and len(paths) < n:
            cost, node, path = heapq.heappop(pq)

            if (node, tuple(path)) in visited:
                continue
            visited.add((node, tuple(path)))

            if node == dest:
                paths.append(path)
                continue

            for neighbor in graph.neighbors(node):
                if neighbor not in path:
                    new_cost = cost + graph[node][neighbor]['weight']
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, neighbor, new_path))

        return paths

    def allocate_spectrum(self, no_of_slots: int, bandwidth: float):
        """Initialize spectrum band"""
        self.spectrum_band = SpectrumBand(no_of_slots, bandwidth)
        print(f"Initialized {no_of_slots} spectrum slots with total bandwidth {bandwidth / 1e9} GHz")

    def links_in_path(self, path: List[int]) -> List[Tuple[int, int]]:
        """Extract links from path"""
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def assign_spectrum(self, path: List[int], required_slots: int,
                        strategy: str = "first-fit") -> bool:
        """Assign spectrum to path with conflict checking"""
        if not self.spectrum_band:
            raise ValueError("Spectrum band not initialized")

        self.performance_metrics['total_requests'] += 1

        path_links = self.links_in_path(path)

        # Check for spectrum conflicts
        shared_links = {link for link in path_links if link in self.link_spectrum_usage}
        if shared_links:
            print(f"Path {path} shares links {shared_links} with existing allocations")
            self.performance_metrics['blocked_requests'] += 1
            return False

        # Find and allocate slots
        start, end = self.spectrum_band.find_contiguous_slots(required_slots, strategy)

        if start is not None and end is not None:
            connection_id = f"conn_{len(self.path_spectrum_map)}"
            self.spectrum_band.allocate_slots(start, end, connection_id)
            self.path_spectrum_map[tuple(path)] = (start, end, connection_id)

            # Update link usage
            for link in path_links:
                self.link_spectrum_usage[link] = (start, end, connection_id)

            # Update metrics
            utilization = self.spectrum_band.get_utilization()
            fragmentation = self.spectrum_band.get_fragmentation()
            self.performance_metrics['spectrum_utilization'].append(utilization)
            self.performance_metrics['fragmentation'].append(fragmentation)

            slot_range = (
                self.spectrum_band.slots[start].start_frequency,
                self.spectrum_band.slots[end].end_frequency
            )

            print(f"✓ Spectrum allocated to path {path}: Slots {start}-{end}, "
                  f"Bandwidth {slot_range[1] - slot_range[0]:.2f} Hz")
            return True
        else:
            print(f"✗ Failed to allocate spectrum to path {path}. Not enough contiguous slots.")
            self.performance_metrics['blocked_requests'] += 1
            return False

    def release_spectrum(self, path: List[int]):
        """Release spectrum allocated to a path"""
        path_tuple = tuple(path)
        if path_tuple in self.path_spectrum_map:
            start, end, connection_id = self.path_spectrum_map[path_tuple]
            self.spectrum_band.release_slots(start, end)
            del self.path_spectrum_map[path_tuple]

            # Update link usage
            path_links = self.links_in_path(path)
            for link in path_links:
                if link in self.link_spectrum_usage:
                    del self.link_spectrum_usage[link]

            print(f"✓ Spectrum released for path {path}: Slots {start}-{end}")
        else:
            print(f"✗ No spectrum allocated to path {path}, nothing to release")

    def handle_connection_requests(self, requests: List[Tuple]):
        """Handle multiple connection requests"""
        for src, dest, bandwidth, duration in requests:
            print(f"Processing request: {src} → {dest}, BW: {bandwidth / 1e9} GHz, Duration: {duration}")

            required_slots = int(bandwidth / self.spectrum_band.slot_width)
            paths = self.dijkstra_shortest_paths(src, dest, n=1)

            if paths:
                path = paths[0]
                if self.assign_spectrum(path, required_slots):
                    self.connection_requests.append((path, duration))
            else:
                print(f"✗ No path found from {src} to {dest}")

    def decrement_durations(self):
        """Decrement durations and release expired connections"""
        completed_requests = []

        for idx, (path, duration) in enumerate(self.connection_requests):
            if duration > 1:
                self.connection_requests[idx] = (path, duration - 1)
            else:
                completed_requests.append(path)

        for path in completed_requests:
            self.release_spectrum(path)

        self.connection_requests = [
            req for req in self.connection_requests
            if req[0] not in completed_requests
        ]

        if completed_requests:
            print(f"Released {len(completed_requests)} expired connections")

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if self.performance_metrics['total_requests'] > 0:
            blocking_probability = (
                    self.performance_metrics['blocked_requests'] /
                    self.performance_metrics['total_requests']
            )
        else:
            blocking_probability = 0

        return {
            'blocking_probability': blocking_probability,
            'total_requests': self.performance_metrics['total_requests'],
            'blocked_requests': self.performance_metrics['blocked_requests'],
            'current_utilization': self.spectrum_band.get_utilization() if self.spectrum_band else 0,
            'current_fragmentation': self.spectrum_band.get_fragmentation() if self.spectrum_band else 0,
            'active_connections': len(self.connection_requests)
        }

    def plot_graphs(self, src: Optional[int] = None, dest: Optional[int] = None,
                    n: int = 1, required_slots: int = 10):
        """Enhanced visualization with spectrum information"""
        original_G = nx.Graph()
        redundant_G = nx.Graph()

        for u, v, w in self.original_graph:
            original_G.add_edge(u, v, weight=w)
        for u, v, w in self.redundant_graph_edges:
            redundant_G.add_edge(u, v, weight=w)

        if self.pos is None:
            self.pos = nx.spring_layout(original_G, k=0.3, seed=42)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot original graph
        nx.draw_networkx_nodes(original_G, self.pos, node_color='skyblue',
                               node_size=500, ax=ax1)
        nx.draw_networkx_edges(original_G, self.pos, ax=ax1)
        nx.draw_networkx_labels(original_G, self.pos, font_size=10,
                                font_weight='bold', ax=ax1)
        edge_labels_original = nx.get_edge_attributes(original_G, 'weight')
        nx.draw_networkx_edge_labels(original_G, self.pos,
                                     edge_labels=edge_labels_original, ax=ax1)
        ax1.set_title("Original Graph")
        ax1.axis('off')

        # Plot redundant graph with spectrum info
        nx.draw_networkx_nodes(redundant_G, self.pos, node_color='lightgreen',
                               node_size=500, ax=ax2)
        nx.draw_networkx_edges(redundant_G, self.pos, ax=ax2)
        nx.draw_networkx_labels(redundant_G, self.pos, font_size=10,
                                font_weight='bold', ax=ax2)

        # Highlight source and destination if provided
        if src is not None and dest is not None:
            nx.draw_networkx_nodes(redundant_G, self.pos, nodelist=[src],
                                   node_color='orange', node_size=700, ax=ax2)
            nx.draw_networkx_nodes(redundant_G, self.pos, nodelist=[dest],
                                   node_color='purple', node_size=700, ax=ax2)

            # Find and highlight paths
            paths = self.dijkstra_shortest_paths(src, dest, n)
            colors = itertools.cycle(['red', 'blue', 'purple', 'orange', 'green', 'brown'])

            for idx, path in enumerate(paths):
                path_edges = list(zip(path[:-1], path[1:]))
                color = next(colors)
                path_weight = sum(redundant_G[u][v]['weight'] for u, v in path_edges)

                nx.draw_networkx_edges(
                    redundant_G, self.pos, edgelist=path_edges, width=3,
                    edge_color=color, style="dashed", ax=ax2,
                    label=f"Path {idx + 1} (Weight: {path_weight})"
                )

        ax2.set_title("Redundant Graph with Spectrum Allocation")
        ax2.axis('off')
        ax2.legend()

        # Add spectrum utilization info
        if self.spectrum_band:
            metrics = self.get_performance_metrics()
            info_text = (
                f"Spectrum Utilization: {metrics['current_utilization']:.1f}%\n"
                f"Fragmentation: {metrics['current_fragmentation']:.1f}%\n"
                f"Active Connections: {metrics['active_connections']}\n"
                f"Blocking Probability: {metrics['blocking_probability']:.3f}"
            )
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()