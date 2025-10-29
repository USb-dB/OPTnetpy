import heapq
import networkx as nx
from typing import List, Tuple, Dict, Optional
import numpy as np
import copy


class DijkstraRouter:
    """Enhanced Dijkstra's algorithm for optical network routing"""

    def __init__(self, graph):
        self.graph = graph
        self.visited_paths = set()

    def find_shortest_paths(self, src: int, dest: int, n: int = 1,
                            constraints: Dict = None) -> List[List[int]]:
        """Find n shortest paths with optional constraints"""
        if constraints is None:
            constraints = {}

        # Create networkx graph from current edges
        G = nx.Graph()
        for u, v, w in self.graph.redundant_graph_edges:
            G.add_edge(u, v, weight=w)

        # Check if source and destination are in graph
        if src not in G or dest not in G:
            return []

        pq = [(0, src, [src])]  # (cost, current_node, path)
        paths = []
        visited = set()

        while pq and len(paths) < n:
            cost, node, path = heapq.heappop(pq)

            # Check for cycles using path tuple
            path_tuple = tuple(path)
            if (node, path_tuple) in visited:
                continue
            visited.add((node, path_tuple))

            if node == dest:
                if self._satisfies_constraints(path, constraints):
                    paths.append(path)
                continue

            for neighbor in G.neighbors(node):
                if neighbor not in path:  # Simple cycle prevention
                    new_cost = cost + G[node][neighbor]['weight']
                    new_path = path + [neighbor]

                    # Check path feasibility before adding to queue
                    if self._is_path_feasible(new_path, constraints):
                        heapq.heappush(pq, (new_cost, neighbor, new_path))

        return paths

    def _satisfies_constraints(self, path: List[int], constraints: Dict) -> bool:
        """Check if path satisfies all constraints"""
        if 'max_hops' in constraints:
            if len(path) - 1 > constraints['max_hops']:
                return False

        if 'excluded_nodes' in constraints:
            if any(node in constraints['excluded_nodes'] for node in path):
                return False

        if 'required_nodes' in constraints:
            if not all(node in path for node in constraints['required_nodes']):
                return False

        return True

    def _is_path_feasible(self, path: List[int], constraints: Dict) -> bool:
        """Check if current path is feasible given constraints"""
        return True


class KSPRouter:
    """K-Shortest Paths algorithm for diverse routing - FIXED VERSION"""

    def __init__(self, graph):
        self.graph = graph

    def find_k_shortest_paths(self, src: int, dest: int, k: int = 3) -> List[List[int]]:
        """Find k shortest paths using Yen's algorithm - FIXED"""
        # First, find the shortest path
        router = DijkstraRouter(self.graph)
        shortest_paths = router.find_shortest_paths(src, dest, n=1)

        if not shortest_paths:
            return []

        A = [shortest_paths[0]]  # k-shortest paths
        B = []  # candidate paths

        # Store original graph state
        original_edges = self.graph.redundant_graph_edges.copy()

        for ki in range(1, k):
            # For each node in the previous shortest path (except destination)
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i + 1]

                # Create a temporary graph without edges used in root path
                temp_edges = self._create_temp_graph(root_path, A)

                # Find spur path from spur node to destination
                temp_graph = RedundantGraph(self.graph.V, temp_edges)
                temp_router = DijkstraRouter(temp_graph)
                spur_paths = temp_router.find_shortest_paths(spur_node, dest, n=1)

                if spur_paths:
                    total_path = root_path[:-1] + spur_paths[0]

                    # Check if this path is valid and not already found
                    if (self._is_valid_path(total_path, original_edges) and
                            total_path not in A and total_path not in B):
                        path_cost = self._calculate_path_cost(total_path, original_edges)
                        B.append((path_cost, total_path))

            if not B:
                break

            # Select the shortest path from B
            B.sort(key=lambda x: x[0])
            best_cost, best_path = B.pop(0)
            A.append(best_path)

        return A

    def _create_temp_graph(self, root_path: List[int], existing_paths: List[List[int]]) -> List[Tuple]:
        """Create temporary graph by removing edges that would create duplicates"""
        # Start with all original edges
        temp_edges = self.graph.redundant_graph_edges.copy()

        # Remove edges from root path that are in existing paths
        edges_to_remove = []
        for path in existing_paths:
            path_edges = self._get_path_edges(path)
            root_edges = self._get_path_edges(root_path)

            # Remove edges that share nodes with root path
            for edge in path_edges:
                if edge[0] in root_path or edge[1] in root_path:
                    if edge not in edges_to_remove:
                        edges_to_remove.append(edge)

        # Remove the identified edges
        for edge in edges_to_remove:
            temp_edges = [e for e in temp_edges
                          if not ((e[0] == edge[0] and e[1] == edge[1]) or
                                  (e[0] == edge[1] and e[1] == edge[0]))]

        return temp_edges

    def _get_path_edges(self, path: List[int]) -> List[Tuple[int, int]]:
        """Extract edges from a path"""
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def _calculate_path_cost(self, path: List[int], edges: List[Tuple]) -> float:
        """Calculate total cost of a path"""
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        cost = 0
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                cost += G[path[i]][path[i + 1]]['weight']
            else:
                return float('inf')  # Invalid path
        return cost

    def _is_valid_path(self, path: List[int], edges: List[Tuple]) -> bool:
        """Check if path is valid (connected)"""
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False
        return True


class EnhancedDijkstraRouter:
    """More robust Dijkstra implementation with better error handling"""

    def __init__(self, graph):
        self.graph = graph

    def find_shortest_path(self, src: int, dest: int) -> List[int]:
        """Find single shortest path with comprehensive error handling"""
        G = nx.Graph()
        for u, v, w in self.graph.redundant_graph_edges:
            G.add_edge(u, v, weight=w)

        try:
            if src not in G or dest not in G:
                return []

            path = nx.shortest_path(G, source=src, target=dest, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            print(f"Routing error: {e}")
            return []

    def find_all_shortest_paths(self, src: int, dest: int, max_paths: int = 3) -> List[List[int]]:
        """Find multiple shortest paths"""
        G = nx.Graph()
        for u, v, w in self.graph.redundant_graph_edges:
            G.add_edge(u, v, weight=w)

        try:
            if src not in G or dest not in G:
                return []

            # Find all simple paths and sort by length
            all_paths = list(nx.all_simple_paths(G, source=src, target=dest, cutoff=10))

            # Calculate path costs and sort
            path_costs = []
            for path in all_paths:
                cost = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
                path_costs.append((cost, path))

            path_costs.sort(key=lambda x: x[0])

            # Return top paths
            return [path for _, path in path_costs[:max_paths]]

        except Exception as e:
            print(f"Multi-path routing error: {e}")
            return []