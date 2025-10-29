import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class NetworkMetrics:
    """Comprehensive network performance metrics collector"""

    def __init__(self):
        self.metrics_history = []
        self.start_time = datetime.now()

    def record_metrics(self, metrics: Dict):
        """Record metrics with timestamp"""
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)

    def get_utilization_stats(self) -> Dict:
        """Get utilization statistics"""
        utilizations = [m.get('current_utilization', 0) for m in self.metrics_history]

        return {
            'mean_utilization': np.mean(utilizations) if utilizations else 0,
            'max_utilization': np.max(utilizations) if utilizations else 0,
            'min_utilization': np.min(utilizations) if utilizations else 0,
            'std_utilization': np.std(utilizations) if utilizations else 0
        }

    def get_blocking_stats(self) -> Dict:
        """Get blocking probability statistics"""
        blocking_probs = [m.get('blocking_probability', 0) for m in self.metrics_history]

        return {
            'mean_blocking': np.mean(blocking_probs) if blocking_probs else 0,
            'max_blocking': np.max(blocking_probs) if blocking_probs else 0,
            'min_blocking': np.min(blocking_probs) if blocking_probs else 0
        }

    def get_fragmentation_stats(self) -> Dict:
        """Get fragmentation statistics"""
        fragmentations = [m.get('current_fragmentation', 0) for m in self.metrics_history]

        return {
            'mean_fragmentation': np.mean(fragmentations) if fragmentations else 0,
            'max_fragmentation': np.max(fragmentations) if fragmentations else 0,
            'min_fragmentation': np.min(fragmentations) if fragmentations else 0
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        return {
            'simulation_duration': str(datetime.now() - self.start_time),
            'total_metrics_recorded': len(self.metrics_history),
            'utilization': self.get_utilization_stats(),
            'blocking': self.get_blocking_stats(),
            'fragmentation': self.get_fragmentation_stats(),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {}
        }


class ConnectionMetrics:
    """Connection-level performance metrics"""

    def __init__(self):
        self.connections = []
        self.connection_stats = {
            'total_established': 0,
            'total_blocked': 0,
            'total_released': 0
        }

    def record_connection(self, connection_id: str, src: int, dest: int,
                          bandwidth: float, duration: int, path: List[int]):
        """Record connection establishment"""
        connection = {
            'id': connection_id,
            'src': src,
            'dest': dest,
            'bandwidth': bandwidth,
            'duration': duration,
            'path': path,
            'establish_time': datetime.now(),
            'status': 'established'
        }
        self.connections.append(connection)
        self.connection_stats['total_established'] += 1

    def record_blocked_connection(self, src: int, dest: int, bandwidth: float, reason: str):
        """Record blocked connection"""
        connection = {
            'id': f"blocked_{len(self.connections)}",
            'src': src,
            'dest': dest,
            'bandwidth': bandwidth,
            'block_time': datetime.now(),
            'status': 'blocked',
            'reason': reason
        }
        self.connections.append(connection)
        self.connection_stats['total_blocked'] += 1

    def record_release(self, connection_id: str):
        """Record connection release"""
        for conn in self.connections:
            if conn['id'] == connection_id and conn['status'] == 'established':
                conn['release_time'] = datetime.now()
                conn['status'] = 'released'
                conn['actual_duration'] = (conn['release_time'] - conn['establish_time']).total_seconds()
                self.connection_stats['total_released'] += 1
                break

    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        established_conns = [c for c in self.connections if c['status'] == 'established']
        blocked_conns = [c for c in self.connections if c['status'] == 'blocked']
        released_conns = [c for c in self.connections if c['status'] == 'released']

        return {
            'total_connections': len(self.connections),
            'established_connections': len(established_conns),
            'blocked_connections': len(blocked_conns),
            'released_connections': len(released_conns),
            'blocking_rate': len(blocked_conns) / len(self.connections) if self.connections else 0
        }