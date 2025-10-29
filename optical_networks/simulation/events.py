from enum import Enum
from typing import Any, Callable, List
import heapq
from datetime import datetime


class EventType(Enum):
    """Types of simulation events"""
    CONNECTION_REQUEST = 1
    CONNECTION_RELEASE = 2
    SPECTRUM_ALLOCATION = 3
    SPECTRUM_RELEASE = 4
    METRICS_UPDATE = 5
    DEFRAGMENTATION = 6
    ML_RETRAINING = 7


class Event:
    """Simulation event representation"""

    def __init__(self, event_type: EventType, timestamp: float,
                 data: Any = None, callback: Callable = None):
        self.event_type = event_type
        self.timestamp = timestamp
        self.data = data
        self.callback = callback
        self.created_at = datetime.now()

    def __lt__(self, other):
        """Comparison for priority queue"""
        return self.timestamp < other.timestamp

    def __repr__(self):
        return f"Event({self.event_type.name}, t={self.timestamp})"


class EventManager:
    """Event-driven simulation manager"""

    def __init__(self):
        self.event_queue = []
        self.current_time = 0
        self.event_handlers = {}
        self.event_history = []

    def schedule_event(self, event: Event):
        """Schedule an event"""
        heapq.heappush(self.event_queue, event)

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def process_events(self, max_events: int = None):
        """Process events from the queue"""
        processed = 0

        while self.event_queue and (max_events is None or processed < max_events):
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            self.event_history.append(event)

            # Call event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    handler(event)

            # Call event-specific callback
            if event.callback:
                event.callback(event)

            processed += 1

        return processed

    def schedule_connection_request(self, timestamp: float, src: int, dest: int,
                                    bandwidth: float, duration: int):
        """Schedule a connection request event"""
        event_data = {
            'src': src,
            'dest': dest,
            'bandwidth': bandwidth,
            'duration': duration
        }
        event = Event(EventType.CONNECTION_REQUEST, timestamp, event_data)
        self.schedule_event(event)

        # Schedule corresponding release event
        release_time = timestamp + duration
        release_event = Event(EventType.CONNECTION_RELEASE, release_time, event_data)
        self.schedule_event(release_event)

    def schedule_periodic_events(self, event_type: EventType, interval: float,
                                 total_duration: float, data: Any = None):
        """Schedule periodic events"""
        current_time = 0
        while current_time <= total_duration:
            event = Event(event_type, current_time, data)
            self.schedule_event(event)
            current_time += interval

    def get_event_statistics(self) -> dict:
        """Get statistics about processed events"""
        event_counts = {}
        for event in self.event_history:
            event_type = event.event_type.name
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            'total_events': len(self.event_history),
            'event_counts': event_counts,
            'simulation_duration': self.current_time,
            'events_per_time_unit': len(self.event_history) / self.current_time if self.current_time > 0 else 0
        }