import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings


class TrafficPredictor:
    """ML-based traffic prediction for proactive spectrum allocation"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['hour', 'day_of_week', 'historical_avg', 'seasonal_factor']

    def extract_features(self, timestamp: float, historical_data: Dict) -> np.ndarray:
        """Extract time-based and historical features"""
        # Simple feature extraction - can be enhanced
        hour = timestamp % 24
        day_of_week = (timestamp // 24) % 7
        historical_avg = np.mean(list(historical_data.values())) if historical_data else 0
        seasonal_factor = np.sin(2 * np.pi * hour / 24)  # Daily seasonality

        return np.array([[hour, day_of_week, historical_avg, seasonal_factor]])

    def train(self, historical_data: List[Tuple[float, float]]):
        """Train the prediction model"""
        if len(historical_data) < 10:
            warnings.warn("Insufficient data for training. Using default predictions.")
            return

        X = []
        y = []

        for i in range(len(historical_data) - 1):
            timestamp, traffic = historical_data[i]
            features = self.extract_features(timestamp,
                                             {ts: val for ts, val in historical_data[:i]})
            X.append(features[0])
            y.append(historical_data[i + 1][1])

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, timestamp: float, historical_data: Dict) -> float:
        """Predict future traffic demand"""
        if not self.is_trained:
            # Return simple average if not trained
            return np.mean(list(historical_data.values())) if historical_data else 1.0

        features = self.extract_features(timestamp, historical_data)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        return max(0, prediction)  # Ensure non-negative


class MLPredictiveAllocator:
    """ML-enhanced spectrum allocation with predictive capabilities"""

    def __init__(self, spectrum_band):
        self.spectrum_band = spectrum_band
        self.traffic_predictor = TrafficPredictor()
        self.historical_traffic = {}
        self.prediction_horizon = 24  # hours

    def update_traffic_history(self, timestamp: float, traffic: float):
        """Update historical traffic data"""
        self.historical_traffic[timestamp] = traffic

        # Retrain model periodically
        if len(self.historical_traffic) % 100 == 0:
            training_data = list(self.historical_traffic.items())
            self.traffic_predictor.train(training_data)

    def predict_spectrum_demand(self, current_time: float) -> int:
        """Predict spectrum slots needed based on traffic prediction"""
        future_traffic = []

        for i in range(self.prediction_horizon):
            future_time = current_time + i
            prediction = self.traffic_predictor.predict(future_time, self.historical_traffic)
            future_traffic.append(prediction)

        avg_predicted_traffic = np.mean(future_traffic)

        # Convert traffic to spectrum slots (simplified)
        slot_width = self.spectrum_band.slot_width
        required_slots = max(1, int(avg_predicted_traffic / slot_width))

        return min(required_slots, self.spectrum_band.num_slots)

    def proactive_allocation(self, current_time: float) -> Tuple[Optional[int], Optional[int]]:
        """Perform proactive spectrum allocation based on predictions"""
        required_slots = self.predict_spectrum_demand(current_time)
        return self.spectrum_band.find_contiguous_slots(required_slots, strategy="first-fit")