"""
ML-based Predictive Spectrum Allocation Example
Author: Prantik Basu
"""

import numpy as np
import matplotlib.pyplot as plt
from optical_networks import OpticalNetworkSimulator
from optical_networks.algorithms.ml_predictors import TrafficPredictor


def ml_predictive_example():
    print("ML Predictive Spectrum Allocation Demo")
    print("=" * 50)

    # Create simulator
    simulator = OpticalNetworkSimulator(nodes=6, channels=32, total_bandwidth=160e9)

    # Generate training data with periodic traffic patterns
    print("Generating training data...")
    historical_data = []
    for hour in range(100):  # 100 hours of historical data
        # Create daily pattern with noise
        base_traffic = 50e9 + 30e9 * np.sin(2 * np.pi * (hour % 24) / 24)
        noise = np.random.normal(0, 5e9)
        traffic = max(10e9, base_traffic + noise)
        historical_data.append((hour, traffic))

    # Train the predictor
    predictor = TrafficPredictor()
    predictor.train(historical_data)

    print("Training completed. Making predictions...")

    # Test predictions
    test_hours = range(100, 124)  # Next 24 hours
    actual_traffic = []
    predicted_traffic = []

    for hour in test_hours:
        # Generate actual traffic (similar pattern but with variations)
        base_actual = 50e9 + 30e9 * np.sin(2 * np.pi * (hour % 24) / 24)
        actual = max(10e9, base_actual + np.random.normal(0, 8e9))
        actual_traffic.append(actual)

        # Get prediction
        historical_dict = {h: t for h, t in historical_data}
        prediction = predictor.predict(hour, historical_dict)
        predicted_traffic.append(prediction)

        # Update historical data for next prediction
        historical_data.append((hour, actual))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_hours, [t / 1e9 for t in actual_traffic], 'b-', label='Actual Traffic', linewidth=2)
    plt.plot(test_hours, [t / 1e9 for t in predicted_traffic], 'r--', label='Predicted Traffic', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Traffic (Gbps)')
    plt.title('ML Traffic Prediction vs Actual Traffic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate prediction accuracy
    mape = np.mean(np.abs((np.array(actual_traffic) - np.array(predicted_traffic)) / np.array(actual_traffic))) * 100
    print(f"Prediction Accuracy (MAPE): {100 - mape:.2f}%")

    # Demonstrate proactive allocation
    print("\nProactive Spectrum Allocation Demo:")
    ml_allocator = simulator.ml_allocator

    for hour in test_hours[:6]:  # Show first 6 hours
        predicted_slots = ml_allocator.predict_spectrum_demand(hour)
        start, end = ml_allocator.proactive_allocation(hour)

        if start is not None:
            print(f"Hour {hour}: Predicted {predicted_slots} slots, "
                  f"Proactively allocated slots {start}-{end}")
        else:
            print(f"Hour {hour}: Predicted {predicted_slots} slots, "
                  f"Could not allocate (insufficient contiguous slots)")


if __name__ == "__main__":
    ml_predictive_example()