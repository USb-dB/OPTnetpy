"""
WDM-specific simulation example
Author: Prantik Basu
"""

import numpy as np
from optical_networks.core.wdm_channels import WDMChannelManager

def wdm_simulation_example():
    print("WDM Network Simulation Example")
    print("=" * 40)

    # Create WDM system
    wdm_system = WDMChannelManager(
        start_frequency=191.3e12,
        end_frequency=196.1e12,
        channel_spacing=50e9,
        channel_bandwidth=37.5e9
    )

    print(f"Initialized WDM system with {len(wdm_system.channels)} channels")
    print(f"Frequency range: {wdm_system.start_frequency/1e12:.1f} - {wdm_system.end_frequency/1e12:.1f} THz")
    print(f"Channel spacing: {wdm_system.channel_spacing/1e9} GHz")
    print(f"Channel bandwidth: {wdm_system.channel_bandwidth/1e9} GHz")

    # Simulate channel allocations with different transmission distances
    connections = [
        ([0, 1, 2], -2.0, 80.0),   # channels, power (dBm), distance (km)
        ([5, 6], -1.5, 120.0),
        ([10, 11, 12, 13], -3.0, 200.0)
    ]

    for channel_ids, power, distance in connections:
        wdm_system.allocate_channels(channel_ids, power, distance)
        print(f"Allocated channels {channel_ids} with power {power} dBm, distance {distance} km")

    # Calculate performance metrics
    utilization = wdm_system.get_channel_utilization()
    system_performance = wdm_system.get_system_performance()

    print(f"\nWDM System Performance:")
    print(f"Allocated channels: {system_performance['allocated_channels']}/{system_performance['total_channels']}")
    print(f"Channel utilization: {utilization:.2f}%")
    print(f"Average OSNR: {system_performance['avg_osnr']:.2f} dB")
    print(f"Average BER: {system_performance['avg_ber']:.2e}")
    print(f"Average Q-factor: {system_performance['avg_q_factor']:.2f}")

    # Demonstrate BER calculations for specific channels
    print("\nChannel Quality Analysis:")
    for channel in wdm_system.channels[:8]:  # Show first 8 channels
        if channel.is_allocated:
            ber_qpsk = channel.calculate_ber(channel.osnr, modulation="QPSK")
            ber_16qam = channel.calculate_ber(channel.osnr, modulation="16QAM")
            q_factor = channel.calculate_q_factor(ber_qpsk)

            print(f"Channel {channel.channel_id}: "
                  f"OSNR={channel.osnr:.1f}dB, "
                  f"QPSK-BER={ber_qpsk:.2e}, "
                  f"16QAM-BER={ber_16qam:.2e}, "
                  f"Q-factor={q_factor:.2f}")

    # Demonstrate finding best channels
    print("\nFinding Best Available Channels:")
    best_channels = wdm_system.find_best_channels(num_channels=3, min_osnr=15.0)
    if best_channels:
        print(f"Found {len(best_channels)} best available channels:")
        for channel in best_channels:
            print(f"  Channel {channel.channel_id}: Estimated OSNR={channel.osnr:.1f}dB")
    else:
        print("No suitable channels found with minimum OSNR requirement")

    # Test channel release
    print("\nTesting Channel Release:")
    wdm_system.release_channels([0, 1])  # Release first two channels
    utilization_after_release = wdm_system.get_channel_utilization()
    print(f"Utilization after releasing channels 0 and 1: {utilization_after_release:.2f}%")

def advanced_wdm_example():
    """Advanced WDM example with performance analysis"""
    print("\n" + "="*50)
    print("Advanced WDM Performance Analysis")
    print("="*50)

    # Create a dense WDM system
    wdm_system = WDMChannelManager(
        start_frequency=191.3e12,
        end_frequency=196.1e12,
        channel_spacing=25e9,  # Denser spacing
        channel_bandwidth=20e9  # Narrower bandwidth
    )

    print(f"Dense WDM System: {len(wdm_system.channels)} channels")
    print(f"Channel spacing: {wdm_system.channel_spacing/1e9} GHz")

    # Simulate different modulation formats
    modulation_formats = ["QPSK", "16QAM", "64QAM"]
    transmission_distances = [50.0, 100.0, 200.0]  # km

    print("\nModulation Format Performance Comparison:")
    print("Format | Distance | OSNR | BER")
    print("-" * 40)

    for modulation in modulation_formats:
        for distance in transmission_distances:
            # Test a sample channel
            test_channel = wdm_system.channels[0]
            test_channel.power = 0.0  # dBm
            osnr = test_channel.calculate_osnr_from_power(distance)
            ber = test_channel.calculate_ber(osnr, modulation)

            print(f"{modulation:6} | {distance:8.0f} | {osnr:4.1f} | {ber:.2e}")

if __name__ == "__main__":
    wdm_simulation_example()
    advanced_wdm_example()