import torch
import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing mass balance for current routing.py...")

try:
    from neuralhydrology.modelzoo.routing import RoutingModule

    # Create test connectivity data
    connectivity_data = [
        {
            "upstream_basin_id": "basin_1",
            "downstream_basin_id": "basin_3",
            "weight": 1.0,
        },
        {
            "upstream_basin_id": "basin_2",
            "downstream_basin_id": "basin_3",
            "weight": 1.0,
        },
    ]

    # Save to CSV
    pd.DataFrame(connectivity_data).to_csv("test_connectivity.csv", index=False)

    # Create configuration
    class Config:
        basin_ids = ["basin_1", "basin_2", "basin_3"]
        routing_connectivity_file = "test_connectivity.csv"
        device = torch.device("cpu")
        routing_method = "route_none"

    # Initialize routing module
    routing = RoutingModule(Config())

    # Test data: 1 batch, 4 time steps, 3 basins
    lateral_flows = torch.tensor(
        [
            [
                [1.0, 2.0, 0.5],  # Time 1: total = 3.5
                [1.5, 1.8, 0.3],  # Time 2: total = 3.6
                [2.0, 1.2, 0.8],  # Time 3: total = 4.0
                [0.8, 1.5, 0.2],  # Time 4: total = 2.5
            ]
        ],
        dtype=torch.float32,
    )

    print("Input lateral flows shape:", lateral_flows.shape)
    print("Input lateral flows:")
    for t in range(lateral_flows.shape[1]):
        time_flows = lateral_flows[0, t, :]
        print(
            f"  Time {t+1}: {time_flows.tolist()} (sum: {time_flows.sum().item():.1f})"
        )

    # Run routing
    results = routing.forward(lateral_flows)

    print("\nRouting results:")
    print("Routed flows shape:", results["routed_flows"].shape)

    # Check mass balance
    total_input = lateral_flows.sum().item()
    total_output = results["outlet_flow"].sum().item()
    mass_balance_error = total_input - total_output

    print(f"\nMass Balance Check:")
    print(f"Total input: {total_input:.1f}")
    print(f"Total output at outlet: {total_output:.1f}")
    print(f"Mass balance error: {mass_balance_error:.6f}")

    # Check time-by-time balance
    print(f"\nTime-by-time analysis:")
    for t in range(lateral_flows.shape[1]):
        time_input = lateral_flows[0, t, :].sum().item()
        time_output = results["outlet_flow"][0, t, 0].item()
        time_error = time_input - time_output
        print(
            f"  Time {t+1}: Input={time_input:.1f}, Output={time_output:.1f}, Error={time_error:.6f}"
        )

    # Detailed flow component analysis
    print(f"\nFlow Components Analysis:")
    print(f"Lateral flow: {results['lateral_flow']}")
    print(f"Upstream inflow: {results['upstream_inflow']}")
    print(f"Channel storage: {results['channel_storage']}")
    print(f"Channel outflow: {results['channel_outflow']}")
    print(f"Total outflow: {results['routed_flows']}")

    if abs(mass_balance_error) < 1e-6:
        print(f"\n✓ SUCCESS: Mass balance is correct!")
    else:
        print(f"\n✗ FAILURE: Mass balance error = {mass_balance_error:.6f}")

        # Additional debugging info
        print(f"\nDEBUG INFO:")
        print(f"Basin ordering: {routing.basin_ids}")
        print(f"Outlet basin: {routing.basin_ids[routing.outlet_idx]}")
        print(f"Connectivity matrix:\n{routing.connectivity_matrix}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

# Clean up
try:
    os.remove("test_connectivity.csv")
except:
    pass
