import numpy as np
import matplotlib.pyplot as plt

# Load the generated data (normalized to [0,1])
X_data = np.load('X_data.npy')  # Shape: (N, 3, 64, 64)
Y_data = np.load('Y_data.npy')  # Shape: (N, 64, 64)

print(f"X_data shape: {X_data.shape}")
print(f"Y_data shape: {Y_data.shape}")

# Un-scaling constants (from readme.txt or known ranges)
# Channel 0 Q: min=0.0, max=150.0
# Channel 1 k: min=149.0, max=260.0
# Channel 2 h: min=50.0, max=400.0  (approx, since TSV_H_FACTOR=8, H_REF=50)
# Temperature: min=25.0, max=144.0

Q_min, Q_max = 0.0, 150.0
k_min, k_max = 149.0, 260.0
h_min, h_max = 50.0, 400.0  # Approximate
T_min, T_max = 25.0, 144.0

# Example: Inspect and un-scale the first sample
sample_idx = 100
Q_norm = X_data[sample_idx, 0]
k_norm = X_data[sample_idx, 1]
h_norm = X_data[sample_idx, 2]
T_norm = Y_data[sample_idx]

# Un-scale to physical units
Q_grid = Q_norm * (Q_max - Q_min) + Q_min
k_grid = k_norm * (k_max - k_min) + k_min
h_grid = h_norm * (h_max - h_min) + h_min
T_map = T_norm * (T_max - T_min) + T_min

print(f"Sample {sample_idx}: Peak T = {T_map.max():.1f} °C")

# Simplified plot function
def simple_plot_sample(Q, k, h, T, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Sample Visualization", fontsize=13)

    # Power Q
    im = axes[0].imshow(Q, cmap="Blues", origin="upper")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Power Q [W]")

    # Conductivity k
    im = axes[1].imshow(k, cmap="Greens", origin="upper")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Conductivity k [W/(m·K)]")

    # Cooling h
    im = axes[2].imshow(h, cmap="Purples", origin="upper")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Cooling h [W/(m²·K)]")

    # Temperature
    im = axes[3].imshow(T, cmap="hot", origin="upper", vmin=T_min)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Temperature [°C]")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

# Generate plot for the first sample
import os; os.makedirs("figures", exist_ok=True)
simple_plot_sample(Q_grid, k_grid, h_grid, T_map, save_path="figures/sample_100.png")