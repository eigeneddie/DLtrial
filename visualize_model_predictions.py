import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_generator import (
    build_cooling_grid,
    build_material_grid,
    disaggregated_layout,
    fdm_steady_state,
    hub_and_spoke_layout,
)


# -----------------------------------------------------------------------------
# Model definition matching model_trainer.py
# -----------------------------------------------------------------------------

class ThermalSurrogate(nn.Module):
    def __init__(self):
        super(ThermalSurrogate, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.clamp(0, 1)


# -----------------------------------------------------------------------------
# Normalisation constants from readme.txt
# -----------------------------------------------------------------------------

X_SCALES = {
    "Q": (0.0, 299.9977),
    "k": (149.0, 260.0),
    "h": (50.0, 400.0),
}
Y_SCALE = (25.0, 268.1102)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fresh thermal samples and compare model predictions to ground truth."
    )
    parser.add_argument("--count", type=int, default=4, help="Number of fresh samples to generate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sample generation.")
    parser.add_argument("--output-dir", type=str, default="eval_outputs", help="Directory to save comparison figures.")
    parser.add_argument("--model-path", type=str, default="thermal_surrogate.pth", help="Path to the saved surrogate model weights.")
    parser.add_argument("--show", action="store_true", help="Display the figures interactively.")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = ThermalSurrogate().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_input(X_raw: np.ndarray) -> np.ndarray:
    """Min-max normalise one fresh sample to [0, 1] per channel."""
    X_norm = np.zeros_like(X_raw, dtype=np.float32)
    for idx, key in enumerate(("Q", "k", "h")):
        min_val, max_val = X_SCALES[key]
        denom = max_val - min_val
        X_norm[idx] = (X_raw[idx] - min_val) / denom
    return X_norm


def unscale_temperature(T_norm: np.ndarray) -> np.ndarray:
    t_min, t_max = Y_SCALE
    return T_norm * (t_max - t_min) + t_min


def generate_sample(rng: np.random.Generator):
    if rng.random() < 0.5:
        Q_grid, chiplets = hub_and_spoke_layout(rng)
    else:
        Q_grid, chiplets = disaggregated_layout(rng)

    k_grid, material = build_material_grid(rng)
    h_grid = build_cooling_grid(rng)
    T_gt = fdm_steady_state(Q_grid, k_grid, h_grid)

    X_raw = np.stack([Q_grid, k_grid, h_grid], axis=0).astype(np.float32)
    return X_raw, T_gt, k_grid, h_grid, chiplets, material


def plot_comparison(index: int, X_raw: np.ndarray, T_gt: np.ndarray, T_pred: np.ndarray):
    Q_grid = X_raw[0]
    vmin = min(T_gt.min(), T_pred.min())
    vmax = max(T_gt.max(), T_pred.max())
    err_map = np.abs(T_gt - T_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Model Evaluation Sample #{index + 1}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    im = ax.imshow(Q_grid, cmap="Blues", origin="upper")
    ax.set_title("Input Power Q [W]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(T_gt, cmap="hot", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth Temperature [°C]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im = ax.imshow(T_pred, cmap="hot", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title("Predicted Temperature [°C]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(err_map, cmap="inferno", origin="upper")
    ax.set_title("Absolute Error |T_pred - T_gt| [°C]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im = ax.imshow(X_raw[1], cmap="Greens", origin="upper")
    ax.set_title("Conductivity k [W/(m·K)]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    im = ax.imshow(X_raw[2], cmap="Purples", origin="upper")
    ax.set_title("Cooling h [W/(m²·K)]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flatten():
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def evaluate_samples(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    rng = np.random.default_rng(args.seed)
    metrics = []

    for sample_idx in range(args.count):
        X_raw, T_gt, k_grid, h_grid, chiplets, material = generate_sample(rng)
        X_norm = normalize_input(X_raw)

        with torch.no_grad():
            x_tensor = torch.tensor(X_norm[None], dtype=torch.float32, device=device)
            y_pred_norm = model(x_tensor)[0, 0].cpu().numpy()

        T_pred = unscale_temperature(y_pred_norm)
        mse = float(np.mean((T_pred - T_gt) ** 2))
        max_err = float(np.max(np.abs(T_pred - T_gt)))
        peak_gt = float(T_gt.max())
        peak_pred = float(T_pred.max())

        metrics.append({
            "sample": sample_idx + 1,
            "mse": mse,
            "max_error": max_err,
            "peak_gt": peak_gt,
            "peak_pred": peak_pred,
            "peak_diff": abs(peak_pred - peak_gt),
            "material": material,
            "chiplets": len(chiplets),
        })

        plot_comparison(sample_idx, X_raw, T_gt, T_pred)

    print("\nSummary:")
    print("sample | material | chips | mse   | max_err | peak_gt | peak_pred | peak_diff")
    for m in metrics:
        print(
            f"{m['sample']:>6d} | {m['material']:^8s} | {m['chiplets']:>5d} | "
            f"{m['mse']:.4f} | {m['max_error']:.3f} | "
            f"{m['peak_gt']:.2f} | {m['peak_pred']:.2f} | {m['peak_diff']:.2f}"
        )


if __name__ == "__main__":
    args = parse_args()
    evaluate_samples(args)
