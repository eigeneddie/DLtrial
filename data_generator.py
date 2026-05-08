"""
data_generator.py
=================
AI Thermal & Mechanical Co-Design Surrogate — Data Factory

Generates synthetic 2.5D Heterogeneous Integration (CoWoS-style) layouts and
solves the steady-state heat equation (2D Poisson) via Finite Difference Method.

Physics Model:
  Conduction  (Fourier's Law) : heat spreads via cell-neighbour averaging
  Variable k                  : higher substrate conductivity → lower T
  Variable h                  : stronger local cooling → lower T
  Source term                 : Q / k  (heat generation scaled by conductivity)

3-Channel Input X  — shape (N, 3, 64, 64):
  Channel 0 — Power Q         [W]       : chiplet heat sources
  Channel 1 — Conductivity k  [W/(m·K)] : substrate material (Si vs AlN)
  Channel 2 — Cooling h       [W/(m²·K)]: ambient + embedded TSV cooling

Output Y  — shape (N, 64, 64):
  Steady-state temperature field [°C]

Thermo-mechanical overlay (not saved, used for flagging):
  Stress [ppm·°C] = |CTE_chip − CTE_substrate| × (T − T_ambient)
  Cells above CTE_STRESS_THRESHOLD are flagged as shear-stress failure risk.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================================================================
# PHYSICS PARAMETERS
# Mechanical Engineers: all tunable knobs live here.
# ==============================================================================

GRID_SIZE = 64
T_AMBIENT = 25.0     # [°C]  coolant / environment temperature

# Substrate material library — (k [W/(m·K)], CTE [ppm/°C])
MATERIALS = {
    "silicon": (149.0, 2.6),
    "aln":     (260.0, 4.5),
}

# Reference values for normalising the FDM source and cooling terms
K_REF = 149.0   # [W/(m·K)]   Silicon reference conductivity
H_REF = 50.0    # [W/(m²·K)]  baseline ambient convection coefficient

# Calibration: with POWER_SCALE == COOL_RATE, the steady-state identity is:
#   T_peak  ≈  T_AMBIENT + Q_cell  [°C]   (at base cooling on Si substrate)
# Increase COOL_RATE to lower peak temperatures; decrease to raise them.
COOL_RATE   = 0.05   # dimensionless convective cooling fraction
POWER_SCALE = 0.05   # [°C/W] power-to-temperature gain (keep equal to COOL_RATE)

# FDM solver settings
MAX_ITERATIONS  = 5_000
CONVERGENCE_TOL = 1e-3

# CTE of all chiplets (assumed Silicon dies)
CTE_CHIP_PPM = 2.6     # [ppm/°C]
# Flag a cell if its stress indicator exceeds this value
CTE_STRESS_THRESHOLD = 150.0   # [ppm·°C]  ≈ onset of microbump fatigue risk

# TSV cooling enhancement: how many times stronger than H_REF
TSV_H_FACTOR = 8.0

# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================

NUM_SAMPLES = 100   # fast test batch (increase to 5_000 for full training run)
OUTPUT_DIR  = "."

# Power ranges per die type [W]
LOGIC_POWER_RANGE  = (80.0,  150.0)
MEMORY_POWER_RANGE = (5.0,   30.0)
TILE_POWER_RANGE   = (10.0,  60.0)


# ==============================================================================
# HELPERS
# ==============================================================================

def _place_rect(grid, occupied, r0, c0, h, w, value):
    """
    Place a rectangle of 'value' on grid at (r0,c0) with size (h×w).
    Returns True if placed, False if out-of-bounds or overlapping.
    """
    r1, c1 = r0 + h, c0 + w
    if r0 < 1 or c0 < 1 or r1 > GRID_SIZE - 1 or c1 > GRID_SIZE - 1:
        return False
    if np.any(occupied[r0:r1, c0:c1]):
        return False
    grid[r0:r1, c0:c1]    = value
    occupied[r0:r1, c0:c1] = True
    return True


# ==============================================================================
# STEP 1A — HUB & SPOKE LAYOUT
# Mirrors TSMC CoWoS GPU+HBM configuration:
#   one large central Logic die, 2-4 smaller Memory dies at cardinal positions.
# ==============================================================================

def hub_and_spoke_layout(rng):
    """
    Returns a 64×64 power grid [W] with hub-and-spoke chiplet placement.
    """
    Q        = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # Central Logic die (the hub)
    hub_h = int(rng.integers(14, 22))
    hub_w = int(rng.integers(14, 22))
    hub_r = GRID_SIZE // 2 - hub_h // 2
    hub_c = GRID_SIZE // 2 - hub_w // 2
    _place_rect(Q, occupied, hub_r, hub_c, hub_h, hub_w,
                float(rng.uniform(*LOGIC_POWER_RANGE)))

    # Peripheral Memory dies (the spokes) — one at each cardinal direction
    mem_h = int(rng.integers(5, 10))
    mem_w = int(rng.integers(5, 10))
    gap   = 2   # empty cells between hub and memory die

    for r0, c0 in [
        (hub_r - mem_h - gap,          hub_c + hub_w // 2 - mem_w // 2),  # North
        (hub_r + hub_h + gap,          hub_c + hub_w // 2 - mem_w // 2),  # South
        (hub_r + hub_h // 2 - mem_h // 2, hub_c - mem_w - gap),           # West
        (hub_r + hub_h // 2 - mem_h // 2, hub_c + hub_w + gap),           # East
    ]:
        _place_rect(Q, occupied, r0, c0, mem_h, mem_w,
                    float(rng.uniform(*MEMORY_POWER_RANGE)))

    return Q


# ==============================================================================
# STEP 1B — DISAGGREGATED TILE LAYOUT
# Mirrors tile-based chiplet designs (Intel Meteor Lake, AMD MI300X):
#   4–8 compute tiles distributed across the interposer.
# ==============================================================================

def disaggregated_layout(rng):
    """
    Returns a 64×64 power grid [W] with randomly distributed compute tiles.
    """
    Q        = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    n_tiles = int(rng.integers(4, 9))
    for _ in range(n_tiles):
        h = int(rng.integers(5, 12))
        w = int(rng.integers(5, 12))
        for _ in range(60):   # up to 60 placement attempts per tile
            r = int(rng.integers(1, GRID_SIZE - h - 1))
            c = int(rng.integers(1, GRID_SIZE - w - 1))
            if _place_rect(Q, occupied, r, c, h, w,
                           float(rng.uniform(*TILE_POWER_RANGE))):
                break

    return Q


# ==============================================================================
# STEP 1C — MATERIAL GRID (Channel 1)
# ==============================================================================

def build_material_grid(rng):
    """
    Builds the 64×64 conductivity map k [W/(m·K)].
    Randomly selects one bulk substrate material (Silicon or AlN) per sample.
    Returns the grid and the material name string.
    """
    material      = rng.choice(list(MATERIALS.keys()))
    k_bulk, _     = MATERIALS[material]
    k_grid        = np.full((GRID_SIZE, GRID_SIZE), k_bulk, dtype=np.float32)
    return k_grid, material


# ==============================================================================
# STEP 1D — COOLING GRID (Channel 2)
# ==============================================================================

def build_cooling_grid(rng):
    """
    Builds the 64×64 convection map h [W/(m²·K)].
    Background = H_REF (ambient).
    1–3 randomly placed TSV strips (vertical or horizontal) have 8× higher h.
    TSVs represent embedded thermal vias that channel heat to the back-side cooler.
    """
    h_grid = np.full((GRID_SIZE, GRID_SIZE), H_REF, dtype=np.float32)

    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:   # vertical strip
            col   = int(rng.integers(5, GRID_SIZE - 5))
            width = int(rng.integers(1, 4))
            h_grid[:, col:col + width] = H_REF * TSV_H_FACTOR
        else:                     # horizontal strip
            row    = int(rng.integers(5, GRID_SIZE - 5))
            height = int(rng.integers(1, 4))
            h_grid[row:row + height, :] = H_REF * TSV_H_FACTOR

    return h_grid


# ==============================================================================
# STEP 2 — FDM STEADY-STATE SOLVER (2D Poisson)
# ==============================================================================

def fdm_steady_state(Q_grid, k_grid, h_grid):
    """
    Solves the 2D steady-state Poisson heat equation using Jacobi iteration.

    Governing discrete update for each interior cell:

        T_new[i,j] = ( T_above + T_below + T_left + T_right
                       + POWER_SCALE * Q[i,j] / k_norm[i,j]
                       + h_norm[i,j]  * T_AMBIENT            )
                     / ( 4 + h_norm[i,j] )

    Where:
        k_norm  = k / K_REF               — relative conductivity
                  (AlN = 1.74, Si = 1.0)  higher → lower temperature rise
        h_norm  = (h / H_REF) * COOL_RATE — relative cooling rate
                  higher h → stronger cooling → lower T

    Calibration identity (Si substrate, base cooling, isolated chiplet region):
        T_peak  ≈  T_AMBIENT + Q_cell  [°C]

    Boundary condition: all four edges pinned to T_AMBIENT (Dirichlet BC).

    Parameters
    ----------
    Q_grid : (64, 64) — power layout [W]
    k_grid : (64, 64) — conductivity  [W/(m·K)]
    h_grid : (64, 64) — convection    [W/(m²·K)]

    Returns
    -------
    T : (64, 64) float32 — steady-state temperature [°C]
    """
    k_norm = (k_grid / K_REF).astype(np.float64)
    h_norm = ((h_grid / H_REF) * COOL_RATE).astype(np.float64)

    # Pre-compute constant numerator source and denominator maps
    source_map = (POWER_SCALE * Q_grid.astype(np.float64) / k_norm
                  + h_norm * T_AMBIENT)
    denom = 4.0 + h_norm

    # Initialise to ambient; boundary cells never change
    T = np.full((GRID_SIZE, GRID_SIZE), T_AMBIENT, dtype=np.float64)

    for _ in range(MAX_ITERATIONS):
        T_old = T.copy()

        neighbour_sum = (T[:-2, 1:-1] + T[2:,  1:-1] +
                         T[1:-1, :-2] + T[1:-1, 2:  ])

        T[1:-1, 1:-1] = (neighbour_sum + source_map[1:-1, 1:-1]) / denom[1:-1, 1:-1]

        # Re-pin boundaries (corners can drift from vectorised update)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_AMBIENT

        if np.max(np.abs(T - T_old)) < CONVERGENCE_TOL:
            break

    return T.astype(np.float32)


# ==============================================================================
# STEP 3 — CTE STRESS OVERLAY
# ==============================================================================

def compute_cte_stress(temp_map, k_grid):
    """
    Computes the thermo-mechanical shear stress indicator at each cell.

    Formula (JEDEC/IPC practice):
        Stress [ppm·°C] = |CTE_chip − CTE_substrate| × (T_cell − T_ambient)

    Chiplet CTE  = 2.6 ppm/°C  (Silicon)
    Substrate CTE is inferred from k_grid:
        k ≥ 200 W/(m·K) → AlN  (4.5 ppm/°C) — MISMATCH with Si chiplet
        k <  200 W/(m·K) → Si   (2.6 ppm/°C) — matched, near-zero stress

    Returns
    -------
    stress_map : (64, 64) float32 — stress indicator per cell [ppm·°C]
    flag_map   : (64, 64) bool    — True where stress > CTE_STRESS_THRESHOLD
    """
    cte_substrate = np.where(k_grid >= 200.0,
                             MATERIALS["aln"][1],      # 4.5 ppm/°C
                             MATERIALS["silicon"][1])  # 2.6 ppm/°C

    stress_map = np.abs(CTE_CHIP_PPM - cte_substrate) * (temp_map - T_AMBIENT)
    flag_map   = stress_map > CTE_STRESS_THRESHOLD

    return stress_map.astype(np.float32), flag_map


# ==============================================================================
# STEP 4 — SANITY-CHECK PLOT
# ==============================================================================

def plot_sanity_check(X_sample, Y_sample, k_grid, h_grid, save_path=None):
    """
    1×4 subplot sanity check for one generated sample.

    Panel 1 — Ch 0  Power Q      : where the chiplets are and their wattage
    Panel 2 — Ch 1  Conductivity : substrate material (dark=Si, bright=AlN)
    Panel 3 — Ch 2  Cooling h    : base ambient + bright TSV strips
    Panel 4 — Label Temperature  : heat diffusion result + CTE failure overlay
    """
    Q_grid     = X_sample[0]
    _, flag_map = compute_cte_stress(Y_sample, k_grid)
    material   = "AlN" if k_grid.mean() >= 200 else "Silicon"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Sanity Check — 2.5D Co-Design Sample  |  Substrate: {material}",
        fontsize=13, fontweight="bold"
    )

    # --- Panel 1: Power layout ---
    ax = axes[0]
    im = ax.imshow(Q_grid, cmap="Blues", origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Power [W]", fontsize=8)
    ax.set_title("Ch 0 — Power Q", fontsize=10)
    active = Q_grid[Q_grid > 0]
    ax.set_xlabel(
        f"Chiplet cells: {len(active)}  |  "
        f"{active.min():.0f}–{active.max():.0f} W" if len(active) else "no chiplets",
        fontsize=8
    )

    # --- Panel 2: Conductivity map ---
    ax = axes[1]
    im = ax.imshow(k_grid, cmap="Greens", origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("k [W/(m·K)]", fontsize=8)
    ax.set_title(f"Ch 1 — Conductivity k\n({material} substrate)", fontsize=10)

    # --- Panel 3: Cooling map ---
    ax = axes[2]
    im = ax.imshow(h_grid, cmap="Purples", origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("h [W/(m²·K)]", fontsize=8)
    ax.set_title("Ch 2 — Cooling h\n(bright strips = TSVs)", fontsize=10)

    # --- Panel 4: Temperature + CTE flags ---
    ax = axes[3]
    im = ax.imshow(Y_sample, cmap="hot", origin="upper", interpolation="bilinear",
                   vmin=T_AMBIENT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature [°C]", fontsize=8)

    # Isotherm contours every 10 °C
    levels = np.arange(np.ceil((T_AMBIENT + 5) / 10) * 10, Y_sample.max(), 10)
    if len(levels):
        cs = ax.contour(Y_sample, levels=levels, colors="white", linewidths=0.5, alpha=0.6)
        ax.clabel(cs, fmt="%d°C", fontsize=6, inline=True)

    # CTE failure cells as cyan overlay
    rows, cols = np.where(flag_map)
    if len(rows):
        ax.scatter(cols, rows, c="cyan", s=1, alpha=0.5, label=f"CTE fail ({len(rows)} cells)")
        ax.legend(fontsize=7, loc="upper right", markerscale=5)

    ax.set_title(
        f"Label Y — Temperature\n"
        f"Peak: {Y_sample.max():.1f} °C  |  CTE flags: {flag_map.sum()} cells",
        fontsize=10
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# ==============================================================================
# STEP 5 — NORMALIZATION  (Min-Max scaling to [0, 1])
# ==============================================================================

def normalize_dataset(X_data, Y_data):
    """
    Applies Min-Max normalization to every channel of X and to Y.

    Each X channel is scaled independently so that its physical range maps to
    [0, 1].  Y (temperature) is scaled using the global min/max across the
    whole dataset.

    Returns
    -------
    X_norm  : same shape as X_data, values in [0, 1]
    Y_norm  : same shape as Y_data, values in [0, 1]
    scales  : dict of {label: (min, max)} for each channel — needed to
              un-scale predictions back to physical units.
    """
    X_norm = np.zeros_like(X_data)
    scales = {}

    ch_names = ["Q_power_W", "k_conductivity_W_per_mK", "h_cooling_W_per_m2K"]
    for ch, name in enumerate(ch_names):
        ch_min = float(X_data[:, ch].min())
        ch_max = float(X_data[:, ch].max())
        denom  = ch_max - ch_min if ch_max != ch_min else 1.0
        X_norm[:, ch] = (X_data[:, ch] - ch_min) / denom
        scales[name]  = (ch_min, ch_max)

    y_min = float(Y_data.min())
    y_max = float(Y_data.max())
    Y_norm = (Y_data - y_min) / (y_max - y_min)
    scales["temperature_C"] = (y_min, y_max)

    return X_norm.astype(np.float32), Y_norm.astype(np.float32), scales


# ==============================================================================
# STEP 6 — DATA DICTIONARY  (readme.txt)
# ==============================================================================

def save_readme(scales, X_shape, Y_shape, path):
    """
    Writes a plain-text readme.txt documenting the channel mappings and the
    exact Min/Max values used for normalization.

    ML engineers use this to convert model outputs back to real temperatures:
        T_actual = T_norm * (T_max - T_min) + T_min
    """
    q_min,  q_max  = scales["Q_power_W"]
    k_min,  k_max  = scales["k_conductivity_W_per_mK"]
    h_min,  h_max  = scales["h_cooling_W_per_m2K"]
    t_min,  t_max  = scales["temperature_C"]

    content = f"""AI Thermal & Mechanical Co-Design Surrogate — Dataset Dictionary
Generated by data_generator.py
================================================================

FILES
-----
  X_data.npy   shape {X_shape}   dtype float32   range [0, 1]
  Y_data.npy   shape {Y_shape}   dtype float32   range [0, 1]

All values are Min-Max normalised before saving.
Raw physical values are recovered via:
    X_physical = X_norm * (X_max - X_min) + X_min
    Y_physical = Y_norm * (Y_max - Y_min) + Y_min

CHANNEL MAPPINGS  (X_data axis=1)
----------------------------------
  Channel 0 — Power Q            [W]        Chiplet heat sources (0 = substrate)
  Channel 1 — Conductivity k     [W/(m·K)]  Substrate material (Si or AlN)
  Channel 2 — Cooling h          [W/(m²·K)] Ambient convection + TSV-enhanced regions

NORMALIZATION CONSTANTS
-----------------------
  Channel 0  Q      min = {q_min:>10.4f} W          max = {q_max:>10.4f} W
  Channel 1  k      min = {k_min:>10.4f} W/(m·K)   max = {k_max:>10.4f} W/(m·K)
  Channel 2  h      min = {h_min:>10.4f} W/(m²·K)  max = {h_max:>10.4f} W/(m²·K)
  Label      T      min = {t_min:>10.4f} °C         max = {t_max:>10.4f} °C

UN-SCALING PREDICTIONS (Python)
--------------------------------
  import numpy as np
  T_norm  = model(X_norm)           # CNN output, shape (N, 64, 64), range [0,1]
  T_deg_C = T_norm * ({t_max:.4f} - {t_min:.4f}) + {t_min:.4f}

CTE STRESS OVERLAY (post-processing, not saved in .npy)
---------------------------------------------------------
  Stress [ppm·°C] = |CTE_chip - CTE_substrate| * (T_actual - T_ambient)
  CTE_chip (Silicon) = 2.6 ppm/°C
  CTE_substrate      = 2.6 ppm/°C (Si) or 4.5 ppm/°C (AlN)
  Flag threshold     = {CTE_STRESS_THRESHOLD} ppm·°C
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ==============================================================================
# STEP 7 — MAIN LOOP
# ==============================================================================

def main():
    print("=" * 60)
    print("  AI Thermal & Mech Co-Design Surrogate — Data Factory")
    print("=" * 60)
    print(f"  Samples  : {NUM_SAMPLES}")
    print(f"  Grid     : {GRID_SIZE} × {GRID_SIZE}")
    print(f"  X shape  : (N, 3, {GRID_SIZE}, {GRID_SIZE})  [Q, k, h]")
    print(f"  Y shape  : (N, {GRID_SIZE}, {GRID_SIZE})     [T °C]")
    print(f"  T_amb    : {T_AMBIENT} °C")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    X_data = np.zeros((NUM_SAMPLES, 3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    Y_data = np.zeros((NUM_SAMPLES, GRID_SIZE, GRID_SIZE),    dtype=np.float32)
    last_k, last_h = None, None

    for i in range(NUM_SAMPLES):
        # Step 1: layout (50/50 split between architectures)
        Q_grid = hub_and_spoke_layout(rng) if rng.random() < 0.5 else disaggregated_layout(rng)

        # Step 2: material and cooling channels
        k_grid, material = build_material_grid(rng)
        h_grid           = build_cooling_grid(rng)

        # Step 3: solve heat equation
        T_map = fdm_steady_state(Q_grid, k_grid, h_grid)

        # Step 4: pack raw (un-normalised) values into arrays
        X_data[i, 0] = Q_grid
        X_data[i, 1] = k_grid
        X_data[i, 2] = h_grid
        Y_data[i]    = T_map

        last_k, last_h = k_grid, h_grid

        if (i + 1) % 20 == 0:
            _, flags = compute_cte_stress(T_map, k_grid)
            print(
                f"  [{i+1:3d}/{NUM_SAMPLES}]  substrate: {material:7s}  |  "
                f"peak T: {T_map.max():6.1f} °C  |  CTE flags: {flags.sum():4d} cells"
            )

    # Sanity-check plot uses raw physical values — do this before normalizing
    plot_path = os.path.join(OUTPUT_DIR, "sanity_check.png")
    plot_sanity_check(X_data[-1], Y_data[-1], last_k, last_h, save_path=plot_path)

    # Normalize to [0, 1] and record scaling constants
    print("Normalising to [0, 1] ...")
    X_norm, Y_norm, scales = normalize_dataset(X_data, Y_data)

    # Save normalised arrays
    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "Y_data.npy")
    r_path = os.path.join(OUTPUT_DIR, "readme.txt")
    np.save(x_path, X_norm)
    np.save(y_path, Y_norm)
    save_readme(scales, X_norm.shape, Y_norm.shape, r_path)

    print()
    print(f"Saved: {x_path}   {X_norm.shape}  ({os.path.getsize(x_path)/1e6:.2f} MB)")
    print(f"Saved: {y_path}   {Y_norm.shape}  ({os.path.getsize(y_path)/1e6:.2f} MB)")
    print(f"Saved: {r_path}   (data dictionary + un-scaling constants)")
    print()
    print(f"  T range in dataset: {scales['temperature_C'][0]:.1f} – {scales['temperature_C'][1]:.1f} °C")
    print()
    print("Next step: run model_trainer.py to train the CNN surrogate.")


if __name__ == "__main__":
    import sys
    if "--preview" in sys.argv:
        # Single-sample preview — no files written
        _rng     = np.random.default_rng(seed=0)
        _Q       = hub_and_spoke_layout(_rng)
        _k, _mat = build_material_grid(_rng)
        _h       = build_cooling_grid(_rng)
        _T       = fdm_steady_state(_Q, _k, _h)
        print(f"Preview  |  substrate: {_mat}  |  peak T: {_T.max():.1f} °C")
        _save = "sanity_check.png" if "--save" in sys.argv else None
        plot_sanity_check(np.stack([_Q, _k, _h]), _T, _k, _h, save_path=_save)
    else:
        main()
