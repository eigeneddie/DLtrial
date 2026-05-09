"""
app.py — Chilli Chiplets: 2.5D Co-Design Dashboard
=======================================================
Professional EDA/CAD dashboard for thermal-mechanical co-design.

Features:
  - Interactive Component Placer via st.data_editor
  - Live Power Grid Blueprint
  - FDM Thermal Simulation
  - Engineering Analysis Assistant (RAG Co-Pilot)

Run:
    source /Users/drkinjangi/coding/data/.venv/bin/activate
    streamlit run app.py
"""

import base64
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Ensure local imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator import (
    GRID_SIZE, H_REF, MATERIALS, T_AMBIENT, TSV_H_FACTOR,
    _place_rect, compute_cte_stress, fdm_steady_state,
)

try:
    from rag.rag_engine import ChipletRAG
except Exception:
    ChipletRAG = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "thermal_surrogate.pth"

X_SCALES = {
    "Q": (0.0, 149.9663),
    "k": (149.0, 260.0),
    "h": (50.0, 400.0),
}
Y_SCALE = (25.0, 143.9612)


class ThermalSurrogate(nn.Module):
    """CNN architecture matching model_trainer.py."""

    def __init__(self):
        super().__init__()
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@st.cache_resource(show_spinner="Loading trained thermal surrogate...")
def load_surrogate_model():
    if not MODEL_PATH.exists():
        return None, f"Model weights not found: {MODEL_PATH.name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalSurrogate().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return (model, device), None


def normalize_input(q_grid, k_grid, h_grid):
    x_raw = np.stack([q_grid, k_grid, h_grid], axis=0).astype(np.float32)
    x_norm = np.zeros_like(x_raw, dtype=np.float32)
    for idx, key in enumerate(("Q", "k", "h")):
        min_val, max_val = X_SCALES[key]
        x_norm[idx] = (x_raw[idx] - min_val) / (max_val - min_val)
    return np.clip(x_norm, 0.0, 1.0)


def unscale_temperature(t_norm):
    t_min, t_max = Y_SCALE
    return t_norm * (t_max - t_min) + t_min


def surrogate_steady_state(q_grid, k_grid, h_grid):
    loaded, error = load_surrogate_model()
    if error:
        raise RuntimeError(error)

    model, device = loaded
    x_norm = normalize_input(q_grid, k_grid, h_grid)
    with torch.no_grad():
        x_tensor = torch.tensor(x_norm[None], dtype=torch.float32, device=device)
        pred_norm = model(x_tensor)[0, 0].cpu().numpy()
    return unscale_temperature(pred_norm).astype(np.float32)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    layout      = "wide",
    page_title  = "Thermal Co-Design",
    page_icon   = "📐",
)

# ── Custom CSS for EDA Aesthetic ───────────────────────────────────────────
st.markdown("""
<style>
    /* Global CAD-like styling */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4CAF50; /* Tech green */
        font-family: 'Consolas', 'Courier New', monospace;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 2px;
        border: none;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        border-color: #1565C0;
        color: white;
    }

    /* Tables/Dataframe */
    .stDataFrame {
        font-family: 'Consolas', 'Courier New', monospace;
    }

    /* Divider */
    hr {
        border-color: #333333;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
        border-right: 1px solid #333;
    }

    /* Console / Chat */
    .chat-message {
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 14px;
        border-left: 3px solid #1E88E5;
        background-color: #1E1E1E;
    }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

if "components_df" not in st.session_state:
    st.session_state.components_df = pd.DataFrame([
        {"Type": "Logic", "Power_W": 120.0, "Width": 8, "Height": 8, "X_Col": 28, "Y_Row": 28},
        {"Type": "Memory", "Power_W": 20.0, "Width": 4, "Height": 4, "X_Col": 12, "Y_Row": 12},
        {"Type": "Memory", "Power_W": 20.0, "Width": 4, "Height": 4, "X_Col": 48, "Y_Row": 12},
        {"Type": "Memory", "Power_W": 20.0, "Width": 4, "Height": 4, "X_Col": 12, "Y_Row": 48},
        {"Type": "Memory", "Power_W": 20.0, "Width": 4, "Height": 4, "X_Col": 48, "Y_Row": 48},
    ])

if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "T_map" not in st.session_state:
    st.session_state.T_map = None
if "flag_map" not in st.session_state:
    st.session_state.flag_map = None
if "heatmap_b64" not in st.session_state:
    st.session_state.heatmap_b64 = None
if "sim_metrics" not in st.session_state:
    st.session_state.sim_metrics = {}
if "sim_grids" not in st.session_state:
    st.session_state.sim_grids = {}
if "validation" not in st.session_state:
    st.session_state.validation = None
if "last_blueprint_click" not in st.session_state:
    st.session_state.last_blueprint_click = None

COMPONENT_PRESETS = {
    "Logic": {"Power_W": 120.0, "Width": 8, "Height": 8},
    "Memory": {"Power_W": 20.0, "Width": 4, "Height": 4},
    "Controller": {"Power_W": 45.0, "Width": 5, "Height": 5},
}


def add_component_from_click(component_type, row, col):
    preset = COMPONENT_PRESETS[component_type]
    height = int(preset["Height"])
    width = int(preset["Width"])
    y_row = int(np.clip(row - height // 2, 1, GRID_SIZE - height - 1))
    x_col = int(np.clip(col - width // 2, 1, GRID_SIZE - width - 1))
    new_component = {
        "Type": component_type,
        "Power_W": float(preset["Power_W"]),
        "Width": width,
        "Height": height,
        "X_Col": x_col,
        "Y_Row": y_row,
    }
    st.session_state.components_df = pd.concat(
        [st.session_state.components_df, pd.DataFrame([new_component])],
        ignore_index=True,
    )
    st.session_state.simulation_run = False


def render_placement_blueprint(components_df, tsv_count, tsv_orientation, size_px=576):
    cell = size_px // GRID_SIZE
    image = Image.new("RGB", (size_px, size_px), "#111417")
    draw = ImageDraw.Draw(image, "RGBA")

    # Cooling strips are quiet background context, not the primary interaction.
    for i in range(int(tsv_count)):
        if tsv_orientation == "Vertical":
            col = 12 + i * 16
            draw.rectangle(
                [col * cell, 0, (col + 3) * cell - 1, size_px],
                fill=(126, 87, 194, 70),
            )
        else:
            row = 12 + i * 16
            draw.rectangle(
                [0, row * cell, size_px, (row + 3) * cell - 1],
                fill=(126, 87, 194, 70),
            )

    for grid_idx in range(0, GRID_SIZE + 1, 4):
        pos = grid_idx * cell
        color = (55, 64, 72, 150) if grid_idx % 16 else (96, 112, 124, 185)
        draw.line([(pos, 0), (pos, size_px)], fill=color, width=1)
        draw.line([(0, pos), (size_px, pos)], fill=color, width=1)

    colors = {
        "Logic": (30, 136, 229, 215),
        "Memory": (76, 175, 80, 215),
        "Controller": (255, 183, 77, 215),
    }
    outlines = {
        "Logic": (144, 202, 249, 255),
        "Memory": (165, 214, 167, 255),
        "Controller": (255, 224, 178, 255),
    }
    for _, row in components_df.iterrows():
        try:
            comp_type = str(row["Type"])
            x0 = int(row["X_Col"]) * cell
            y0 = int(row["Y_Row"]) * cell
            x1 = min(size_px - 1, (int(row["X_Col"]) + int(row["Width"])) * cell - 1)
            y1 = min(size_px - 1, (int(row["Y_Row"]) + int(row["Height"])) * cell - 1)
            draw.rectangle([x0, y0, x1, y1], fill=colors.get(comp_type, (30, 136, 229, 215)))
            draw.rectangle([x0, y0, x1, y1], outline=outlines.get(comp_type, (220, 220, 220, 255)), width=2)
        except Exception:
            continue

    draw.rectangle([0, 0, size_px - 1, size_px - 1], outline=(130, 145, 158, 255), width=2)
    return image


def run_engine(engine_name, q_grid, k_grid, h_grid):
    start_time = time.perf_counter()
    if engine_name == "AI Surrogate":
        temp_map = surrogate_steady_state(q_grid, k_grid, h_grid)
    elif engine_name == "FDM Physics":
        temp_map = fdm_steady_state(q_grid, k_grid, h_grid)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return temp_map, elapsed_ms


def build_validation_result(ai_map, fdm_map, ai_ms, fdm_ms):
    err_map = np.abs(ai_map - fdm_map)
    peak_ai = float(ai_map.max())
    peak_fdm = float(fdm_map.max())
    mae = float(err_map.mean())
    rmse = float(np.sqrt(np.mean((ai_map - fdm_map) ** 2)))
    max_error = float(err_map.max())
    peak_error = abs(peak_ai - peak_fdm)
    speedup = float(fdm_ms / ai_ms) if ai_ms > 0 else float("inf")
    if peak_error <= 3.0 and mae <= 2.0:
        verdict = "Validated for early design exploration"
        verdict_detail = "The AI surrogate tracks the FDM reference closely enough for fast layout screening."
    elif peak_error <= 7.0 and mae <= 4.0:
        verdict = "Usable with engineering review"
        verdict_detail = "The surrogate is directionally useful, but this layout should be checked with FDM before decisions."
    else:
        verdict = "Needs FDM confirmation"
        verdict_detail = "The surrogate error is high for this layout; use the FDM map as the trusted reference."

    return {
        "ai_map": ai_map,
        "fdm_map": fdm_map,
        "err_map": err_map,
        "ai_ms": float(ai_ms),
        "fdm_ms": float(fdm_ms),
        "speedup": speedup,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "peak_ai": peak_ai,
        "peak_fdm": peak_fdm,
        "peak_error": peak_error,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — DESIGN ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

st.title("Thermal-Mechanical Co-Design Environment")
st.markdown("Configure package constraints, place active components, and execute steady-state thermal analysis.")
st.divider()

col_canvas, col_config = st.columns([1.5, 1])

with col_config:
    st.subheader("Component Placer")
    st.caption("Choose a component, then click directly on the blueprint.")

    place_type = st.segmented_control(
        "Component",
        options=list(COMPONENT_PRESETS.keys()),
        default="Logic",
    )

    preset = COMPONENT_PRESETS[place_type]
    c_size, c_power = st.columns(2)
    c_size.metric("Footprint", f"{preset['Width']} x {preset['Height']}")
    c_power.metric("Power", f"{preset['Power_W']:.0f} W")

    if st.button("CLEAR LAYOUT"):
        st.session_state.components_df = st.session_state.components_df.iloc[0:0].copy()
        st.session_state.simulation_run = False
        st.rerun()

    st.subheader("Substrate & Cooling")
    material_choice = st.selectbox("Substrate Material", list(MATERIALS.keys()), format_func=lambda x: x.upper())
    solver_mode = st.radio(
        "Thermal Engine",
        ["AI Surrogate", "FDM Physics"],
        horizontal=False,
        help="AI Surrogate uses thermal_surrogate.pth. FDM Physics keeps the original numerical solver.",
    )

    c_tsv, c_ori = st.columns(2)
    with c_tsv:
        num_tsvs = st.number_input("TSV Cooling Strips", min_value=0, max_value=5, value=1)
    with c_ori:
        tsv_orientation = st.selectbox("Strip Orientation", ["Vertical", "Horizontal"])

    run_sim = st.button("▶ RUN THERMAL ANALYSIS")
    run_validation = st.button("VALIDATE AI VS FDM")

    with st.expander("Advanced layout table", expanded=False):
        edited_df = st.data_editor(
            st.session_state.components_df,
            num_rows="dynamic",
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    "Type", help="Component Type", options=["Logic", "Memory", "Controller"], required=True
                ),
                "Power_W": st.column_config.NumberColumn("Power (W)", min_value=1.0, max_value=200.0, format="%.1f"),
                "Width": st.column_config.NumberColumn("Width", min_value=1, max_value=64),
                "Height": st.column_config.NumberColumn("Height", min_value=1, max_value=64),
                "X_Col": st.column_config.NumberColumn("X_Col", min_value=0, max_value=64),
                "Y_Row": st.column_config.NumberColumn("Y_Row", min_value=0, max_value=64),
            },
            width="stretch",
        )
        st.session_state.components_df = edited_df

edited_df = st.session_state.components_df

# ── Build LIVE blueprint grids ──────────────────────────────────────────────
Q_grid   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

# Place components from dataframe
for idx, row in edited_df.iterrows():
    try:
        _place_rect(
            Q_grid, occupied,
            int(row["Y_Row"]), int(row["X_Col"]),
            int(row["Height"]), int(row["Width"]),
            float(row["Power_W"])
        )
    except Exception:
        pass # Ignore out of bounds placing temporarily while editing

k_bulk, _ = MATERIALS[material_choice]
k_grid    = np.full((GRID_SIZE, GRID_SIZE), k_bulk, dtype=np.float32)

h_grid = np.full((GRID_SIZE, GRID_SIZE), H_REF, dtype=np.float32)
for i in range(int(num_tsvs)):
    if tsv_orientation == "Vertical":
        col = 12 + i * 16
        if col + 3 <= GRID_SIZE:
            h_grid[:, col:col + 3] = H_REF * TSV_H_FACTOR
    else:
        row = 12 + i * 16
        if row + 3 <= GRID_SIZE:
            h_grid[row:row + 3, :] = H_REF * TSV_H_FACTOR

with col_canvas:
    st.subheader("Package Blueprint")
    blueprint_image = render_placement_blueprint(
        edited_df,
        tsv_count=num_tsvs,
        tsv_orientation=tsv_orientation,
    )
    click = streamlit_image_coordinates(
        blueprint_image,
        width=576,
        key=f"blueprint_click_{len(edited_df)}_{place_type}_{num_tsvs}_{tsv_orientation}",
        cursor="crosshair",
    )
    if click:
        grid_col = int(np.clip(click["x"] / 576 * GRID_SIZE, 0, GRID_SIZE - 1))
        grid_row = int(np.clip(click["y"] / 576 * GRID_SIZE, 0, GRID_SIZE - 1))
        add_component_from_click(place_type, grid_row, grid_col)
        st.rerun()

    st.caption(f"Click the blueprint to place a {place_type}. Purple bands show TSV cooling regions.")

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

if run_sim or run_validation:
    engine_label = {
        "AI Surrogate": "trained AI surrogate",
        "FDM Physics": "finite difference solver",
    }[solver_mode] if run_sim else "AI surrogate and FDM validator"
    with st.spinner(f"Executing {engine_label}..."):
        validation_result = None
        try:
            if run_validation:
                ai_map, ai_ms = run_engine("AI Surrogate", Q_grid, k_grid, h_grid)
                fdm_map, fdm_ms = run_engine("FDM Physics", Q_grid, k_grid, h_grid)
                validation_result = build_validation_result(ai_map, fdm_map, ai_ms, fdm_ms)
                T_map = ai_map if solver_mode == "AI Surrogate" else fdm_map
                elapsed_ms = ai_ms + fdm_ms
            elif solver_mode == "AI Surrogate":
                T_map, elapsed_ms = run_engine("AI Surrogate", Q_grid, k_grid, h_grid)
            else:
                T_map, elapsed_ms = run_engine("FDM Physics", Q_grid, k_grid, h_grid)
        except Exception as e:
            st.error(f"AI surrogate unavailable: {e}")
            st.info("Falling back to FDM Physics for this run.")
            solver_mode = "FDM Physics"
            T_map, elapsed_ms = run_engine("FDM Physics", Q_grid, k_grid, h_grid)
        stress_map, flag_map = compute_cte_stress(T_map, k_grid)

        # Save state
        st.session_state.T_map = T_map
        st.session_state.flag_map = flag_map
        st.session_state.validation = validation_result
        st.session_state.sim_grids = {
            "Q": Q_grid.copy(),
            "k": k_grid.copy(),
            "h": h_grid.copy(),
        }
        st.session_state.simulation_run = True

        st.session_state.sim_metrics = {
            "peak_T": float(T_map.max()),
            "cte_fails": int(flag_map.sum()),
            "stress_max": float(stress_map.max()),
            "delta_T": float(T_map.max() - T_AMBIENT),
            "material": material_choice,
            "tsvs": num_tsvs,
            "engine": "Validation" if run_validation else solver_mode,
            "runtime_ms": float(elapsed_ms),
            "validation_verdict": validation_result["verdict"] if validation_result else "",
        }

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.simulation_run:
    st.divider()
    st.header("Analysis Results")

    m_data = st.session_state.sim_metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Peak Temperature", f"{m_data['peak_T']:.1f} °C", f"+{m_data['delta_T']:.1f} °C (ΔT)")
    m2.metric("CTE Failure Zones", f"{m_data['cte_fails']} cells", delta_color="inverse")
    m3.metric("Max Shear Stress", f"{m_data['stress_max']:.0f} ppm·°C")
    m4.metric("Substrate k", f"{k_bulk:.0f} W/m·K", material_choice.upper())
    m5.metric("Engine", m_data.get("engine", "FDM Physics"), f"{m_data.get('runtime_ms', 0):.1f} ms")

    map_tab, validation_tab = st.tabs(["Thermal Map", "Validation"])

    with map_tab:
        st.markdown("### Thermal Distribution Map")
        fig_res, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor="#121212")
        display_grids = st.session_state.sim_grids or {"Q": Q_grid, "k": k_grid, "h": h_grid}

        panels = [
            (axes[0], display_grids["Q"], "Blues",   "Power Q [W]",      "Power Layout"),
            (axes[1], display_grids["k"], "Greens",  "k [W/(m·K)]",      f"Conductivity ({material_choice.upper()})"),
            (axes[2], display_grids["h"], "Purples", "h [W/(m²·K)]",     "Cooling (TSVs)"),
        ]
        for ax, data, cmap, cbar_label, title in panels:
            ax.set_facecolor("#121212")
            im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest")
            cb = fig_res.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label, fontsize=9, color="#E0E0E0")
            cb.ax.yaxis.set_tick_params(color="#E0E0E0")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#E0E0E0")
            ax.set_title(title, fontsize=11, color="#E0E0E0")
            ax.tick_params(colors="#AAA")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        # Temp map
        ax = axes[3]
        ax.set_facecolor("#121212")
        T_map = st.session_state.T_map
        flag_map = st.session_state.flag_map

        im = ax.imshow(T_map, cmap="inferno", origin="upper", interpolation="bilinear", vmin=T_AMBIENT)
        cb = fig_res.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Temperature [°C]", fontsize=9, color="#E0E0E0")
        cb.ax.yaxis.set_tick_params(color="#E0E0E0")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#E0E0E0")

        levels = np.arange(np.ceil((T_AMBIENT + 5) / 10) * 10, T_map.max(), 10)
        if len(levels):
            cs = ax.contour(T_map, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
            ax.clabel(cs, fmt="%d°C", fontsize=7, inline=True)

        rows, cols = np.where(flag_map)
        if len(rows):
            ax.scatter(cols, rows, c="#00FFFF", s=2, alpha=0.7, label="CTE fail")
            ax.legend(fontsize=8, loc="upper right", facecolor="#1A1A1A", labelcolor="white")

        ax.set_title("Temperature Output", fontsize=11, color="#E0E0E0")
        ax.tick_params(colors="#AAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()

        # Base64 for Co-Pilot
        buf = io.BytesIO()
        fig_res.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#121212")
        buf.seek(0)
        heatmap_bytes  = buf.read()
        st.session_state.heatmap_b64 = base64.b64encode(heatmap_bytes).decode("utf-8")

        st.pyplot(fig_res)
        plt.close(fig_res)

    with validation_tab:
        validation = st.session_state.validation
        if not validation:
            st.info("Click `VALIDATE AI VS FDM` to collect validation data for the current layout.")
        else:
            st.subheader(validation["verdict"])
            st.caption(validation["verdict_detail"])

            fig_plot, axes_plot = plt.subplots(1, 2, figsize=(14, 5), facecolor="#121212")
            ai_flat = validation["ai_map"].ravel()
            fdm_flat = validation["fdm_map"].ravel()
            sample_idx = np.linspace(0, len(ai_flat) - 1, min(700, len(ai_flat)), dtype=int)
            min_temp = min(float(ai_flat.min()), float(fdm_flat.min()))
            max_temp = max(float(ai_flat.max()), float(fdm_flat.max()))

            ax = axes_plot[0]
            ax.set_facecolor("#121212")
            ax.scatter(
                fdm_flat[sample_idx],
                ai_flat[sample_idx],
                s=8,
                alpha=0.35,
                color="#64B5F6",
                edgecolors="none",
            )
            ax.plot([min_temp, max_temp], [min_temp, max_temp], color="#FFB74D", linewidth=1.5)
            ax.set_title("AI vs FDM Cell Temperatures", color="#E0E0E0", fontsize=11)
            ax.set_xlabel("FDM Reference [°C]", color="#E0E0E0")
            ax.set_ylabel("AI Surrogate [°C]", color="#E0E0E0")
            ax.tick_params(colors="#AAA")
            ax.grid(color="#333333", linewidth=0.5, alpha=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

            ax = axes_plot[1]
            ax.set_facecolor("#121212")
            center_row = GRID_SIZE // 2
            x_axis = np.arange(GRID_SIZE)
            ax.plot(x_axis, validation["fdm_map"][center_row], color="#FFB74D", linewidth=2, label="FDM")
            ax.plot(x_axis, validation["ai_map"][center_row], color="#64B5F6", linewidth=2, label="AI")
            ax.fill_between(
                x_axis,
                validation["fdm_map"][center_row],
                validation["ai_map"][center_row],
                color="#90CAF9",
                alpha=0.18,
            )
            ax.set_title("Centerline Temperature Profile", color="#E0E0E0", fontsize=11)
            ax.set_xlabel("Column Index", color="#E0E0E0")
            ax.set_ylabel("Temperature [°C]", color="#E0E0E0")
            ax.tick_params(colors="#AAA")
            ax.grid(color="#333333", linewidth=0.5, alpha=0.8)
            ax.legend(loc="upper right", facecolor="#1A1A1A", edgecolor="#333", labelcolor="#E0E0E0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

            fig_plot.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, wspace=0.28)
            st.pyplot(fig_plot)
            plt.close(fig_plot)

            st.markdown(
                f"**Values:** AI peak **{validation['peak_ai']:.1f} °C** | "
                f"FDM peak **{validation['peak_fdm']:.1f} °C** | "
                f"Peak error **{validation['peak_error']:.2f} °C** | "
                f"MAE **{validation['mae']:.2f} °C** | "
                f"RMSE **{validation['rmse']:.2f} °C** | "
                f"Max error **{validation['max_error']:.2f} °C** | "
                f"AI **{validation['ai_ms']:.1f} ms** | "
                f"FDM **{validation['fdm_ms']:.1f} ms** | "
                f"Runtime ratio **{validation['speedup']:.1f}x**"
            )

# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — ENGINEERING CONSOLE (CO-PILOT)
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Engineering Analysis Console")
st.markdown("LLM-powered assistant grounded in indexed thermal literature and simulation data.")

@st.cache_resource(show_spinner="Initializing RAG Engine...")
def load_rag():
    if ChipletRAG is None:
        return None, "RAG module is not installed in this checkout."
    try:
        return ChipletRAG(), None
    except Exception as e:
        return None, str(e)

rag, rag_error = load_rag()

if rag_error:
    st.warning(f"Co-pilot unavailable: {rag_error}")
elif not rag.ready:
    st.warning("Database unavailable. Run `python rag/ingest_thermal_papers.py` to index literature.")
else:
    st.caption(f"Knowledge Base Active: {rag.chunk_count()} vectors loaded.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Analysis console initialized. Awaiting queries."}]

for msg in st.session_state.messages:
    role_label = "ANALYSIS ASSISTANT" if msg["role"] == "assistant" else "ENGINEER"
    st.markdown(f"**{role_label}**<br/><div class='chat-message'>{msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Enter query (e.g. 'Analyze the CTE failure zones in this layout')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# If last message is user, generate response
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]

    sim_state = st.session_state.sim_metrics if st.session_state.simulation_run else {}
    # Inject current layout data into sim state
    sim_state["components"] = edited_df.to_dict("records")

    with st.spinner("Processing analysis query..."):
        try:
            if rag is None:
                response = f"[ERROR] RAG execution failed: {rag_error}"
            else:
                response = rag.ask(
                    query       = user_prompt,
                    sim_state   = sim_state,
                    heatmap_b64 = st.session_state.heatmap_b64,
                    history     = st.session_state.messages[:-1],
                    provider    = "gemini",
                )
        except Exception as e:
            response = f"[ERROR] RAG execution failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
