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
import hashlib
import io
import json
import os
import requests
import sys
import time
from datetime import datetime, timezone
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
    CTE_CHIP_PPM, CTE_STRESS_THRESHOLD, GRID_SIZE, H_REF, MATERIALS, T_AMBIENT, TSV_H_FACTOR,
    _place_rect, compute_cte_stress, fdm_steady_state,
)

try:
    from rag.rag_engine import ChipletRAG
except Exception:
    ChipletRAG = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "thermal_surrogate.pth"
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/v1/sonar"

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "query_topic": {"type": "string"},
        "evidence_items": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": ["paper", "datasheet", "standard", "vendor_note", "article", "unknown"],
                    },
                    "key_claim": {"type": "string"},
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "relevance_to_current_design": {"type": "string"},
                    "suitable_for_docling_ingestion": {"type": "boolean"},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "citation_index": {"type": "integer"},
                },
                "required": [
                    "title",
                    "source_type",
                    "key_claim",
                    "variables",
                    "relevance_to_current_design",
                    "suitable_for_docling_ingestion",
                    "confidence",
                    "citation_index",
                ],
                "additionalProperties": False,
            },
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommended_next_search": {"type": "string"},
    },
    "required": ["query_topic", "evidence_items", "caveats", "recommended_next_search"],
    "additionalProperties": False,
}

X_SCALES = {
    "Q": (0.0, 299.9977),
    "k": (149.0, 260.0),
    "h": (50.0, 400.0),
}
Y_SCALE = (25.0, 268.1102)


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
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).clamp(0, 1)


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

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #E5E7EB;
        color: #1F2937;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #111827;
        font-weight: 650;
        letter-spacing: 0;
    }

    [data-testid="stMetricValue"] {
        color: #2563EB;
        font-family: 'Consolas', 'Courier New', monospace;
    }

    [data-testid="stMetricDelta"] {
        color: #64748B;
    }

    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        letter-spacing: 0;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
        color: white;
    }

    .stDataFrame {
        font-family: 'Consolas', 'Courier New', monospace;
    }

    hr {
        border-color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #D1D5DB;
    }

    .chat-message {
        padding: 12px 14px;
        border-radius: 8px;
        margin: 8px 0 12px;
        font-size: 14px;
        border: 1px solid #DBEAFE;
        border-left: 3px solid #2563EB;
        background-color: #FFFFFF;
        color: #1F2937;
    }

    div[data-testid="stExpander"] {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }

    div[data-testid="stTabs"] button {
        color: #334155;
    }

    /* Segmented control — base label style */
    [data-testid="stSegmentedControl"] label {
        font-weight: 600 !important;
        transition: background 0.15s, color 0.15s;
    }
    [data-testid="stSegmentedControl"] label[data-chip="CPU"] { color: #1a4fba !important; }
    [data-testid="stSegmentedControl"] label[data-chip="CPU"][aria-checked="true"],
    [data-testid="stSegmentedControl"] label[data-chip="CPU"]:hover {
        background-color: #4285F4 !important; color: white !important;
    }
    [data-testid="stSegmentedControl"] label[data-chip="GPU"] { color: #1a6630 !important; }
    [data-testid="stSegmentedControl"] label[data-chip="GPU"][aria-checked="true"],
    [data-testid="stSegmentedControl"] label[data-chip="GPU"]:hover {
        background-color: #34A853 !important; color: white !important;
    }
    [data-testid="stSegmentedControl"] label[data-chip="HBM"] { color: #b45000 !important; }
    [data-testid="stSegmentedControl"] label[data-chip="HBM"][aria-checked="true"],
    [data-testid="stSegmentedControl"] label[data-chip="HBM"]:hover {
        background-color: #FB8C00 !important; color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

if "components_df" not in st.session_state:
    st.session_state.components_df = pd.DataFrame(
        columns=["Type", "Power_W", "Width", "Height", "X_Col", "Y_Row"]
    )

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
if "design_state" not in st.session_state:
    st.session_state.design_state = {}
if "last_blueprint_click" not in st.session_state:
    st.session_state.last_blueprint_click = None

# Board: 50mm × 50mm CoWoS-style interposer → 1 cell ≈ 0.78 mm (GRID_SIZE=64)
# Dimensions from TABLE II (Multi-GPU System, Chen et al.)
BOARD_MM = 50.0
CELL_MM = BOARD_MM / GRID_SIZE  # 0.78125 mm/cell

COMPONENT_PRESETS = {
    "CPU":  {"Power_W": 105.0, "Width": 15, "Height": 15},   # 12×12 mm
    "GPU":  {"Power_W": 295.0, "Width": 23, "Height": 23},   # 18.2×18.2 mm
    "HBM":  {"Power_W":  20.0, "Width": 10, "Height": 15},   # 7.75×11.87 mm
}

# h multipliers — None=ambient only, High=8× (TSV_H_FACTOR from data_generator)
# Real h values: None=50, Low=75, Medium=125, High=400 W/(m²·K)
TSV_DENSITY_LEVELS = {
    "None":   1.0,
    "Low":    1.5,
    "Medium": 2.5,
    "High":   TSV_H_FACTOR,
}
TSV_DENSITY_LABELS = {
    "None":   "None  — h = 50 W/(m²·K)  · ambient convection only",
    "Low":    "Low   — h = 75 W/(m²·K)  · sparse TSV array (~100 µm pitch)",
    "Medium": "Medium — h = 125 W/(m²·K) · standard TSV array (~50 µm pitch)",
    "High":   "High  — h = 400 W/(m²·K) · dense TSV array (~20 µm pitch)  ⚠ high cost",
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


def render_placement_blueprint(components_df, size_px=576):
    from PIL import ImageFont
    cell = size_px // GRID_SIZE
    image = Image.new("RGB", (size_px, size_px), "#F8FAFC")
    draw = ImageDraw.Draw(image, "RGBA")

    for grid_idx in range(0, GRID_SIZE + 1, 4):
        pos = grid_idx * cell
        color = (203, 213, 225, 170) if grid_idx % 16 else (148, 163, 184, 220)
        draw.line([(pos, 0), (pos, size_px)], fill=color, width=1)
        draw.line([(0, pos), (size_px, pos)], fill=color, width=1)

    colors = {
        "CPU": (66, 133, 244, 215),   # Google blue
        "GPU": (52, 168, 83, 215),    # Google green
        "HBM": (251, 140, 0, 215),    # amber orange
        "TSV": (99, 102, 241, 200),   # indigo
    }
    outlines = {
        "CPU": (25,  75, 180, 255),
        "GPU": (18,  90,  45, 255),
        "HBM": (180,  80,   0, 255),
        "TSV": (60,  60, 200, 255),
    }
    type_counters = {}
    for _, row in components_df.reset_index(drop=True).iterrows():
        try:
            comp_type = str(row["Type"])
            type_idx = type_counters.get(comp_type, 0)
            type_counters[comp_type] = type_idx + 1

            x0 = int(row["X_Col"]) * cell
            y0 = int(row["Y_Row"]) * cell
            x1 = min(size_px - 1, (int(row["X_Col"]) + int(row["Width"])) * cell - 1)
            y1 = min(size_px - 1, (int(row["Y_Row"]) + int(row["Height"])) * cell - 1)
            draw.rectangle([x0, y0, x1, y1], fill=colors.get(comp_type, (100, 100, 200, 215)))
            draw.rectangle([x0, y0, x1, y1], outline=outlines.get(comp_type, (60, 60, 160, 255)), width=2)

            label = f"{comp_type}{type_idx}"
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            font_size = max(12, min(cell * int(row["Width"]) // 2, 36))
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            lx = cx - tw // 2
            ly = cy - th // 2
            draw.text((lx + 2, ly + 2), label, fill=(0, 0, 0, 160), font=font)
            draw.text((lx, ly), label, fill=(255, 255, 255, 255), font=font)
        except Exception:
            continue

    draw.rectangle([0, 0, size_px - 1, size_px - 1], outline=(148, 163, 184, 255), width=2)

    # Scale label — bottom-left corner
    scale_label = f"{int(BOARD_MM)}×{int(BOARD_MM)} mm  |  1 cell = {CELL_MM:.2f} mm"
    try:
        scale_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        scale_font = ImageFont.load_default()
    sb = draw.textbbox((0, 0), scale_label, font=scale_font)
    sw, sh = sb[2] - sb[0], sb[3] - sb[1]
    px, py = 6, size_px - sh - 8
    draw.rectangle([px - 3, py - 3, px + sw + 3, py + sh + 3], fill=(255, 255, 255, 200))
    draw.text((px, py), scale_label, fill=(80, 80, 80, 255), font=scale_font)

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


def top_hotspots(temp_map, count=5, min_separation=4):
    candidates = np.dstack(np.unravel_index(np.argsort(temp_map.ravel())[::-1], temp_map.shape))[0]
    hotspots = []
    for row, col in candidates:
        temp = float(temp_map[row, col])
        if any(abs(int(row) - h["row"]) + abs(int(col) - h["col"]) < min_separation for h in hotspots):
            continue
        hotspots.append({"row": int(row), "col": int(col), "temperature_C": round(temp, 3)})
        if len(hotspots) >= count:
            break
    return hotspots


def component_records(components_df):
    records = []
    for idx, row in components_df.reset_index(drop=True).iterrows():
        try:
            width = int(row["Width"])
            height = int(row["Height"])
            x_col = int(row["X_Col"])
            y_row = int(row["Y_Row"])
            power = float(row["Power_W"])
            records.append({
                "id": int(idx + 1),
                "type": str(row["Type"]),
                "power_W": round(power, 3),
                "width_cells": width,
                "height_cells": height,
                "x_col": x_col,
                "y_row": y_row,
                "center_col": round(x_col + width / 2, 2),
                "center_row": round(y_row + height / 2, 2),
                "area_cells": int(width * height),
                "power_density_W_per_cell": round(power / max(width * height, 1), 4),
            })
        except Exception:
            continue
    return records


def nearest_component(row, col, components):
    if not components:
        return None
    distances = [
        (abs(row - comp["center_row"]) + abs(col - comp["center_col"]), comp)
        for comp in components
    ]
    _, comp = min(distances, key=lambda item: item[0])
    return {"id": comp["id"], "type": comp["type"]}


def build_structured_design_state(
    components_df,
    q_grid,
    k_grid,
    h_grid,
    temp_map,
    stress_map,
    flag_map,
    material_choice,
    solver_mode,
    runtime_ms,
    validation_result=None,
):
    active_cells = q_grid > 0
    hotspot_list = top_hotspots(temp_map)
    components = component_records(components_df)
    for hot in hotspot_list:
        hot["nearest_component"] = nearest_component(hot["row"], hot["col"], components)

    grad_row, grad_col = np.gradient(temp_map)
    grad_mag = np.sqrt(grad_row ** 2 + grad_col ** 2)
    power_by_type = {}
    for comp in components:
        power_by_type[comp["type"]] = round(power_by_type.get(comp["type"], 0.0) + comp["power_W"], 3)

    state = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_problem": "2.5D chiplet thermal-mechanical co-design",
        "grid": {
            "rows": GRID_SIZE,
            "cols": GRID_SIZE,
            "coordinate_system": "row/col indices, origin at top-left",
        },
        "engine": {
            "selected_mode": solver_mode,
            "runtime_ms": round(float(runtime_ms), 3),
            "ai_model": "thermal_surrogate.pth" if solver_mode in {"AI Surrogate", "Validation"} else None,
            "physics_reference": "2D finite-difference steady-state heat equation",
        },
        "materials": {
            "substrate": material_choice,
            "substrate_k_W_per_mK": float(MATERIALS[material_choice][0]),
            "substrate_cte_ppm_per_C": float(MATERIALS[material_choice][1]),
            "chiplet_cte_ppm_per_C": float(CTE_CHIP_PPM),
            "ambient_temperature_C": float(T_AMBIENT),
        },
        "cooling": {
            "tsv_density_level": tsv_density,
            "baseline_h_W_per_m2K": float(H_REF),
            "max_h_W_per_m2K": float(h_grid.max()),
            "tsv_h_enhancement_factor": float(TSV_H_FACTOR),
            "enhanced_cooling_cell_fraction": round(float(np.mean(h_grid > H_REF)), 5),
        },
        "components": components,
        "layout_summary": {
            "component_count": len(components),
            "component_counts_by_type": {str(k): int(v) for k, v in components_df["Type"].value_counts().to_dict().items()},
            "total_power_W": round(float(q_grid[active_cells].sum()), 3) if np.any(active_cells) else 0.0,
            "power_by_type_W": power_by_type,
            "active_area_cells": int(active_cells.sum()),
            "active_area_fraction": round(float(active_cells.mean()), 5),
        },
        "thermal_metrics": {
            "min_temperature_C": round(float(temp_map.min()), 3),
            "mean_temperature_C": round(float(temp_map.mean()), 3),
            "p95_temperature_C": round(float(np.percentile(temp_map, 95)), 3),
            "peak_temperature_C": round(float(temp_map.max()), 3),
            "delta_peak_above_ambient_C": round(float(temp_map.max() - T_AMBIENT), 3),
            "max_gradient_C_per_cell": round(float(grad_mag.max()), 3),
            "hotspots": hotspot_list,
        },
        "mechanical_metrics": {
            "stress_model": "|CTE_chip - CTE_substrate| * (T - T_ambient)",
            "cte_stress_threshold_ppm_C": float(CTE_STRESS_THRESHOLD),
            "max_stress_ppm_C": round(float(stress_map.max()), 3),
            "mean_stress_ppm_C": round(float(stress_map.mean()), 3),
            "failure_cell_count": int(flag_map.sum()),
            "failure_area_fraction": round(float(flag_map.mean()), 5),
        },
        "recommendation_variables": {
            "layout_controls": ["component type", "x_col", "y_row", "width", "height", "power_W"],
            "material_controls": ["substrate material", "thermal conductivity k", "CTE"],
            "cooling_controls": ["click-to-place TSV clusters (5×5 cells)", "local convection h"],
            "objectives": ["minimize peak temperature", "minimize CTE failure cells", "maximize AI/FDM agreement"],
        },
    }

    if validation_result:
        state["validation"] = {
            "reference": "FDM Physics",
            "surrogate": "AI Surrogate",
            "ai_peak_temperature_C": round(float(validation_result["peak_ai"]), 3),
            "fdm_peak_temperature_C": round(float(validation_result["peak_fdm"]), 3),
            "peak_error_C": round(float(validation_result["peak_error"]), 3),
            "mae_C": round(float(validation_result["mae"]), 3),
            "rmse_C": round(float(validation_result["rmse"]), 3),
            "max_error_C": round(float(validation_result["max_error"]), 3),
            "ai_runtime_ms": round(float(validation_result["ai_ms"]), 3),
            "fdm_runtime_ms": round(float(validation_result["fdm_ms"]), 3),
            "runtime_ratio_fdm_over_ai": round(float(validation_result["speedup"]), 3),
            "verdict": validation_result["verdict"],
        }

    return state


def build_source_query(topic, source_goal, design_state=None):
    context_bits = []
    if design_state:
        material = design_state.get("materials", {}).get("substrate")
        peak = design_state.get("thermal_metrics", {}).get("peak_temperature_C")
        failures = design_state.get("mechanical_metrics", {}).get("failure_cell_count")
        if material:
            context_bits.append(f"current substrate={material}")
        if peak is not None:
            context_bits.append(f"peak temperature={peak} C")
        if failures is not None:
            context_bits.append(f"CTE failure cells={failures}")

    context = "; ".join(context_bits) if context_bits else "no current simulation context"
    return (
        f"Find high-quality sources for a 2.5D chiplet packaging thermal design copilot.\n"
        f"Topic: {topic}\n"
        f"Goal: {source_goal}\n"
        f"Current design context: {context}\n\n"
        "Prioritize open-access papers, arXiv papers, vendor datasheets, standards, "
        "and experimental thermal packaging references. For each source, state the "
        "thermal/material variables it contains, why it matters for a 2.5D/3D chiplet "
        "design app, and whether it looks suitable for Docling ingestion."
    )


def search_perplexity_sources(query, api_key, search_mode="academic", model="sonar-pro"):
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is not configured.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a source discovery assistant for microelectronics packaging. "
                    "Return concise, source-grounded results. Do not invent citations."
                ),
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
        "max_tokens": 900,
        "search_mode": search_mode,
        "return_related_questions": True,
    }
    resp = requests.post(
        PERPLEXITY_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    if not resp.ok:
        raise RuntimeError(f"Perplexity API error: {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    return {
        "answer": content,
        "citations": data.get("citations") or [],
        "search_results": data.get("search_results") or [],
        "related_questions": data.get("related_questions") or [],
        "usage": data.get("usage") or {},
    }


def search_perplexity_structured_evidence(query, api_key, search_mode="academic", model="sonar-pro"):
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is not configured.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract structured evidence for microelectronics packaging. "
                    "Use citations/search results for URLs. Do not place URLs inside the JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    "Return JSON only. Each evidence item should connect one source claim "
                    "to variables useful for 2.5D chiplet thermal design."
                ),
            },
        ],
        "temperature": 0.05,
        "max_tokens": 1200,
        "search_mode": search_mode,
        "return_related_questions": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "thermal_packaging_evidence",
                "schema": EVIDENCE_SCHEMA,
            },
        },
    }
    resp = requests.post(
        PERPLEXITY_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if not resp.ok:
        raise RuntimeError(f"Perplexity API error: {resp.status_code} {resp.text[:300]}")

    data = resp.json()
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    try:
        evidence_json = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Perplexity returned invalid JSON: {e}. Raw: {content[:300]}")

    return {
        "evidence": evidence_json,
        "citations": data.get("citations") or [],
        "search_results": data.get("search_results") or [],
        "related_questions": data.get("related_questions") or [],
        "usage": data.get("usage") or {},
    }


def ask_gemma_with_evidence(question, design_state, evidence_package, model_name):
    prompt = {
        "current_design_state": design_state or {"status": "No simulation run yet."},
        "online_structured_evidence": evidence_package.get("evidence", {}),
        "citation_urls": evidence_package.get("citations", []),
        "engineer_question": question,
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise 2.5D packaging thermal copilot. Use the current design "
                    "state as the engineering truth and use online evidence only as support. "
                    "Do not invent values. Answer in three sections: Direct answer, Evidence, "
                    "Next experiment. Keep it under 180 words."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, indent=2)},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.05, "num_predict": 500},
    }
    resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Ollama API error: {resp.status_code} {resp.text[:300]}")
    return resp.json().get("message", {}).get("content", "").strip() or "No response from Gemma."

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — DESIGN ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

col_blueprint, col_results = st.columns([1, 1])

with st.sidebar:
    st.markdown("## 🌶 Chilli Chiplets")
    st.caption("2.5D thermal co-design surrogate")
    st.divider()

    place_type = st.segmented_control(
        "Place component",
        options=list(COMPONENT_PRESETS.keys()),
        default="CPU",
    )

    # Tag each segmented-control label by its text so CSS can color it per chip type
    st.markdown("""
    <script>
    (function tagChipLabels() {
        const colors = {CPU: "#4285F4", GPU: "#34A853", HBM: "#FB8C00"};
        const ctrl = document.querySelector('[data-testid="stSegmentedControl"]');
        if (!ctrl) { setTimeout(tagChipLabels, 150); return; }
        ctrl.querySelectorAll("label").forEach(lbl => {
            const txt = lbl.innerText.trim();
            if (colors[txt]) {
                lbl.setAttribute("data-chip", txt);
                const checked = lbl.previousElementSibling?.checked;
                lbl.style.color = checked ? "white" : colors[txt];
                lbl.style.fontWeight = "700";
                if (checked) lbl.style.backgroundColor = colors[txt];
            }
        });
    })();
    </script>
    """, unsafe_allow_html=True)

    if place_type is None:
        place_type = list(COMPONENT_PRESETS.keys())[0]
    preset = COMPONENT_PRESETS[place_type]
    c_size, c_power = st.columns(2)
    c_size.markdown(
        f"<div style='font-size:11px;color:#6B7280'><b>Footprint</b><br/>"
        f"{preset['Width']*CELL_MM:.1f} × {preset['Height']*CELL_MM:.1f} mm</div>",
        unsafe_allow_html=True,
    )
    c_power.markdown(
        f"<div style='font-size:11px;color:#6B7280'><b>Power</b><br/>"
        f"{'%g W' % preset['Power_W'] if preset['Power_W'] > 0 else 'Cooling'}</div>",
        unsafe_allow_html=True,
    )

    if st.button("Clear layout"):
        st.session_state.components_df = st.session_state.components_df.iloc[0:0].copy()
        st.session_state.simulation_run = False
        st.session_state.pop("_last_click", None)
        st.session_state["_editor_key"] = st.session_state.get("_editor_key", 0) + 1
        st.rerun()

    st.markdown("#### Package")
    material_choice = st.selectbox("Substrate Material", list(MATERIALS.keys()), format_func=lambda x: x.upper())
    tsv_density = st.select_slider(
        "Interposer TSV Density",
        options=list(TSV_DENSITY_LEVELS.keys()),
        value="Low",
    )
    st.caption(TSV_DENSITY_LABELS[tsv_density])
    st.divider()
    run_sim = st.button("Run Analysis", use_container_width=True)
    st.divider()
    use_physics = st.toggle("Switch to physics engine mode")
    solver_mode = "High-Fidelity Physics" if use_physics else "AI Surrogate"

edited_df = st.session_state.components_df

# ── Build LIVE blueprint grids ──────────────────────────────────────────────
Q_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

for idx, row in edited_df.iterrows():
    try:
        r0, c0 = int(row["Y_Row"]), int(row["X_Col"])
        r1 = min(r0 + int(row["Height"]), GRID_SIZE)
        c1 = min(c0 + int(row["Width"]),  GRID_SIZE)
        if r0 >= 0 and c0 >= 0 and r1 > r0 and c1 > c0:
            Q_grid[r0:r1, c0:c1] = float(row["Power_W"])
    except Exception:
        pass

k_bulk, _ = MATERIALS[material_choice]
k_grid    = np.full((GRID_SIZE, GRID_SIZE), k_bulk, dtype=np.float32)

h_scale = TSV_DENSITY_LEVELS[tsv_density]
h_grid  = np.full((GRID_SIZE, GRID_SIZE), H_REF * h_scale, dtype=np.float32)

_layout_hash = hashlib.md5(edited_df.to_json().encode()).hexdigest()
if st.session_state.get("_blueprint_hash") != _layout_hash:
    st.session_state["_blueprint_image"] = render_placement_blueprint(edited_df)
    st.session_state["_blueprint_hash"] = _layout_hash

if not st.session_state.simulation_run:
    with col_results:
        st.info("Place chiplets on the canvas, then click **Run Analysis**.")

with col_blueprint:
    blueprint_image = st.session_state["_blueprint_image"]
    click = streamlit_image_coordinates(
        blueprint_image,
        width=576,
        key="blueprint_click",
        cursor="crosshair",
    )
    _sidebar_sig = (place_type, material_choice, tsv_density, use_physics)
    _sidebar_changed = st.session_state.get("_prev_sidebar_sig") != _sidebar_sig
    st.session_state["_prev_sidebar_sig"] = _sidebar_sig
    if click and click != st.session_state.get("_last_click"):
        st.session_state["_last_click"] = click
        if not _sidebar_changed:
            grid_col = int(np.clip(click["x"] / 576 * GRID_SIZE, 0, GRID_SIZE - 1))
            grid_row = int(np.clip(click["y"] / 576 * GRID_SIZE, 0, GRID_SIZE - 1))
            add_component_from_click(place_type, grid_row, grid_col)
            st.rerun()

    st.caption(
        f"Selected: **{place_type}** ({COMPONENT_PRESETS[place_type]['Width']*CELL_MM:.1f}×"
        f"{COMPONENT_PRESETS[place_type]['Height']*CELL_MM:.1f} mm). "
        f"Canvas: **{BOARD_MM:.0f}×{BOARD_MM:.0f} mm** CoWoS interposer · "
        f"{GRID_SIZE}×{GRID_SIZE} grid · 1 cell = {CELL_MM:.2f} mm · "
        f"Scaled to match TAP-2.5D Case 1 (Ma et al., DATE 2021). "
        f"Click to place."
    )

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

if run_sim:
    engine_label = "trained AI surrogate" if solver_mode == "AI Surrogate" else "high-fidelity physics solver"
    engine_key   = "AI Surrogate" if solver_mode == "AI Surrogate" else "FDM Physics"
    with st.spinner(f"Executing {engine_label}..."):
        try:
            T_map, elapsed_ms = run_engine(engine_key, Q_grid, k_grid, h_grid)
        except Exception as e:
            st.error(f"AI surrogate unavailable: {e}")
            st.info("Falling back to high-fidelity physics solver.")
            engine_key = "FDM Physics"
            T_map, elapsed_ms = run_engine(engine_key, Q_grid, k_grid, h_grid)
        stress_map, flag_map = compute_cte_stress(T_map, k_grid)
        design_state = build_structured_design_state(
            components_df=edited_df,
            q_grid=Q_grid,
            k_grid=k_grid,
            h_grid=h_grid,
            temp_map=T_map,
            stress_map=stress_map,
            flag_map=flag_map,
            material_choice=material_choice,
            solver_mode=solver_mode,
            runtime_ms=elapsed_ms,
            validation_result=st.session_state.get("validation"),
        )
        st.session_state.T_map = T_map
        st.session_state.flag_map = flag_map
        st.session_state.design_state = design_state
        st.session_state.sim_grids = {"Q": Q_grid.copy(), "k": k_grid.copy(), "h": h_grid.copy()}
        st.session_state.simulation_run = True

        st.session_state.sim_metrics = {
            "peak_T": float(T_map.max()),
            "cte_fails": int(flag_map.sum()),
            "stress_max": float(stress_map.max()),
            "delta_T": float(T_map.max() - T_AMBIENT),
            "material": material_choice,
            "engine": solver_mode,  # display name (AI Surrogate / High-Fidelity Physics)
            "runtime_ms": float(elapsed_ms),
        }

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.simulation_run:
    m_data  = st.session_state.sim_metrics
    T_map   = st.session_state.T_map
    flag_map = st.session_state.flag_map

    # ── Results in right column, next to blueprint ─────────────────────────
    with col_results:
        ra, rb = st.columns(2)
        ra.metric("Peak T", f"{m_data['peak_T']:.1f} °C", f"+{m_data['delta_T']:.1f} °C")
        rb.metric("CTE Failures", f"{m_data['cte_fails']} cells")
        rc, rd = st.columns(2)
        rc.metric("Max Stress", f"{m_data['stress_max']:.0f} ppm·°C")
        rd.metric(m_data.get("engine", "FDM"), f"{m_data.get('runtime_ms', 0):.0f} ms")

        fig_inline, ax_inline = plt.subplots(figsize=(6, 5.5), facecolor="#FFFFFF")
        ax_inline.set_facecolor("#FFFFFF")
        im_inline = ax_inline.imshow(T_map, cmap="inferno", origin="upper",
                                     interpolation="bilinear", vmin=T_AMBIENT)
        cb_inline = fig_inline.colorbar(im_inline, ax=ax_inline, fraction=0.046, pad=0.04)
        cb_inline.set_label("°C", fontsize=8, color="#334155")
        cb_inline.ax.yaxis.set_tick_params(color="#475569", labelsize=7)

        levels_i = np.arange(np.ceil((T_AMBIENT + 5) / 20) * 20, T_map.max(), 20)
        if len(levels_i):
            cs_i = ax_inline.contour(T_map, levels=levels_i, colors="white",
                                     linewidths=0.4, alpha=0.5)
            ax_inline.clabel(cs_i, fmt="%d°C", fontsize=6, inline=True)

        rows_i, cols_i = np.where(flag_map)
        if len(rows_i):
            ax_inline.scatter(cols_i, rows_i, c="#00FFFF", s=1.5, alpha=0.7, label="CTE fail")
            ax_inline.legend(fontsize=7, loc="upper right",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#111827")

        ax_inline.set_title("Temperature Map", fontsize=9, color="#111827")
        ax_inline.tick_params(colors="#64748B", labelsize=7)
        for spine in ax_inline.spines.values():
            spine.set_edgecolor("#CBD5E1")
        fig_inline.tight_layout()

        # Encode for copilot
        buf_i = io.BytesIO()
        fig_inline.savefig(buf_i, format="png", dpi=110, bbox_inches="tight", facecolor="#FFFFFF")
        buf_i.seek(0)
        st.session_state.heatmap_b64 = base64.b64encode(buf_i.read()).decode("utf-8")

        st.pyplot(fig_inline)
        plt.close(fig_inline)

    # ── Full detail below (full-width) ────────────────────────────────────
    # Pre-render input grids to bytes so st.image (not st.pyplot) is used
    # inside the tab — avoids Streamlit rendering the figure outside the tab
    # container during reruns, which caused a duplicate heatmap flash.
    display_grids = st.session_state.sim_grids or {"Q": Q_grid, "k": k_grid, "h": h_grid}
    fig_res, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="#FFFFFF")
    for ax, (data, cmap, cbar_label, title) in zip(axes, [
        (display_grids["Q"], "Blues",   "Power Q [W]",  "Power Layout"),
        (display_grids["k"], "Greens",  "k [W/(m·K)]",  f"Conductivity ({material_choice.upper()})"),
        (display_grids["h"], "Purples", "h [W/(m²·K)]", "Cooling (TSVs)"),
    ]):
        ax.set_facecolor("#FFFFFF")
        im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest")
        cb = fig_res.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=9, color="#334155")
        cb.ax.yaxis.set_tick_params(color="#475569")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")
        ax.set_title(title, fontsize=11, color="#111827")
        ax.tick_params(colors="#64748B")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CBD5E1")
    fig_res.tight_layout()
    _buf_res = io.BytesIO()
    fig_res.savefig(_buf_res, format="png", dpi=100, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig_res)
    _buf_res.seek(0)
    _map_tab_img = _buf_res.read()

    map_tab, validation_tab = st.tabs(["Input Grids", "Validation"])

    with map_tab:
        st.image(_map_tab_img, use_container_width=True)

    with validation_tab:
        st.button(
            "Compare AI vs Physics",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"_validate_requested": True}),
        )
        if st.session_state.pop("_validate_requested", False):
            with st.spinner("Running AI + physics comparison..."):
                try:
                    ai_map, ai_ms   = run_engine("AI Surrogate", Q_grid, k_grid, h_grid)
                    fdm_map, fdm_ms = run_engine("FDM Physics",  Q_grid, k_grid, h_grid)
                    st.session_state.validation = build_validation_result(ai_map, fdm_map, ai_ms, fdm_ms)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
        validation = st.session_state.validation
        if not validation:
            st.caption("Run a comparison to see how closely the AI surrogate matches the physics solver for this layout.")
        else:
            st.subheader(validation["verdict"])
            st.caption(validation["verdict_detail"])

            fig_plot, axes_plot = plt.subplots(1, 2, figsize=(14, 5), facecolor="#FFFFFF")
            ai_flat  = validation["ai_map"].ravel()
            fdm_flat = validation["fdm_map"].ravel()
            sample_idx = np.linspace(0, len(ai_flat) - 1, min(700, len(ai_flat)), dtype=int)
            min_temp = min(float(ai_flat.min()), float(fdm_flat.min()))
            max_temp = max(float(ai_flat.max()), float(fdm_flat.max()))

            ax = axes_plot[0]
            ax.set_facecolor("#FFFFFF")
            ax.scatter(fdm_flat[sample_idx], ai_flat[sample_idx],
                       s=8, alpha=0.35, color="#64B5F6", edgecolors="none")
            ax.plot([min_temp, max_temp], [min_temp, max_temp], color="#FFB74D", linewidth=1.5)
            ax.set_title("AI vs FDM Cell Temperatures", color="#111827", fontsize=11)
            ax.set_xlabel("FDM Reference [°C]", color="#334155")
            ax.set_ylabel("AI Surrogate [°C]", color="#334155")
            ax.tick_params(colors="#64748B")
            ax.grid(color="#E2E8F0", linewidth=0.6, alpha=0.9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E1")

            ax = axes_plot[1]
            ax.set_facecolor("#FFFFFF")
            center_row = GRID_SIZE // 2
            x_axis = np.arange(GRID_SIZE)
            ax.plot(x_axis, validation["fdm_map"][center_row], color="#FFB74D", linewidth=2, label="FDM")
            ax.plot(x_axis, validation["ai_map"][center_row],  color="#64B5F6", linewidth=2, label="AI")
            ax.fill_between(x_axis, validation["fdm_map"][center_row], validation["ai_map"][center_row],
                            color="#90CAF9", alpha=0.18)
            ax.set_title("Centerline Temperature Profile", color="#111827", fontsize=11)
            ax.set_xlabel("Column Index", color="#334155")
            ax.set_ylabel("Temperature [°C]", color="#334155")
            ax.tick_params(colors="#64748B")
            ax.grid(color="#E2E8F0", linewidth=0.6, alpha=0.9)
            ax.legend(loc="upper right", facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#111827")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E1")

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
# PART 3 — DESIGN ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════

st.divider()

@st.cache_resource(show_spinner="Initializing RAG Engine...")
def load_rag():
    if ChipletRAG is None:
        return None, "RAG module is not installed in this checkout."
    try:
        return ChipletRAG(), None
    except Exception as e:
        return None, str(e)

for _key, _default in [
    ("messages", []),
    ("local_answer", ""),
    ("online_answer", ""),
    ("online_citations", []),
    ("copilot_model", "gemma4:e4b"),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

rag, rag_error = load_rag()

# ── Section header ──────────────────────────────────────────────────────────
st.subheader("Design Assistant")
rag_status = ""
if rag is None or rag_error:
    rag_status = f"Local RAG unavailable — {rag_error}"
elif not rag.ready:
    rag_status = "Local RAG not indexed. Run `python rag/ingest_thermal_papers.py` to activate."
else:
    rag_status = f"Local knowledge base active · {rag.chunk_count()} vectors"
st.caption(rag_status)

# ── Single question bar ─────────────────────────────────────────────────────
q_col, btn_local, btn_online = st.columns([5, 1, 1.4])
with q_col:
    assistant_question = st.text_input(
        "question",
        value="What does the heat map indicate and what should I change first?",
        label_visibility="collapsed",
        placeholder="Ask anything about this thermal layout...",
    )
with btn_local:
    ask_local = st.button("Ask Copilot", use_container_width=True)
with btn_online:
    ask_online = st.button("Ask + Evidence", use_container_width=True)

# ── Assistant settings (collapsed by default) ───────────────────────────────
with st.expander("Settings", expanded=False):
    set_c1, set_c2, set_c3 = st.columns(3)
    with set_c1:
        st.markdown("**Local model**")
        copilot_model = st.text_input(
            "Ollama model",
            value=st.session_state.copilot_model,
            help="Any local Ollama chat model, e.g. gemma4:e4b, llama3.1, qwen3.5:9b",
        )
        st.session_state.copilot_model = copilot_model
    with set_c2:
        st.markdown("**Online search**")
        source_topic = st.selectbox(
            "Evidence topic",
            [
                "2.5D chiplet thermal modeling",
                "TSV thermal cooling and heat extraction",
                "HBM and logic thermal crosstalk",
                "AlN silicon glass interposer material properties",
                "CTE mismatch microbump fatigue reliability",
                "TIM heat spreader package thermal resistance",
                "AI surrogate models for thermal simulation",
            ],
            label_visibility="collapsed",
        )
        source_mode = st.radio("Search mode", ["academic", "web"], horizontal=True)
    with set_c3:
        st.markdown("**Perplexity API**")
        api_from_env = os.getenv("PERPLEXITY_API_KEY", "")
        temp_key = st.text_input(
            "API key",
            value="",
            type="password",
            help="Perplexity API is paid — get a key at perplexity.ai/api. Leave blank to use PERPLEXITY_API_KEY env var.",
            label_visibility="collapsed",
        )
        source_model = st.selectbox("Model", ["sonar-pro", "sonar"], index=0)
        if not api_from_env and not temp_key:
            st.caption("No key set — Ask + Evidence will be disabled.")

# ── Local copilot execution ─────────────────────────────────────────────────
if ask_local:
    sim_state = st.session_state.design_state if st.session_state.simulation_run else {
        "status": "No simulation run yet.",
        "components": component_records(edited_df),
    }
    with st.spinner("Querying local knowledge base..."):
        try:
            if rag is None:
                local_resp = f"RAG unavailable: {rag_error}"
            elif not rag.ready:
                local_resp = "Knowledge base not indexed. Run `python rag/ingest_thermal_papers.py` first."
            else:
                local_resp = rag.ask(
                    query=assistant_question,
                    sim_state=sim_state,
                    heatmap_b64=st.session_state.heatmap_b64,
                    history=[],
                    provider="ollama",
                    api_key=st.session_state.copilot_model,
                )
        except Exception as e:
            local_resp = f"Error: {e}"
    st.session_state.local_answer = local_resp
    st.session_state.messages.append({"role": "user", "content": assistant_question})
    st.session_state.messages.append({"role": "assistant", "content": local_resp})

# ── Online evidence execution ───────────────────────────────────────────────
if ask_online:
    _key = (temp_key if "temp_key" in locals() and temp_key else None) or api_from_env
    _topic = source_topic if "source_topic" in locals() else "2.5D chiplet thermal modeling"
    _goal = "grounding copilot recommendations with variables, equations, and experimental data"
    _mode = source_mode if "source_mode" in locals() else "academic"
    _model = source_model if "source_model" in locals() else "sonar-pro"
    source_query = build_source_query(
        _topic, _goal,
        design_state=st.session_state.design_state if st.session_state.design_state else None,
    )
    if not _key:
        st.session_state.online_answer = (
            "No Perplexity API key found. Add `PERPLEXITY_API_KEY` to your environment "
            "or paste a key in Settings above. The Perplexity API is a paid service — "
            "sign up at perplexity.ai/api."
        )
        st.session_state.online_citations = []
    else:
        with st.spinner("Fetching evidence and querying Gemma..."):
            try:
                evidence_package = search_perplexity_structured_evidence(
                    source_query, api_key=_key, search_mode=_mode, model=_model,
                )
                gemma_answer = ask_gemma_with_evidence(
                    assistant_question,
                    st.session_state.design_state,
                    evidence_package,
                    st.session_state.copilot_model,
                )
                st.session_state.online_answer = gemma_answer
                st.session_state.online_citations = evidence_package.get("citations", [])
            except Exception as e:
                st.session_state.online_answer = f"Error: {e}"
                st.session_state.online_citations = []

# ── Side-by-side response area ──────────────────────────────────────────────
res_local, res_online = st.columns(2)

with res_local:
    st.markdown("##### Local Copilot")
    st.caption("RAG knowledge base + local Ollama model")
    if st.session_state.local_answer:
        st.markdown(
            f"<div class='chat-message'>{st.session_state.local_answer}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='chat-message' style='color:#94A3B8;'>Response appears here after you click **Ask Copilot**.</div>",
            unsafe_allow_html=True,
        )

with res_online:
    st.markdown("##### Online Evidence")
    st.caption("Perplexity academic search → Gemma synthesis")
    if st.session_state.online_answer:
        st.markdown(
            f"<div class='chat-message'>{st.session_state.online_answer}</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.online_citations:
            with st.expander(f"Citations ({len(st.session_state.online_citations)})", expanded=False):
                for _url in st.session_state.online_citations:
                    st.markdown(f"- [{_url}]({_url})")
    else:
        st.markdown(
            "<div class='chat-message' style='color:#94A3B8;'>Response appears here after you click **Ask + Evidence**.</div>",
            unsafe_allow_html=True,
        )
