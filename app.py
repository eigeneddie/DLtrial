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
if "design_state" not in st.session_state:
    st.session_state.design_state = {}
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
    image = Image.new("RGB", (size_px, size_px), "#F8FAFC")
    draw = ImageDraw.Draw(image, "RGBA")

    # Cooling strips are quiet background context, not the primary interaction.
    for i in range(int(tsv_count)):
        if tsv_orientation == "Vertical":
            col = 12 + i * 16
            draw.rectangle(
                [col * cell, 0, (col + 3) * cell - 1, size_px],
                fill=(99, 102, 241, 45),
            )
        else:
            row = 12 + i * 16
            draw.rectangle(
                [0, row * cell, size_px, (row + 3) * cell - 1],
                fill=(99, 102, 241, 45),
            )

    for grid_idx in range(0, GRID_SIZE + 1, 4):
        pos = grid_idx * cell
        color = (203, 213, 225, 170) if grid_idx % 16 else (148, 163, 184, 220)
        draw.line([(pos, 0), (pos, size_px)], fill=color, width=1)
        draw.line([(0, pos), (size_px, pos)], fill=color, width=1)

    colors = {
        "Logic": (30, 136, 229, 215),
        "Memory": (76, 175, 80, 215),
        "Controller": (255, 183, 77, 215),
    }
    outlines = {
        "Logic": (30, 64, 175, 255),
        "Memory": (22, 101, 52, 255),
        "Controller": (180, 83, 9, 255),
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

    draw.rectangle([0, 0, size_px - 1, size_px - 1], outline=(148, 163, 184, 255), width=2)
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
    tsv_count,
    tsv_orientation,
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
            "tsv_strip_count": int(tsv_count),
            "tsv_orientation": str(tsv_orientation),
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
            "cooling_controls": ["TSV strip count", "TSV orientation", "local convection h"],
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

st.title("Chilli Chiplets")
st.markdown("Place chiplets, run the trained thermal surrogate, and validate against the physics solver when needed.")
st.divider()

col_canvas, col_config = st.columns([1.5, 1])

with col_config:
    st.subheader("Design Controls")

    place_type = st.segmented_control(
        "Place component",
        options=list(COMPONENT_PRESETS.keys()),
        default="Logic",
    )

    preset = COMPONENT_PRESETS[place_type]
    c_size, c_power = st.columns(2)
    c_size.metric("Footprint", f"{preset['Width']} x {preset['Height']}")
    c_power.metric("Power", f"{preset['Power_W']:.0f} W")

    if st.button("Clear layout"):
        st.session_state.components_df = st.session_state.components_df.iloc[0:0].copy()
        st.session_state.simulation_run = False
        st.rerun()

    st.markdown("#### Package")
    material_choice = st.selectbox("Substrate Material", list(MATERIALS.keys()), format_func=lambda x: x.upper())
    solver_mode = "AI Surrogate"

    c_tsv, c_ori = st.columns(2)
    with c_tsv:
        num_tsvs = st.number_input("TSV strips", min_value=0, max_value=5, value=1)
    with c_ori:
        tsv_orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"])

    run_sim = st.button("Run thermal analysis")
    run_validation = st.button("Validate AI vs FDM")

    with st.expander("Advanced", expanded=False):
        solver_mode = st.radio(
            "Thermal engine",
            ["AI Surrogate", "FDM Physics"],
            horizontal=True,
            help="AI Surrogate uses thermal_surrogate.pth. FDM Physics keeps the numerical solver.",
        )
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

    st.caption(f"Selected: {place_type}. Click the package grid to place it. Soft blue bands show TSV cooling regions.")

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
        effective_engine = "Validation" if run_validation else solver_mode
        design_state = build_structured_design_state(
            components_df=edited_df,
            q_grid=Q_grid,
            k_grid=k_grid,
            h_grid=h_grid,
            temp_map=T_map,
            stress_map=stress_map,
            flag_map=flag_map,
            material_choice=material_choice,
            tsv_count=num_tsvs,
            tsv_orientation=tsv_orientation,
            solver_mode=effective_engine,
            runtime_ms=elapsed_ms,
            validation_result=validation_result,
        )

        # Save state
        st.session_state.T_map = T_map
        st.session_state.flag_map = flag_map
        st.session_state.validation = validation_result
        st.session_state.design_state = design_state
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
            "engine": effective_engine,
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
        fig_res, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor="#FFFFFF")
        display_grids = st.session_state.sim_grids or {"Q": Q_grid, "k": k_grid, "h": h_grid}

        panels = [
            (axes[0], display_grids["Q"], "Blues",   "Power Q [W]",      "Power Layout"),
            (axes[1], display_grids["k"], "Greens",  "k [W/(m·K)]",      f"Conductivity ({material_choice.upper()})"),
            (axes[2], display_grids["h"], "Purples", "h [W/(m²·K)]",     "Cooling (TSVs)"),
        ]
        for ax, data, cmap, cbar_label, title in panels:
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

        # Temp map
        ax = axes[3]
        ax.set_facecolor("#FFFFFF")
        T_map = st.session_state.T_map
        flag_map = st.session_state.flag_map

        im = ax.imshow(T_map, cmap="inferno", origin="upper", interpolation="bilinear", vmin=T_AMBIENT)
        cb = fig_res.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Temperature [°C]", fontsize=9, color="#334155")
        cb.ax.yaxis.set_tick_params(color="#475569")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")

        levels = np.arange(np.ceil((T_AMBIENT + 5) / 10) * 10, T_map.max(), 10)
        if len(levels):
            cs = ax.contour(T_map, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
            ax.clabel(cs, fmt="%d°C", fontsize=7, inline=True)

        rows, cols = np.where(flag_map)
        if len(rows):
            ax.scatter(cols, rows, c="#00FFFF", s=2, alpha=0.7, label="CTE fail")
            ax.legend(fontsize=8, loc="upper right", facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#111827")

        ax.set_title("Temperature Output", fontsize=11, color="#111827")
        ax.tick_params(colors="#64748B")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CBD5E1")

        plt.tight_layout()

        # Base64 for Co-Pilot
        buf = io.BytesIO()
        fig_res.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FFFFFF")
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

            fig_plot, axes_plot = plt.subplots(1, 2, figsize=(14, 5), facecolor="#FFFFFF")
            ai_flat = validation["ai_map"].ravel()
            fdm_flat = validation["fdm_map"].ravel()
            sample_idx = np.linspace(0, len(ai_flat) - 1, min(700, len(ai_flat)), dtype=int)
            min_temp = min(float(ai_flat.min()), float(fdm_flat.min()))
            max_temp = max(float(ai_flat.max()), float(fdm_flat.max()))

            ax = axes_plot[0]
            ax.set_facecolor("#FFFFFF")
            ax.scatter(
                fdm_flat[sample_idx],
                ai_flat[sample_idx],
                s=8,
                alpha=0.35,
                color="#64B5F6",
                edgecolors="none",
            )
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
            ax.plot(x_axis, validation["ai_map"][center_row], color="#64B5F6", linewidth=2, label="AI")
            ax.fill_between(
                x_axis,
                validation["fdm_map"][center_row],
                validation["ai_map"][center_row],
                color="#90CAF9",
                alpha=0.18,
            )
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
# PART 3 — OPTIONAL CO-PILOT
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_copilot_answer" not in st.session_state:
    st.session_state.latest_copilot_answer = ""
if "latest_copilot_question" not in st.session_state:
    st.session_state.latest_copilot_question = ""

with st.expander("Thermal copilot", expanded=False):
    rag, rag_error = load_rag()

    if rag_error:
        st.warning(f"Co-pilot unavailable: {rag_error}")
    elif not rag.ready:
        st.warning("Database unavailable. Run `python rag/ingest_thermal_papers.py` to index literature.")
    else:
        st.caption(f"Knowledge base active: {rag.chunk_count()} vectors loaded.")

    show_model_settings = st.checkbox("Show model settings", value=False)
    if show_model_settings:
        copilot_model = st.text_input(
            "Local Ollama model",
            value="gemma4:e4b",
            help="Use any local Ollama chat model, for example gemma4:e4b, granite3.2-vision, qwen3.5:9b, or llama3.1.",
        )
    if "copilot_model" not in locals():
        copilot_model = "gemma4:e4b"

    quick_question = st.text_input(
        "Ask about this layout",
        value="What does the heat map indicate and what should I change first?",
    )
    if st.button("Ask copilot"):
        sim_state = st.session_state.design_state if st.session_state.simulation_run else {
            "status": "No simulation has been run yet.",
            "components": component_records(edited_df),
        }
        with st.spinner("Reading simulation state..."):
            try:
                if rag is None:
                    response = f"[ERROR] RAG execution failed: {rag_error}"
                else:
                    response = rag.ask(
                        query=quick_question,
                        sim_state=sim_state,
                        heatmap_b64=st.session_state.heatmap_b64,
                        history=[],
                        provider="ollama",
                        api_key=copilot_model,
                    )
            except Exception as e:
                response = f"[ERROR] RAG execution failed: {e}"

        st.session_state.latest_copilot_question = quick_question
        st.session_state.latest_copilot_answer = response
        st.session_state.messages.append({"role": "user", "content": quick_question})
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.latest_copilot_answer:
        st.caption(st.session_state.latest_copilot_question)
        st.markdown(
            f"<div class='chat-message'>{st.session_state.latest_copilot_answer}</div>",
            unsafe_allow_html=True,
        )

with st.expander("Online evidence copilot", expanded=False):
    st.caption("Perplexity gathers current evidence; Gemma turns it into a concise layout recommendation.")
    api_from_env = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_from_env:
        st.warning("Set `PERPLEXITY_API_KEY` in the environment, or paste a temporary key below.")

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
    )
    temp_key = st.text_input(
        "Temporary Perplexity API key",
        value="",
        type="password",
        help="Leave empty to use PERPLEXITY_API_KEY from the environment.",
    )
    evidence_question = st.text_input(
        "Question",
        value="What does outside evidence suggest I should test next for this thermal result?",
    )

    source_goal = "grounding copilot recommendations with variables, equations, and experimental data"
    source_mode = "academic"
    source_model = "sonar-pro"
    show_online_settings = st.checkbox("Show online search settings", value=False)
    if show_online_settings:
        source_goal = st.text_input(
            "Source goal",
            value=source_goal,
        )
        source_mode = st.radio("Search mode", ["academic", "web"], horizontal=True)
        source_model = st.selectbox("Perplexity model", ["sonar-pro", "sonar"], index=0)

    source_query = build_source_query(
        source_topic,
        source_goal,
        design_state=st.session_state.design_state if st.session_state.design_state else None,
    )
    show_source_prompt = st.checkbox("Show generated search prompt", value=False)
    if show_source_prompt:
        st.code(source_query, language="text")

    if st.button("Ask with online evidence"):
        key = temp_key or api_from_env
        try:
            with st.spinner("Searching evidence and asking Gemma..."):
                evidence_package = search_perplexity_structured_evidence(
                    source_query,
                    api_key=key,
                    search_mode=source_mode,
                    model=source_model,
                )
                gemma_answer = ask_gemma_with_evidence(
                    evidence_question,
                    st.session_state.design_state,
                    evidence_package,
                    copilot_model if "copilot_model" in locals() else "gemma4:e4b",
                )
                st.session_state.evidence_recommendation = {
                    "evidence_package": evidence_package,
                    "gemma_answer": gemma_answer,
                }
        except Exception as e:
            st.session_state.evidence_recommendation = {"error": str(e)}

    evidence_result = st.session_state.get("evidence_recommendation")
    if evidence_result:
        st.markdown("#### Online Evidence Recommendation")
        if evidence_result.get("error"):
            st.error(evidence_result["error"])
        else:
            st.markdown(evidence_result["gemma_answer"])
            package = evidence_result["evidence_package"]
            if st.checkbox("Show structured Sonar evidence", value=False):
                st.json(package.get("evidence", {}))
            citations = package.get("citations", [])
            if citations:
                if st.checkbox("Show evidence citation URLs", value=False):
                    for url in citations:
                        st.markdown(f"- [{url}]({url})")
