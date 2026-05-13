"""
Surrogate justification plot — two panels:
  1. Per-evaluation speedup (HotSpot 23 s vs CNN 10 ms)
  2. SA optimization total runtime vs iterations

Runtime reference: Ma et al., TAP-2.5D, DATE 2021
  "The runtime for each HotSpot simulation is 23 seconds on average."

Run:  python plot_roi.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
HOTSPOT_SEC   = 23.0     # seconds per HotSpot call (TAP-2.5D, DATE 2021)
CNN_SEC       = 0.010    # seconds per CNN inference (~10 ms)
SPEEDUP       = HOTSPOT_SEC / CNN_SEC

N_EVALUATIONS = np.arange(0, 12_001, 1)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Justification for AI Surrogate — Speedup over HotSpot Thermal Simulation\n"
    r"$\it{Reference: Ma\ et\ al.,\ TAP\text{-}2.5D,\ DATE\ 2021}$",
    fontsize=13, fontweight="bold"
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Per-evaluation speedup bar chart
# ══════════════════════════════════════════════════════════════════════════════
methods  = ["HotSpot\n(TAP-2.5D)", "CNN Surrogate\n(ours)"]
times_ms = [HOTSPOT_SEC * 1000, CNN_SEC * 1000]
colors   = ["#d93025", "#4c8bf5"]

bars = ax1.bar(methods, times_ms, color=colors, width=0.4, zorder=3)
ax1.set_yscale("log")
ax1.set_ylabel("Time per evaluation (ms, log scale)", fontsize=11)
ax1.set_title("Panel 1 — Single Evaluation Time", fontsize=11, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_ylim(1, 1e6)

# Value labels on bars
for bar, t, label in zip(bars, times_ms, ["23,000 ms\n(23 seconds)", "10 ms"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        t * 2.5,
        label,
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color=bar.get_facecolor()
    )

# Speedup annotation
ax1.annotate(
    f"{SPEEDUP:,.0f}× faster",
    xy=(1, CNN_SEC * 1000),
    xytext=(1.25, 500),
    fontsize=11, fontweight="bold", color="#34a853",
    arrowprops=dict(arrowstyle="->", color="#34a853", lw=1.2),
)

ax1.text(
    0.5, 0.02,
    "Source: Ma et al., TAP-2.5D, DATE 2021\n\"The runtime for each HotSpot simulation is 23 seconds on average.\"",
    transform=ax1.transAxes, ha="center", va="bottom",
    fontsize=7.5, style="italic", color="#555555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc")
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — SA optimization total runtime vs iterations
# ══════════════════════════════════════════════════════════════════════════════
hotspot_hours = (N_EVALUATIONS * HOTSPOT_SEC) / 3600
cnn_seconds   = N_EVALUATIONS * CNN_SEC

ax2_left = ax2
ax2_right = ax2.twinx()

l1, = ax2_left.plot(N_EVALUATIONS, hotspot_hours, color="#d93025", linewidth=2.5,
                    label="HotSpot (hours)")
l2, = ax2_right.plot(N_EVALUATIONS, cnn_seconds, color="#4c8bf5", linewidth=2.5,
                     linestyle="--", label="CNN Surrogate (seconds)")

# Annotate key milestones
for n_iter, label_offset in [(2_000, 0.3), (5_000, 0.3), (10_000, 0.3)]:
    hs_h  = n_iter * HOTSPOT_SEC / 3600
    cnn_s = n_iter * CNN_SEC
    ax2_left.annotate(
        f"{hs_h:.0f} hrs",
        xy=(n_iter, hs_h),
        xytext=(n_iter - 800, hs_h + 4),
        fontsize=8, color="#d93025",
        arrowprops=dict(arrowstyle="->", color="#d93025", lw=0.8),
    )
    ax2_right.annotate(
        f"{cnn_s:.0f} sec",
        xy=(n_iter, cnn_s),
        xytext=(n_iter + 200, cnn_s + 5),
        fontsize=8, color="#4c8bf5",
        arrowprops=dict(arrowstyle="->", color="#4c8bf5", lw=0.8),
    )

ax2_left.set_xlabel("Layout evaluations", fontsize=11)
ax2_left.set_ylabel("HotSpot total time (hours)", fontsize=11, color="#d93025")
ax2_right.set_ylabel("CNN surrogate total time (seconds)", fontsize=11, color="#4c8bf5")
ax2_left.set_title("Panel 2 — Total Optimization Runtime", fontsize=11, fontweight="bold")
ax2_left.tick_params(axis="y", labelcolor="#d93025")
ax2_right.tick_params(axis="y", labelcolor="#4c8bf5")
ax2_left.set_xlim(0, N_EVALUATIONS[-1])
ax2_left.set_ylim(0)
ax2_right.set_ylim(0)
ax2_left.grid(axis="y", linestyle="--", alpha=0.3)

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, fontsize=9, loc="upper left")

# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
import os; os.makedirs("figures", exist_ok=True)
plt.savefig("figures/roi_surrogate_vs_simulation.png", dpi=150, bbox_inches="tight")
print("Saved: figures/roi_surrogate_vs_simulation.png")
plt.show()
