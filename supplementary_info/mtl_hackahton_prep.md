# SYSTEM BLUEPRINT: AI Thermal & Mechanical Co-Design Surrogate for 2.5D-HI

## 1. Project Context & Business Problem
We are building an Electronic Design Automation (EDA) tool for 2.5D Heterogeneous Integration (e.g., TSMC CoWoS). 
* **The Problem:** Advanced packaging suffers from severe thermal crosstalk and Coefficient of Thermal Expansion (CTE) mismatch during the reflow process and operation. Traditional finite-element simulations take 24+ hours, delaying time-to-market.
* **The Solution:** An AI Surrogate Model that predicts steady-state thermal distribution in milliseconds, allowing engineers to instantly co-design layout, substrate materials, and cooling strategies while automatically flagging CTE shear stress failures.

## 2. System Architecture
1. **Data Factory (`data_generator.py`):** A NumPy-based Finite Difference Method (FDM) physics engine. Generates synthetic 2.5D layouts and solves the 2D Poisson equation for steady-state heat.
2. **ML Surrogate (`model_trainer.py`):** A PyTorch Convolutional Neural Network (CNN) trained on the synthetic data.
3. **UI Dashboard (`app.py`):** A Streamlit web app for real-time interactive design. I've generated a sanity_check.png that shows 1. chip layout, 2. substrate material, and 3. TSV (cooling channel) layout. The UI dashboard can do something like displaying these 3 panels. For Panel 1, a user can place the logic die or memory die where ever they want. Note that these two elements are the main heat generating components. Panel 2 is the easiest - just swap between two substrate base materials. Panel 3 is where the user can place where the cooling strip can be.

## 3. The Physics Engine Details (The Data Factory)
The simulator uses a 64x64 NumPy grid representing the 2.5D interposer.
* **Procedural Layout Generation:** The generator must place chiplets using common 2.5D architectures, primarily:
  * *Hub & Spoke:* A large central high-power Logic die surrounded by smaller Memory dies.
  * *Disaggregated:* 4 to 8 smaller compute tiles spaced out.
* **The 3-Channel Input (Co-Design Parameters):** To allow the AI to learn material and cooling impacts, the input `X` must have 3 channels:
  * Channel 0: Power Grid ($Q$, in Watts) - The chiplet locations.
  * Channel 1: Thermal Conductivity Grid ($k$, in W/mK) - The substrate material (e.g., Silicon vs. Aluminum Nitride).
  * Channel 2: Convection/Cooling Grid ($h$) - Ambient cooling or embedded thermal TSVs.
* **The FDM Loop:** Solve for steady-state temperature using the discrete 2D Poisson approximation: average the 4 neighboring cells and add the local heat generation factor `(Q / k)`.

## 4. Mechanical Constraints (CTE Mismatch)
We evaluate thermo-mechanical reliability using: `Stress = |CTE_top - CTE_bottom| * (T_final - T_ambient)`.
The AI will predict `T_final`, and we will overlay this CTE logic to flag areas where microbump shear stress exceeds safe thresholds.

## 5. Data Format
* **X_train:** Shape `(N, 3, 64, 64)` representing Power, Material $k$, and Cooling $h$.
* **Y_train:** Shape `(N, 64, 64)` representing the fully diffused thermal heatmap (°C).
* **Storage:** Save batches as `X_data.npy` and `Y_data.npy`.

## 6. Data generator pipeline for LLM to help write
Write the `data_generator.py` script based on this pipeline. 
Specific requirements:
1. Write the FDM `while` loop that generates the synthetic data for `N=100` samples (as a fast test batch).
2. **Sanity Check Plot:** At the very end of the script, include a Matplotlib visualization that plots one random sample from the generated batch. It should display a 1x4 subplot showing: Channel 0 (Power), Channel 1 (Conductivity $k$), Channel 2 (Cooling $h$), and the Label (Final Temperature Heatmap).
3. Ensure the code is heavily commented.
4. **Normalization:** Ensure the script normalizes all values in the `X_data` and `Y_data` arrays to a `[0, 1]` scale before saving them using Min-Max scaling. 
5. **Data Dictionary:** Have the Python script automatically generate and save a `readme.txt` file alongside the `.npy` files. The readme must document the channel mappings (0: Power, 1: Material k, 2: Cooling h) and record the exact Min and Max values used for the normalization so the ML engineer can un-scale the predictions later.

## 7. Hardware realism guideline from the hackathon
- Synthesizability: If your project involves AI-generated circuit cells or device models, it should be synthetically viable. A "cool" design that cannot be physically realized on a chip or FPGA is a failure of engineering.
- Resource Awareness: Bonus points are awarded for AI models that are "Hardware Aware"—designs that consider power consumption (µW/mW) and thermal constraints.
- The Demo: You must provide a "Proof of Concept." If you don't have physical devices (obviously!), you must show a high-fidelity simulation or clear datasheet of the system.

## 8. Originality
- The "Fresh Start" Rule: While you may use open-source libraries, frameworks, and pre-trained models, the core logic of your project must be developed during the hackathon.
- Attribution: Any pre-existing IP or third-party datasets used must be clearly disclosed in your final presentation.


## 9. Judging criteria 
The projects will be judged based on Progress During The Event, all the projects must be related to microelectronics.  

Each team should submit: project title and one description paragraph before Sunday noon. 
Based on that order, each team will give a 5-10 min presentation with Q&A. 

Projects in each track will be evaluated based on:
Impact potential: feasibility and scalability
Innovation and originality: highlight your original ideas and approaches
Validation: clean benchmark, testing and show comparative performance


## 10 Relevant SOTA papers regarding 2.5D chip layout optimization
Here are the top 5 papers that define the state-of-the-art for 2.5D chiplet floorplanning, balancing both thermal and latency constraints:

1. DiffChip: Thermally Aware Chip Placement with Automatic Differentiation (2025)
The Core Idea: This paper tackles the slow convergence of traditional guess-and-check methods (like Simulated Annealing) by proposing a placement algorithm based entirely on automatic differentiation (AD).

Hackathon Relevance: It relies on a differentiable thermal solver that computes the exact sensitivity of the temperature map with respect to the X and Y coordinates of the chiplets. Since your team is already using a PyTorch U-Net, this paper provides the blueprint for how you might use gradients to automatically "push" chiplets away from thermal hotspots while mathematically minimizing total wirelength.

2. ChipletPart: Cost-Aware Partitioning for 2.5D Systems (2025/2026)
The Core Idea: This paper introduces a partitioner that specifically addresses the harsh physical constraints of 2.5D systems, most notably the limited physical "reach" of inter-chiplet I/O transceivers.

Hackathon Relevance: It combines genetic algorithms with a Simulated Annealing-based floorplanner. It proves that accounting for the physical limits of 2.5D interconnects is mandatory, showing that state-of-the-art min-cut partitioners often yield layouts that are physically impossible to route. It also provides open-source tools and cost models you could potentially adapt.

3. TAP-2.5D: A Thermally-Aware Chiplet Placement Methodology for 2.5D Systems (2021)
The Core Idea: While slightly older, this is a foundational paper for your exact problem. It explicitly argues against tightly packing chiplets (which is standard in 2D design), proving that strategically inserting physical spacing jointly minimizes temperature and total wirelength.

Hackathon Relevance: It introduces a highly practical data structure called the "Occupation Chiplet Matrix" (OCM) to represent unrestricted placements. In one case study, their methodology lowered an infeasible temperature by 20°C and improved the thermal design power envelope by 150W. The OCM concept is a very clean way to represent your layout grid in your Python backend.

4. Effects of Poor Workload Partitioning on System Performance for Chiplet-Based Systems (2026)
The Core Idea: A deep dive strictly into the latency and communication side of the equation. It demonstrates through experimental analysis that naive, suboptimal placement can increase inter-chiplet communication latency by up to 10x and cause severe network congestion.

Hackathon Relevance: It validates exactly why your team must include a Latency Score. It outlines how proper partitioning algorithms can achieve an 87.4% reduction in inter-chiplet traffic and improve system throughput by nearly 9x, giving you hard numbers to quote in your final presentation.

5. The Survey of 2.5D Integrated Architecture: An EDA Perspective (2025)
The Core Idea: A comprehensive, up-to-date overview summarizing current methodologies and future directions for chiplet architectures strictly from an Electronic Design Automation (EDA) standpoint.

Hackathon Relevance: This is the perfect backgrounder for the entire team to skim. It connects the front-end architectural choices to back-end physical design and packaging, providing the overarching context of why "Shift-Left" thermal and latency surrogates like yours are currently in such high demand by major foundries.


## 11. Post-mortem — Act 2: Inverse-Design Optimizer (Personal Learning Notes)

This section captures the engineering attempt to extend the diagnostic surrogate
(Act 1) into a generative layout co-pilot (Act 2). It did **not** make it into
the hackathon submission. Lives on the `act2-optimizer` git branch.

### 11.1 Goal
Take the user's starting layout and use gradient descent through the *frozen*
trained CNN to nudge chiplet (x, y) positions toward a layout with lower peak
temperature and shorter inter-die wirelength. Mental model: heat is a repulsive
spring (push hot dies apart), wires are an attractive spring (pull connected
dies together), and α is the priority knob between them.

### 11.2 What was built (`inverse_design.py`, `test_case_study_1.py`)
- **Soft-box chiplet rendering** — erf-blurred rectangles so chiplet (x, y)
  becomes a differentiable function of position (DiffChip §III-B).
- **LSE-HPWL** — Half-Perimeter Wire Length smoothed with Log-Sum-Exp so the
  max/min in the bounding-box formula has gradients flowing to all endpoints.
- **p-norm peak temperature** — `T_max ≈ (Σ T_i^p)^(1/p)` so gradient flows
  through more than just the single hottest cell.
- **Combined cost** — TAP-2.5D Eq. 12: `α·T_norm + (1−α)·W_norm`, both terms
  min-max normalized to [0, 1] so α is a meaningful blend, not unit-mismatched.
- **Auto-α policy** — TAP-2.5D Eq. 13: above 85 °C, α ramps with temperature;
  below, pure wirelength minimization.
- **Non-overlap penalty** — integral of excess occupancy above tolerance,
  prevents chiplets from piling onto each other.
- **Adam optimizer loop** — 50–100 steps, ~5 sec total per layout.
- **TAP-2.5D Case 1 validation script** — runs the published Multi-GPU System
  through our pipeline to compare against the paper's published numbers.

### 11.3 Why it didn't quite work — four root causes

**(a) Power semantics mismatch in the data factory.**
Original `data_generator.py` stamped *total chiplet power* into every cell of
the chiplet (e.g., a 26×26 GPU at 295 W stored "295" in each of its 676 cells).
This produced wildly unphysical Q grids (~200 kW total per layout) and led to
peak temperatures of 158 °C on Case 1 vs. the paper's 95 °C. The fix is per-cell
*power density* (W per cell = total / area), matching how HotSpot works.

**(b) Calibration was tuned for the wrong semantics.**
Once semantics was corrected, the original `POWER_SCALE = 0.05` produced
temperatures of 25 °C (basically ambient — too cold). A retune to
`POWER_SCALE = 16` brought Case 1 down to 88.8 °C — within 7 °C of the paper.
**This part actually worked.** The calibration is now physically meaningful.

**(c) CNN gradient sensitivity is weak.**
After regenerating data with the corrected calibration and retraining, the CNN
learned the right *global* physics but was only weakly sensitive to *small*
chiplet displacements. The optimizer's `∂T/∂coords` was near-zero, so even
with α=0.9 the thermal gradient barely contributed to the cost. The optimizer
ended up dominated by wirelength.

**(d) CNN under-predicts T relative to FDM ground truth.**
On the Case 1 layout, FDM gives 88.8 °C but the CNN predicts ~57 °C. The
auto-α policy reads the CNN value, sees `57 < 85` (the threshold), and sets
α = 0 → "thermally safe → pure wirelength minimization." The optimizer then
*compresses* chiplets, which is the wrong direction.

### 11.4 What we observed vs. the paper

| Metric           | TAP-2.5D paper | Our optimizer       |
|------------------|---------------:|--------------------:|
| Initial peak T   | 95.31 °C       | 88.76 °C (FDM)      |
| Final peak T     | 91.25 °C       | ~89 °C (~flat)      |
| ΔT               | −4.06 °C       | ~0 °C               |
| Initial HPWL     | 88,059 mm      | 244 cells           |
| ΔHPWL            | +10%           | +2% (with α=0.9)    |

The **direction is correct** with α=0.9 (HPWL goes up, matching the paper's
"sacrifice wirelength to gain thermal" trade-off). The **magnitude is small**
because the CNN gradient is too weak to drive aggressive chiplet movement.

### 11.5 What would actually fix it (in order of effort vs payoff)

1. **Replace the CNN with a differentiable FDM in JAX or PyTorch** (DiffChip's
   actual approach). Exact gradients, no surrogate sensitivity gap. Slower
   per step (~0.5 sec instead of 0.01 sec), but still ~25 sec per layout vs
   TAP-2.5D's 25 hours of Simulated Annealing. **Highest payoff.**

2. **Train a more sensitive surrogate.** Wider input distribution, denser
   sampling at boundary regimes, different loss (e.g., L1 with edge-aware
   weighting), or output the full T field as auxiliary loss to force spatial
   sensitivity. Medium effort, medium payoff.

3. **Use bandwidth-weighted HPWL** like TAP-2.5D's MILP routing — multiply
   each net's bounding-box length by its bit width (128 or 1024). Penalizes
   long high-bandwidth nets more aggressively. Easy fix.

4. **Multi-layer 3D FDM** with proper heat spreader, TIM, microbump layers —
   would close the remaining 7 °C gap to HotSpot. Big effort.

5. **Pin clumps for HPWL** (TAP-2.5D §III-A) — measure wirelength edge-to-edge
   instead of center-to-center. Marginal but more physically accurate.

### 11.6 Lessons for the next attempt

- **Surrogate-assisted optimization inherits the surrogate's weaknesses.**
  A CNN that looks fine in isolation (low MAE on test set) can still be useless
  for inverse design if its spatial gradients are too smooth.

- **Calibration and sensitivity are different problems.** Fixing one doesn't
  fix the other. A calibrated CNN can still have weak gradients.

- **Power semantics matter enormously.** Per-chiplet vs per-cell density is
  the difference between unphysical and physical. Always cross-check against
  a known reference case (TAP-2.5D Case 1 was a great anchor).

- **The right direction can be visible even when magnitude is small.** Don't
  conflate "didn't reproduce the paper's numbers" with "didn't validate the
  approach." +2% HPWL in the same direction as the paper's +10% is a real
  signal, just attenuated.

- **Hackathon scope discipline pays off.** Act 1 alone (diagnostic surrogate
  + CTE flagging + LLM copilot + Streamlit UI) was a defensible submission.
  Trying to bolt on Act 2 in parallel would have risked tanking Act 1's
  polish. The branching strategy (`act2-optimizer` lives separately) preserved
  optionality without adding deadline risk.

- **DiffChip's framing > our framing.** They differentiate through the
  physics solver itself (JAX autodiff over an FDM linear solve). We tried to
  borrow their math while keeping a learned CNN as the physics oracle —
  conceptually elegant but practically limited by the surrogate.

---

## 12. Surrogate Model — State as of 2026-05-10 (pre-presentation checkpoint)

### 12.1 What was retrained today

The model in `thermal_surrogate.pth` is a fresh train with these changes vs. the original:

| Change | Before | After |
|--------|--------|-------|
| Q range in training data | 80–150 W/cell | 80–300 W/cell |
| Layout types | hub-and-spoke, disaggregated | + packed GPU (two adjacent 200–300 W dies) |
| N samples | 1000 | 2000 |
| Output activation | `nn.Sigmoid()` | removed → `.clamp(0, 1)` |
| Loss function | `nn.MSELoss()` | `nn.L1Loss()` |
| Final val MAE | — | ~0.0057 normalized ≈ **1.4 °C** |

Motivation: original model was severely under-predicting GPU layouts (295 W normalized to ~1.97 with old max of 149.97 W → OOD, Sigmoid saturated near 0.5).

### 12.2 Known remaining issue — power semantics

The model (and FDM) uses **per-cell power** convention: `Q_grid[r0:r1, c0:c1] = total_chip_power_W`.
This means a 23×23 GPU at 295 W stamps 295 W into each of 529 cells → ~156 kW of simulated total heat. Result: peak temperatures of 250–275 °C for Si/no-TSV/center placement — physically unrealistic (real GPU junction limit ~125 °C).

The **right fix** (documented in §11.3a and §11.3b):
```python
# Per-cell power density instead of total power
Q_grid[r0:r1, c0:c1] = total_W / (height * width)
```
And recalibrate `POWER_SCALE` from `0.05` → `~16` to recover physically meaningful temperatures (~88 °C for TAP-2.5D Case 1, matching the paper).

**Why not done today:** requires full data regen + retrain. The relative comparisons (AlN vs Si, TSV vs no-TSV, corner vs center placement) are still directionally correct and sufficient for the demo.

### 12.3 If retraining for fun after the presentation

1. In `data_generator.py`: change `_place_rect` calls in layout functions to use `power / (h * w)` instead of `power`
2. Change `POWER_SCALE = 0.05` → `POWER_SCALE = 16` (from Act 2 calibration work)
3. In `app.py`: same fix when building `Q_grid` from `components_df`
4. Rerun `data_generator.py` (N=2000), then `model_trainer.py`
5. Cross-check: 2× CPU on Si, no TSV, corner placement → expect peak ~60–80 °C
