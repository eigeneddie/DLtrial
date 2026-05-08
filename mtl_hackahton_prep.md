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