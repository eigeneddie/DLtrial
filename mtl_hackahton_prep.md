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
