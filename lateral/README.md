# Lateral Platoon Merging Simulation

Simulates a human-driven vehicle performing a **lane change to merge into a vehicle platoon** using Nash equilibrium-based shared control between an autonomous system and a human driver.

## What It Does

A platoon of vehicles travels at 72 km/h (20 m/s) in the right lane. A human-driven vehicle in the left lane decides to merge. The system and human driver share control of the steering wheel:
- **System (Player 1):** Slow, smooth, safe lane change (5th-order polynomial trajectory)
- **Human (Player 2):** Faster, more direct maneuver (3rd-order polynomial trajectory)

Dynamic authority allocation (λ) shifts control toward the system when risk increases. The MOBIL model first confirms the merge is safe and beneficial before Nash control activates.

## Quick Start

```bash
pip install numpy scipy matplotlib cvxpy
cd lateral
python main_with_menu.py
```

Choose a scenario and driver type from the interactive menu:

| Scenario | Human x-position | Merge trigger |
|----------|-----------------|---------------|
| Join before | +100 m | t = 3 s |
| Join middle | +35 m | t = 3 s |
| Join after | −20 m | t = 5 s |

Driver types: **cautious** (smooth), **normal** (baseline), **aggressive** (responsive).

Results saved to `convoy_simulation_results/` as PNG plots and GIF animations.

## Architecture

```
Nash Control Step (every 0.05 s):
  Safety Field → Authority Allocation (λ) → Reference Generation (R1_y/ψ, R2_y/ψ)
  → GNE Solve (CVXPY/OSQP) → δ_shared = δ1 + δ2
  → vehicle.update_dynamics(δ_shared)
```

Vehicle dynamics run at 100 Hz. Nash control runs at 20 Hz.

### Modules

| Module | Purpose |
|--------|---------|
| `main_with_menu.py` | Entry point, scenario/driver selection, output |
| `config.py` | All parameters (rates, weights, lane geometry, driver profiles) |
| `vehicle/` | 2-DOF bicycle model with dual body-frame / error-model propagation |
| `control/` | Stanley path-tracking controller, MOBIL lane-change decision, platoon manager |
| `nash_solver/` | GNE solver (CVXPY), safety field, authority allocator, reference generators |
| `simulation/` | Main simulation loop (`LateralSimulation`) and data logging |
| `visualization/` | 9-panel static plots and 4-panel animated GIF |

## Key Concepts

### Generalized Nash Equilibrium (Pustilnik & Borrelli 2025)

Unlike the longitudinal module, the lateral solver uses **Non-Normalized GNE** solved as a convex QP via CVXPY. Control outputs are **added** (not blended):

```
δ_shared = δ1 + δ2
```

Authority allocation is embedded inside the QP costs via α = λ/(1+λ):
- J₁ = ‖z−r₁‖²_Q + R₁‖u₁‖² + S₁‖u₂‖²
- J₂ = α·‖z−r₂‖²_Q + R₂‖u₂‖² + α·S₂‖u₁‖²

### Vehicle Model: Dual Propagation 2-DOF Bicycle

State `[y, ẏ, ψ, ψ̇]` is propagated twice per step with the same δ:
- **Body-frame** — tracks real position for animation and logging
- **Error-model** `[e₁, ė₁, e₂, ė₂]` — fed directly to the Nash solver for planning

### Phase State Machine

```
CRUISE → GAP_SEARCH → LANE_CHANGE → LANE_KEEPING → FOLLOWING
```

Reference generators produce different trajectories per phase. LANE_KEEPING entry uses a 5-second stability requirement. Phase transitions trigger smooth settling trajectories (V4.0) to avoid reference discontinuities.

### Driver Personality

Three profiles (cautious / normal / aggressive) affect all subsystems:

| Component | Cautious | Normal | Aggressive |
|-----------|---------|--------|-----------|
| Stanley k_e | 0.003 | 0.005 | 0.008 |
| Stanley k_psi | 0.3 | 0.5 | 0.7 |
| System T_lc | 9.0 s | 6.75 s | 4.5 s |
| Human T_lc | 6.0 s | 4.5 s | 3.0 s |
| MOBIL politeness | high | medium | low |
| Nash R2 factor | 1.5× | 1.0× | 0.6× |

## Parameters

All tunable parameters are in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SIMULATION_DT` | 0.01 s | Vehicle dynamics rate |
| `NASH_CONTROL_DT` | 0.05 s | Nash control rate |
| `NASH_NP` | 20 steps | Prediction horizon (1 s) |
| `NASH_NU` | 10 steps | Control horizon (0.5 s) |
| `LANE_WIDTH` | 3.5 m | Lane width |
| `NOMINAL_VELOCITY` | 20.0 m/s | Platoon speed |
| `NASH_Q_Y` | 10.0 | Lateral position weight |
| `NASH_Q_PSI` | 10 000 | Heading weight (high for stability) |
| `NASH_R1 = NASH_R2` | 1 000 000 | Control effort cost |
| `NASH_S1 = NASH_S2` | 200 000 | Cooperation coupling |

## Output

Per-scenario visualization:
- **9-panel comprehensive plot**: lateral position, heading, y_dot, steering inputs (system/human/shared), authority ratio, safety field force, y_error, lateral acceleration, phase timeline
- **2D trajectory plot**: bird's-eye view with lane boundaries and phase color-coding
- **9-panel Nash analysis**: state evolution, control inputs, cost and cooperation metrics
- **4-panel animation**: bird's-eye view, lateral position, steering, status bar

## References

- Li et al. (2019): *Shared control with a novel dynamic authority allocation strategy based on game theory and driving safety field*
- Pustilnik & Borrelli (2025): Non-Normalized Generalized Nash Equilibrium for shared control
- Treiber et al. (2000): Intelligent Driver Model (IDM)
- Kesting et al. (2007): MOBIL lane-change decision model
- Rajamani (2012): *Vehicle Dynamics and Control*
