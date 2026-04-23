# Longitudinal Platoon Merging Simulation

Simulates a human-driven vehicle merging into a moving vehicle platoon in the longitudinal direction using **Nash equilibrium-based shared control** between an autonomous system and a human driver.

## What It Does

A platoon of vehicles travels at 120 km/h. A human-driven vehicle (in an adjacent lane) decides to join the platoon. The system negotiates control authority between:
- **Player 1 (System):** Wants safe, efficient platoon integration
- **Player 2 (Human driver):** Has own speed/gap preferences

A dynamic authority ratio λ determines how much the system overrides the human — increasing as risk (safety field force) grows. The result is smooth, cooperative merging without explicit cooperation agreements.

## Quick Start

```bash
pip install numpy scipy matplotlib cvxpy
cd Longitudinal
python main_with_nash.py
```

Choose a scenario from the interactive menu:
1. Join before the platoon (initial x = +100 m)
2. Join middle of the platoon (initial x = −10 m)
3. Join after the platoon (initial x = −100 m)
4. Run all three scenarios
5. Compare Nash vs. non-Nash control

Results are saved to `platoon_sim_kinematic_results/` as PNG plots and GIF animations.

## Architecture

```
Nash Control Step (every 0.1 s):
  Safety Field → Authority Allocation (λ) → Reference Generation (R1, R2)
  → Nash Equilibrium Solve → Shared Control (u = α·u1 + (1-α)·u2)
  → vehicle.nash_acceleration → PlatoonManager applies it
```

Vehicle dynamics run at 100 Hz. Nash runs at 10 Hz (multi-rate via substeps).

### Modules

| Module | Purpose |
|--------|---------|
| `main_with_nash.py` | Entry point, scenario definitions, Nash pipeline |
| `config.py` | All parameters (dt, weights, vehicle specs, safety field) |
| `vehicle/` | Vehicle dynamics: kinematic, state-space, hierarchical, complex powertrain |
| `control/` | Rajamani platoon controller, IDM human driver, lower-level controller |
| `nash_solver/` | Nash QP solver, safety field, authority allocator, reference generators |
| `simulation/` | Base simulation loop and data logging |
| `visualization/` | 9-panel static plots and 4-panel GIF animation |

## Key Concepts

### Nash Game (Li et al. 2019)
Non-cooperative two-player game where each player minimizes their own quadratic cost:
- J₁ = ‖R₁ − Z‖²_Q + R₁‖u₁‖² + S₁‖u₂‖²  *(system)*
- J₂ = α·‖R₂ − Z‖²_Q + R₂‖u₂‖² + α·S₂‖u₁‖²  *(human)*

where α = λ/(1+λ) is the Pustilnik authority scaling. High λ → human's cost aligns with system's → cooperative behavior emerges without explicit agreement.

### Safety Field
Five-component bidirectional field (TTC, Headway, Gap Error, Relative Velocity, Velocity Error):
- Positive force → repulsive (too close)
- Negative force → attractive (too far)

Two operating phases detected dynamically:
- **MERGING:** Aggressive field for gap convergence
- **FOLLOWING:** Soft field for steady-state comfort

### Vehicle Model
Audi TT Coupé 2.0 TFSI parameters. Three selectable motion models:
- **State-space** (double integrator, ZOH) — used for Nash planning
- **Kinematic** — fast simulation without engine dynamics
- **Hierarchical** — Nash plans with double integrator; lower-level controller executes with full powertrain (engine torque curve, 6-speed S-tronic, aero drag)

### Driver Types
Three IDM-based human profiles in `config.py`: **cautious**, **normal**, **aggressive** — varying acceleration limits, headway preference, and gap spacing.

## Parameters

All tunable parameters are in `config.py`. Key entries:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SIMULATION_DT` | 0.01 s | Vehicle dynamics rate |
| `NASH_CONTROL_DT` | 0.1 s | Nash control rate |
| `NASH_NP` | 20 steps | Prediction horizon (2 s) |
| `NASH_NU` | 10 steps | Control horizon (1 s) |
| `h` | 1.5 s | Desired time headway |
| `Q_pos` | 2500 | Position tracking weight |
| `Q_vel` | 50 | Velocity tracking weight |
| `R1 = R2` | 800 | Control effort cost |

## Output

Post-simulation visualization includes:
- **9-panel static plot**: positions, velocities, gaps, accelerations, jerk, Nash authority, phases
- **4-panel animation**: road view, velocities, gaps, timeline
- **Text summary**: cooperation/opposition statistics, risk metrics, authority ratio history

## References

- Li et al. (2019): *Shared control with a novel dynamic authority allocation strategy based on game theory and driving safety field*
- Rajamani (2012): *Vehicle Dynamics and Control*, Chapter 6.7 Transitional Controller
- Pustilnik & Borrelli (2025): Non-Normalized Generalized Nash Equilibrium formulation
