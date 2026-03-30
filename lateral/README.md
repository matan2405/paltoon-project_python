# Lateral Convoy Merging — Algorithm Documentation

## Overview

This module simulates **lateral (lane-change) merging** — a human-driven vehicle changing lanes to join an autonomous convoy traveling in an adjacent lane. The core innovation is the same **Nash equilibrium shared control** framework as the longitudinal module, now applied to the 2D lateral merging problem.

**Primary references**:
- Pustilnik & Borrelli (2025) — Non-normalized GNE for lateral merging
- Kesting, Treiber & Helbing (2007) — MOBIL lane change model
- Wang et al. (2015/2016) — 2D elliptic driving safety field
- Swain & Rath (2023) — Sigmoid authority allocation (Eq. 15)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      main_with_menu.py                        │
│  (scenario menu, timestep loop, Nash integration)             │
└─────┬────────────────────────────────────────────────────────┘
      │
      ├── simulation/simulator.py       ← PlatoonSimulation (orchestrator)
      │        │
      │        ├── control/platoon_control.py   ← PlatoonManager (Stanley)
      │        ├── control/human_driver.py      ← Human lateral behavior
      │        ├── control/mobil_lane_change.py ← MOBIL decision model
      │        ├── vehicle/vehicle.py           ← Vehicle (bicycle model)
      │        └── vehicle/components.py        ← VehicleParameters
      │
      ├── nash_solver/lateral_constrained_nash_solver.py  ← Nash MPC (OSQP)
      ├── nash_solver/lateral_safety_field.py             ← 2D elliptic field
      ├── nash_solver/lateral_authority_allocator.py      ← λ with lateral sigmoid
      ├── nash_solver/system_reference_generator.py       ← R1 (convoy target)
      ├── nash_solver/human_reference_generator.py        ← R2 (human natural path)
      │
      ├── visualization/plots.py     ← Lateral analysis plots
      ├── visualization/animation.py ← Bird's-eye GIF
      └── metrics/                   ← Performance evaluation
```

---

## Algorithm Pipeline (Per Timestep)

```
Every SIMULATION_DT = 0.01 s (100 Hz):

  1. FORMATION CONTROL
     PlatoonManager → Stanley controller for convoy vehicles in left lane (y=0)

  2. HUMAN DRIVING
     HumanDriver → lateral behavior in right lane (y=1.75m)

  3. MOBIL DECISION (every step)
     MOBILLaneChange → should lane change be triggered?
     - Safety criterion: new follower won't brake dangerously
     - Incentive criterion (or mandatory mode): lane change is beneficial

  Every NASH_CONTROL_DT = 0.05 s (20 Hz):

  4. REFERENCE GENERATION
     SystemReferenceGenerator → R1 (convoy target position, Np=20 steps × 4 states)
     HumanReferenceGenerator  → R2 (human natural path, Np=20 steps × 4 states)

  5. RISK ASSESSMENT (2D)
     EllipseLateralSafetyField → (F_x, F_y) forces on human vehicle

  6. AUTHORITY ALLOCATION
     LateralAuthorityAllocator → λ(t)
     - Swain & Rath sigmoid on |e_y| (lateral error)
     - Safety component from |F_y|
     - Emergency override check

  7. NASH EQUILIBRIUM
     ConstrainedLateralNashSolver → (u1*, u2*, u_shared)
     vehicle.nash_acceleration = u_shared  (lateral acceleration [m/s²])

  8. VEHICLE DYNAMICS
     Vehicle.update() → bicycle model integration

  9. DATA LOGGING
```

---

## Key Differences from Longitudinal Module

| Aspect | Longitudinal | Lateral |
|---|---|---|
| Vehicle state | [x, vx] (2D) | [x, vx, y, vy] (4D) |
| Dynamics model | Belousov EOM (RK4) | Linearized bicycle model |
| Nash control rate | 10 Hz (0.1 s) | 20 Hz (0.05 s) |
| Lane change decision | N/A | MOBIL model |
| Human reference gen | HumanDriver.get_sequence() | HumanReferenceGenerator (separate module) |
| Authority sigmoid | Standard sigmoid on force | Swain & Rath (2023) sigmoid on |e_y| |
| Emergency override | Only λ scaling | Hard lateral correction + λ scaling |
| Safety field | 1D (longitudinal gap) | 2D elliptic (x-y space) |
| Results directory | `platoon_sim_kinematic_results/` | `lateral_sim_results_v2/` |

---

## MOBIL Lane Change Model

The MOBIL model (Kesting et al. 2007) decides **when** the human should change lanes. The Nash solver decides **how** to execute it.

### Safety Criterion
```
ã_new_follower ≥ −b_safe
```
The vehicle that will be behind the lane-changer in the target lane must not be forced to brake harder than `b_safe`.

### Incentive Criterion
```
ã_ego − a_ego + p·(ã_new_follower − a_new_follower) > Δa_threshold
```
The lane change must improve the ego's acceleration by at least `Δa_threshold`, accounting for the impact on surrounding vehicles scaled by politeness `p`.

### Mandatory Mode
For platoon joining, `mandatory_mode = True` skips the incentive criterion — the human **must** join the convoy when it is safe to do so.

---

## Authority Allocation: Swain & Rath (2023)

The lateral authority allocation uses a specific sigmoid formulation based on lateral position error from the target (Eq. 15 of Swain & Rath 2023):

```
α_lat = 1 / (1 + exp(−M1 · (|e_y| / e_y_max − M2)))

M1 = 2.0  (steepness)
M2 = 0.5  (center shift)
e_y_max = LANE_WIDTH / 2 = 1.75 m
```

**Interpretation**: At half the lane width of lateral error (1.75 m), the system and human have equal authority (α ≈ 0.5). Larger errors give the system more control.

---

## Configuration

All parameters in `lateral/config.py` — this is **completely separate** from `Longitudinal/config.py`.

Key parameter groups:

| Group | Prefix | Description |
|---|---|---|
| Simulation | `SIMULATION_DT`, `NASH_CONTROL_DT` | Timing |
| Driver types | `DRIVER_PARAMS` | Per-type lateral behavior |
| MOBIL model | `MOBIL_*` | Lane change decision params |
| Nash solver | `NASH_*` | Cost weights, horizons |
| Safety field | Embedded in class defaults | 2D ellipse parameters |
| Authority | `AUTHORITY_SIGMOID_M1`, `M2` | Swain & Rath params |
| Lane geometry | `LANE_WIDTH`, `NOMINAL_VELOCITY` | Road geometry |

---

## How to Run

```bash
cd lateral
python main_with_menu.py
```

Choose from the interactive menu:
- Driver type (cautious / normal / aggressive)
- Nash enabled / disabled
- Scenario configuration

---

## Dependencies

```bash
pip install numpy scipy matplotlib osqp cvxpy
```

---

## Output

Results saved to `lateral_sim_results_v2/`:
- `lateral_analysis_*.png` — lateral position, velocity, forces, authority
- `animation_*.gif` — bird's-eye lane change animation
- `metrics_report_*.txt` — quantitative lane change quality metrics
