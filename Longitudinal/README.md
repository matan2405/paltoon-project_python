# Longitudinal Platoon Merging — Algorithm Documentation

## Overview

This module simulates **longitudinal platoon merging** — a scenario where a human-driven vehicle joins an existing autonomous platoon from behind, ahead, or from the middle. The core innovation is **Nash equilibrium shared control**: the autonomous system and the human driver are modeled as two players in a non-cooperative game, and their actions are blended dynamically based on the current risk level.

**Primary reference**: Li et al. (2019) "Shared control with a novel dynamic authority allocation strategy based on game theory and driving safety field."

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      main_with_nash.py                        │
│  (scenario setup, timestep loop, Nash integration)            │
└─────┬────────────────────────────────────────────────────────┘
      │
      ├── simulation/simulator.py  ←── PlatoonSimulation (orchestrator)
      │        │
      │        ├── control/platoon_control.py  ←── PlatoonManager (Rajamani)
      │        ├── control/human_driver.py     ←── HumanDriver (IDM)
      │        ├── vehicle/vehicle.py          ←── Vehicle (dynamics)
      │        └── vehicle/components.py       ←── VehicleParameters, Engine, Transmission
      │
      ├── nash_solver/longitudinal_constrained_nash_solver.py  ←── Nash MPC (OSQP)
      ├── nash_solver/longitudinal_safety_field.py             ←── Risk force (elliptic field)
      ├── nash_solver/longitudinal_authority_allocator.py      ←── λ computation
      ├── nash_solver/system_reference_generator.py            ←── R1 (system trajectory)
      │
      ├── visualization/plots.py     ←── Static analysis plots
      └── visualization/animation.py ←── Animated GIF
```

---

## Algorithm Pipeline (Per Timestep)

```
Every SIMULATION_DT = 0.01 s (100 Hz):

  1. FORMATION CONTROL
     PlatoonManager → Rajamani controller for each autonomous vehicle
     → desired acceleration a_des for each follower

  2. HUMAN DRIVING
     HumanDriver → IDM acceleration for human vehicle
     → a_human (before merge: independent; after merge: overridden by Nash)

  Every NASH_CONTROL_DT = 0.1 s (10 Hz), also:

  3. REFERENCE GENERATION
     SystemReferenceGenerator → R1 (Np=20 step system trajectory)
     HumanDriver.get_sequence() → R2 (Np=20 step human trajectory)

  4. RISK ASSESSMENT
     EllipseLongitudinalSafetyField → risk force F(t) [N]
     Phase: MERGING (full force) or FOLLOWING (soft force)

  5. AUTHORITY ALLOCATION
     LongitudinalAuthorityAllocator → λ(t)
     Safety: λ_s = f(sigmoid(F)) ∈ [0.1, 100]
     Performance: λ_p = f(|gap_error|) with hysteresis
     λ = max(λ_s, λ_p), smoothed with adaptive EMA

  6. NASH EQUILIBRIUM
     ConstrainedLongitudinalNashSolver → (u1*, u2*)
     Shared: u_shared = α·u1* + (1−α)·u2*   where α = λ/(1+λ)
     → vehicle.nash_acceleration = u_shared

  7. VEHICLE DYNAMICS
     Vehicle.update() → RK4 integration of Belousov Eq. 3.11a
     LowerLevelController → throttle/brake actuators

  8. DATA LOGGING
     All states, inputs, Nash metrics → sim_data
```

---

## Key Algorithms

### Rajamani Longitudinal Controller (Ch. 6.4)

Controls the constant time-headway (CTH) gap policy in the platoon:

```
s_des = L + h·v           (desired gap, h = 1.5 s time headway)
a_des = −k1·a_lead − k2·a_self − k3·Δv − k4·Δs − k5·v
```

String stability guaranteed by: `k1 < −τ/h` where τ = 0.1 s.

### Intelligent Driver Model (IDM)

Human driver behavior during free driving and prediction:

```
a_IDM = a_max·(1−(v/v0)^4) − a_max·(s*/s)²
s* = s_min + v·T + v·Δv/(2√(a·b))
```

### Nash Equilibrium Game (Pustilnik Non-Normalized GNE)

```
J1 = ||Z−R1||²_Q1 + R1·||u1||² + S1·||u2||²
J2 = α·||Z−R2||²_Q1 + R2·||u2||² + α·S2·||u1||²
```

Solved as coupled QP via OSQP with DPP-compliant CVXPY.

### Elliptic Safety Field

```
F = (M · R_i) / (r_elliptic + ε)²
```

Two-phase (MERGING / FOLLOWING) with hysteresis ensures comfort during steady-state following.

### Belousov Nonlinear EOM (Eq. 3.11a)

```
m_eff · v̇x = Fxf + Fxr − Ra − Rg − Rr + vy·Ωz·m_eff
```

Integrated via RK4 at 100 Hz. Nash sees double-integrator via feedforward cancellation.

---

## Configuration

**All parameters live in `config.py` — never hardcode values in module files.**

Key parameter groups:

| Group | Prefix | Location |
|---|---|---|
| Simulation timing | `SIMULATION_DT`, `NASH_CONTROL_DT` | `config.py` |
| Nash solver | `NASH_*` | `config.py` |
| Safety field | `SAFETY_*` | `config.py` |
| Authority allocator | `AUTHORITY_*` | `config.py` |
| System reference gen | `REFGEN_*` | `config.py` |
| Rajamani controller | `RAJAMANI_*` | `config.py` |
| Human driver | `HUMAN_*`, `NASH_DRIVER_PARAMS` | `config.py` |
| Lower-level controller | `LOWER_CTRL_*`, `ENGINE_BRAKING_*` | `config.py` |
| Vehicle hardware | `VehicleParameters` | `vehicle/components.py` |

---

## How to Run

```bash
cd Longitudinal
python main_with_nash.py
```

Choose scenario at prompt:
- `1` — Join before platoon (human starts 100 m ahead)
- `2` — Join middle of platoon (human starts near leader)
- `3` — Join after platoon (human starts 100 m behind)
- `4` — Run all scenarios
- `5` — Compare Nash vs. no-Nash

---

## Dependencies

```bash
pip install numpy scipy matplotlib osqp cvxpy
```

---

## Output

Results saved to `platoon_sim_kinematic_results/`:
- `comprehensive_plots_*.png` — 9-subplot analysis
- `nash_analysis_*.png` — game-theoretic metrics
- `animation_*.gif` — animated 4-panel visualization
- `scenario_summary_*.txt` — formatted metrics report

---

## Key References

| Reference | Usage in this codebase |
|---|---|
| Li et al. (2019) | Nash GNE framework, authority allocation |
| Rajamani, Vehicle Dynamics and Control (Ch. 6) | CTH platoon controller, transitional reference |
| Belousov et al. | Nonlinear EOM, effective mass, road grade resistance |
| Wang et al. (2015/2016) | Elliptic driving safety field |
| Swain & Rath (2023) | Sigmoid authority allocation (Eq. 15) |
