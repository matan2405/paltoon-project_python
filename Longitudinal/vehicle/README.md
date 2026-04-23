# Vehicle Module — Longitudinal Dynamics

## Overview

This module provides the longitudinal vehicle model used throughout the platoon simulation. It implements a **nonlinear longitudinal equation of motion** (Belousov Eq. 3.11a) integrated via RK4 at the plant level, with successive linearization at each timestep to produce state-space matrices for the Nash MPC solver.

---

## Files

| File | Role |
|---|---|
| `vehicle.py` | Main `Vehicle` class — dynamics integration, state-space matrices, mode routing |
| `components.py` | `VehicleParameters`, `VehicleState`, `Engine`, `Transmission` |

---

## Physical Model

### Equation of Motion (Belousov Eq. 3.11a)

```
m_eff · (v̇x − vy·Ωz) = Fxf·cos(δf) + Fxr − Fyf·sin(δf) − Ra − Rg − Rr
```

| Symbol | Meaning | Value / Source |
|---|---|---|
| `m_eff` | Effective mass incl. rotating inertia | m + Iw/r² ≈ 1318 kg |
| `m` | Vehicle mass | 1305 kg (Audi TT 2.0 TFSI) |
| `Iw` | Wheel rotational inertia | 1.2 kg·m² |
| `r` | Wheel radius | 0.3 m |
| `Ra` | Aerodynamic drag | 0.5·ρ·Cd·A·vx², Cd=0.34, A=2.0 m² |
| `Rg` | Grade resistance | m·g·sin(ROAD_GRADE) |
| `Rr` | Rolling resistance | m·g·Cr·cos(ROAD_GRADE), Cr=0.013 |

### Feedforward Cancellation

The lower-level controller cancels all resistive forces, so the Nash MPC sees a **pure double integrator**:

```
v̇x = vy·Ωz + u_accel     ← Nash input u_accel [m/s²]
```

This keeps the prediction matrices simple (B_c = [[0],[1]]) while the full nonlinear physics are handled at the lower level.

---

## Two-Rate Simulation Architecture

```
SIMULATION_DT = 0.01 s (100 Hz)
    │
    ├── RK4: integrate nonlinear EOM → update x_long = [x, vx]
    ├── Update continuous Jacobians A_c, B_c at new operating point
    │
    └── (every 10 steps) NASH_CONTROL_DT = 0.1 s
             │
             ├── get_state_space_matrices(0.1) → ZOH → A_d, B_d
             └── Nash MPC reads A_d, B_d, solves for u_accel
```

**Never call `get_state_space_matrices(SIMULATION_DT)`** — the Nash solver must use the 0.1 s ZOH matrices.

---

## State Representation

### Internal plant state

```python
vehicle.x_long = np.array([x, vx])   # [m, m/s]
```

### Cached Jacobians (updated each sim step)

```python
vehicle.A_c_current   # 2×2 continuous A matrix
vehicle.B_c_current   # 2×1 continuous B matrix
```

### Discrete matrices (ZOH at NASH_CONTROL_DT, read by Nash)

```python
vehicle.A_d    # 2×2 discrete A
vehicle.B_d    # 2×1 discrete B
vehicle.A      # alias (same as A_d)
vehicle.B1     # system input channel
vehicle.B2     # human input channel
```

---

## Motion Models

The `Vehicle` class supports four motion models selectable via `set_motion_model()`:

| Model | Description | Use case |
|---|---|---|
| `'kinematic'` | Simple kinematic integration (no forces) | Fast prototyping |
| `'state_space'` | Double integrator ZOH | Nash-only runs |
| `'hierarchical'` | Upper: double integrator; Lower: full physics via `LowerLevelController` | **Preferred for research** |
| `'complex'` | Full nonlinear dynamics with engine/transmission | Engine analysis |

The **hierarchical** model is the default for Nash-enabled simulations: the Nash MPC plans using simple linear prediction, while actual motion uses the full powertrain model.

---

## Vehicle Hardware Parameters (Audi TT 2.0 TFSI)

| Parameter | Value |
|---|---|
| Mass | 1305 kg |
| Wheelbase | 2.505 m |
| Max velocity | 250 km/h (69.4 m/s) |
| Max acceleration | 2.5 m/s² |
| Max deceleration | −3.5 m/s² |
| Comfortable deceleration | −2.0 m/s² |
| Cd (drag coefficient) | 0.34 |
| Frontal area | 2.0 m² |
| Rolling resistance coeff | 0.013 |

### Engine (turbocharged)

- RPM range: 800–6700 RPM
- Peak torque: defined by piecewise interpolation table
- Turbocharger spool dynamics: torque builds up with a lag
- Gear ratios: 6-speed automatic (DCT-style with smooth blending during shifts)

---

## Nash Acceleration Interface

```python
# Set by Nash solver (main_with_nash.py):
vehicle.nash_acceleration = u_shared   # [m/s²]

# Read and applied by platoon_control.py:
if not is_prediction_mode and vehicle.nash_acceleration is not None:
    a_des = vehicle.nash_acceleration
    vehicle.nash_acceleration = None   # ← MUST reset to prevent stale reuse
```

---

## Key Design Decisions

1. **RK4 for plant accuracy** — avoids Euler integration errors at 100 Hz
2. **Successive linearization** — A_c, B_c recomputed at every timestep around current (x, vx)
3. **ZOH at Nash rate** — `scipy.linalg.expm` computes exact matrix exponential
4. **Feedforward cancellation** — lower controller handles physics so Nash stays in acceleration space
5. **Effective mass** — Iw/r² ≈ 13 kg accounts for wheel rotational inertia (≈1% of total)
