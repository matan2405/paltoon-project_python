# Vehicle Module — Lateral Bicycle Model

## Overview

This module provides the lateral vehicle dynamics model for the lane-change simulation. Unlike the longitudinal module (which uses Belousov's full EOM), the lateral module uses a **linearized bicycle model** appropriate for lateral (lane-change) maneuvers at near-constant longitudinal velocity.

---

## Files

| File | Role |
|---|---|
| `vehicle.py` | `Vehicle` class — bicycle model dynamics, state-space matrices |
| `components.py` | `VehicleParameters`, `VehicleState` — same Audi TT specs |

---

## Bicycle Model

The bicycle model simplifies the 2D vehicle into a single-track (bicycle) approximation, valid for small lateral accelerations and near-constant speed.

### State Vector

```python
state = [x, vx, y, vy]    # 4-state vector
```

| State | Meaning | Units |
|---|---|---|
| `x` | Longitudinal position | m |
| `vx` | Longitudinal velocity | m/s |
| `y` | Lateral position | m |
| `vy` | Lateral velocity | m/s |

The yaw angle (ψ) and yaw rate (Ωz) are computed from the bicycle kinematics but not part of the primary Nash state.

### Continuous State-Space

```
ẋ_lat = A_c · x_lat + B_c · u_lat
y_out = C_c · x_lat
```

The matrices `A_c`, `B_c` are derived from bicycle model linearization at the current operating point (velocity vx, cornering stiffnesses Cf, Cr).

### ZOH Discretization

```
A_d = expm(A_c · NASH_CONTROL_DT)
B_d = (A_d − I) · inv(A_c) · B_c
```

Read by Nash solver as `vehicle.A_d`, `vehicle.B_d`.

---

## Vehicle Parameters (Audi TT 2.0 TFSI)

Same physical parameters as the longitudinal module:

| Parameter | Value |
|---|---|
| Mass | 1305 kg |
| Wheelbase | 2.505 m |
| Front axle to CG | lf (from VehicleParameters) |
| Rear axle to CG | lr (from VehicleParameters) |
| Front cornering stiffness Cf | see VehicleParameters |
| Rear cornering stiffness Cr | see VehicleParameters |
| Nominal longitudinal velocity | 33.0 m/s (≈ 120 km/h) |

---

## Key Differences from Longitudinal Vehicle

| Aspect | Longitudinal | Lateral |
|---|---|---|
| State dimension | 2: [x, vx] | 4: [x, vx, y, vy] |
| Dynamics model | Belousov nonlinear EOM (RK4) | Linearized bicycle model |
| Nash input | u_accel [m/s²] (longitudinal) | u_steer [rad] (lateral) |
| Matrix dimensions | A: 2×2, B: 2×1 | A: 4×4, B: 4×1 |
| Nash control rate | 0.1 s (10 Hz) | 0.05 s (20 Hz) |

---

## Nash Control Rate

The lateral Nash solver runs at **20 Hz** (`NASH_CONTROL_DT = 0.05 s`), twice as fast as the longitudinal solver (10 Hz), because lateral dynamics are faster and require more frequent updates.

---

## Lane Geometry

```
Right lane center:  y = LANE_WIDTH / 2 = 1.75 m   (joining lane)
Left lane center:   y = 0.0 m                      (platoon lane)
Lane width:         LANE_WIDTH = 3.5 m
```

The target lateral position for a vehicle joining the convoy is `y = 0.0 m`.
