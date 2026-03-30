# Nash Solver Module — Lateral Game-Theoretic Shared Control

## Overview

The lateral Nash solver module implements the same **DMPC-IBR** (Distributed MPC with Iterated Best Response) framework as the longitudinal module, adapted for the 2D lateral lane-change problem. The key difference is that the state is 4-dimensional ([x, vx, y, vy]) and the safety field operates in 2D space.

**References**:
- Pustilnik & Borrelli (2025) — Non-normalized GNE for lateral merging
- Wang et al. (2015/2016) — 2D elliptic driving safety field
- Swain & Rath (2023) — Sigmoid authority allocation (Eq. 15)

| File | Role |
|---|---|
| `lateral_constrained_nash_solver.py` | DMPC-IBR Nash solver for lateral game |
| `lateral_safety_field.py` | 2D elliptic safety potential |
| `lateral_authority_allocator.py` | Authority ratio λ with lateral sigmoid |
| `system_reference_generator.py` | Autonomous convoy reference trajectory |
| `human_reference_generator.py` | Human's desired lateral trajectory |

---

## Game Formulation

### Players
- **Player 1 (System)**: Wants the human vehicle to reach its target position in the convoy
- **Player 2 (Human)**: Wants to follow their natural driving path

### Cost Functions

Same Pustilnik Non-Normalized GNE structure as longitudinal:

```
J1 = ||Z − R1||²_Q1 + R1·||u1||² + S1·||u2||²
J2 = α·||Z − R2||²_Q1 + R2·||u2||² + α·S2·||u1||²
```

Where `Z` is the predicted 4D state trajectory, `R1` is the system reference (target convoy position), `R2` is the human reference (natural driving path).

### Key Difference from Longitudinal

The prediction matrices are now **4×4** (A) and **4×1** (B), corresponding to the 4-state bicycle model. This makes the Hessian matrices larger but the structure identical.

---

## 2D Safety Field (`lateral_safety_field.py`)

### Elliptic Potential in 2D

The safety field now operates in both longitudinal (x) and lateral (y) directions:

```
r_elliptic = √((Δx/a_ellipse)² + (Δy/b_ellipse)²)

F_total = F_x · x̂ + F_y · ŷ
F_magnitude = (M · R) / (r_elliptic + ε)²
```

Where:
- `a_ellipse` = longitudinal semi-axis (larger — vehicles have more longitudinal extent)
- `b_ellipse` = lateral semi-axis (smaller — lane width constraint)
- `F_x`, `F_y` = force components in each direction

### Directional Force Decomposition

The force magnitude is projected onto longitudinal and lateral components based on the direction to the obstacle:
```
θ = atan2(Δy, Δx)
F_x = F_magnitude · cos(θ)
F_y = F_magnitude · sin(θ)
```

The **lateral component** is what feeds the authority allocator — it measures how much lateral risk exists from nearby vehicles.

---

## Authority Allocator (`lateral_authority_allocator.py`)

### Lateral Authority Sigmoid (Swain & Rath 2023, Eq. 15)

The lateral authority allocation uses a specific sigmoid formulation based on lateral position error:

```
α_lat = 1 / (1 + exp(−M1 · (|e_y| / e_y_max − M2)))
```

| Parameter | Value | Meaning |
|---|---|---|
| `AUTHORITY_SIGMOID_M1` | 2.0 | Slope steepness |
| `AUTHORITY_SIGMOID_M2` | 0.5 | Center shift (inflection at e_y = M2 · e_y_max) |
| `e_y_max` | `LANE_WIDTH/2 = 1.75 m` | Reference lateral error (half lane width) |

This maps lateral position error to authority:
- Small error (|e_y| ≈ 0): α_lat ≈ sigmoid(−M1·M2) ≈ low authority (human dominant)
- At half lane width (|e_y| = 1.75 m): α_lat ≈ 0.5 (equal authority)
- Large error: α_lat → 1.0 (system dominant)

### Emergency Intervention

Unlike the longitudinal system (which only uses λ scaling), the lateral system has a **hard override** mechanism:

```
if |y_ego − y_target| > EMERGENCY_THRESHOLD:
    apply direct lateral correction force
    (bypasses Nash game entirely)
```

This prevents the vehicle from crossing into oncoming traffic or leaving the road even if the Nash game produces a poor solution.

### Safety Component

Same sigmoid as longitudinal but applied to the 2D safety field force magnitude:
```
λ_safety = λ_min + (λ_max − λ_min) · sigmoid(k · (|F_lateral| − F_mid))
```

### Fusion

```
λ = max(λ_safety, λ_lateral_error)
```

---

## Reference Generators

### System Reference (`system_reference_generator.py`)

Generates the convoy's desired trajectory for the human vehicle:
- Target x-position: aligned with the convoy insertion point
- Target y-position: `y = 0.0 m` (left/convoy lane center)
- Velocity reference: matches convoy nominal velocity (33 m/s)

### Human Reference (`human_reference_generator.py`)

Generates what the human *would naturally do* without intervention:
- Predicts the human staying in their current lane (right lane, y = 1.75 m)
- If a lane change was already initiated, continues on the lane change trajectory
- Based on the human driver's lateral behavior model

This reference is `R2` in the Nash cost function — it represents human preferences that Nash must account for.

---

## Solver Parameters

| Parameter | Lateral | Longitudinal |
|---|---|---|
| Control rate | 20 Hz (0.05 s) | 10 Hz (0.1 s) |
| Prediction horizon Np | 20 | 20 |
| Control horizon Nu | 10 | 10 |
| State dimension | 4 | 2 |
| Input (u) | Lateral acceleration [m/s²] | Longitudinal acceleration [m/s²] |
| Solver backend | OSQP via CVXPY | OSQP via CVXPY |
| Warm start | Yes | Yes |

---

## Lambda Pre-computation

Same as longitudinal: 8 discrete λ levels pre-compiled at startup:
```
NASH_LAMBDA_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
```
