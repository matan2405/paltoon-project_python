# Nash Solver Module — Game-Theoretic Shared Control

## Overview

This module implements the core **shared control** mechanism between an autonomous platoon system and a human driver. It is based on **Distributed Model Predictive Control with Iterated Best Response (DMPC-IBR)**, formulated as a **non-cooperative Nash equilibrium game** following Li et al. (2019).

The module contains four components:

| File | Role |
|---|---|
| `longitudinal_constrained_nash_solver.py` | DMPC-IBR Nash solver (OSQP/CVXPY) |
| `longitudinal_safety_field.py` | Elliptic driving safety field (risk force) |
| `longitudinal_authority_allocator.py` | Dynamic authority ratio λ computation |
| `system_reference_generator.py` | Autonomous system reference trajectory |

---

## Game-Theoretic Framework

### Players

| Player | Identity | Goal |
|---|---|---|
| Player 1 (System) | Autonomous platoon controller | Track platoon's desired trajectory safely |
| Player 2 (Human) | Human driver | Follow their own preferred trajectory |

### Why Non-Cooperative?

The game is **non-cooperative** in the game-theoretic sense: each player minimizes their own cost function independently, with no binding agreement. Despite this, the authority allocation mechanism drives cooperative behavior — the system gradually takes over when risk increases.

### Cost Functions (Pustilnik Non-Normalized GNE)

```
J1(u1, u2) = ||Z − R1||²_Q1  +  R1·||u1||²  +  S1·||u2||²
J2(u1, u2) = α·||Z − R2||²_Q1  +  R2·||u2||²  +  α·S2·||u1||²
```

Where:
- `Z` = predicted state trajectory (Np × 2 flattened)
- `R1` = system reference (from `SystemReferenceGenerator`)
- `R2` = human reference (from `HumanDriver.get_human_acceleration_and_state_sequence`)
- `α = λ/(1+λ)` = system authority ratio (embedded in J2 via Pustilnik scaling)
- `Q1 = diag(Q_POS, Q_VEL)` with terminal weighting on last predicted state

**Key insight**: When λ is large (high risk), α → 1 and J2 aligns with J1 — the human's cost becomes identical to the system's, forcing cooperation.

### Cost Weights

| Parameter | Value | Meaning |
|---|---|---|
| `NASH_Q_POS` | 2500.0 | Position tracking weight |
| `NASH_Q_VEL` | 1500.0 | Velocity tracking weight |
| `NASH_R1` | 7500.0 | System control effort (R1 × 0.25) |
| `NASH_R2` | 12500.0 | Human control effort (R2 × 0.25) |
| `NASH_S1` | 10000.0 | System penalty on human's effort |
| `NASH_S2` | 10000.0 | Human penalty on system's effort |
| Terminal weight | 10×Q_POS, 4×Q_VEL | Emphasize final predicted state |

---

## Nash Solver (`longitudinal_constrained_nash_solver.py`)

### Prediction Horizon

| Parameter | Value | Meaning |
|---|---|---|
| `NASH_NP` | 20 steps | Prediction horizon = 2 s |
| `NASH_NU` | 10 steps | Control horizon = 1 s |
| `NASH_CONTROL_DT` | 0.1 s | Solver timestep |

### Solver: OSQP via CVXPY

The Nash equilibrium reduces to a **coupled QP** solved using **OSQP** (Operator Splitting QP solver). The formulation is **DPP-compliant** (Disciplined Parameterized Programming), enabling warm-start between timesteps.

```python
solver.solve_nash_equilibrium(
    vehicle, system_ref, human_ref, lambda_k, prev_u1, prev_u2
) → {'shared_input': u_shared, 'u1': u1_opt, 'u2': u2_opt, 'lambda': λ}
```

### Pre-computed Lambda Levels

Eight problems are pre-compiled at initialization for discrete authority levels:
```
NASH_LAMBDA_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
```

At runtime, λ is snapped to the nearest level to use the pre-compiled problem — this avoids re-compilation overhead at every timestep.

### Input Constraints

| Constraint | Value |
|---|---|
| System acceleration | u1 ∈ [−3.5, +2.5] m/s² |
| Human acceleration | u2 ∈ [−4.0, +3.0] m/s² |
| System jerk | Δu1 ≤ 1.5 m/s³ |
| Human jerk | Δu2 ≤ 2.0 m/s³ |
| Min velocity | 0 m/s |
| Max velocity | 50 m/s |
| Min gap | 7 m |

---

## Driving Safety Field (`longitudinal_safety_field.py`)

### Elliptic Potential Model (Wang et al. 2015/2016)

The safety field computes a **risk force** that quantifies how dangerous the current situation is. This force feeds the authority allocator.

```
F = (M · R_i) / (r_elliptic + ε)²
```

Where:
- `M` = effective obstacle mass (scaled by risk level)
- `R_i` = safety radius (scaled by velocity and position in platoon)
- `r_elliptic` = elliptic distance to obstacle
- `ε = 0.1` = regularization to prevent division by zero

### Velocity Scaling

Safety radius increases with speed:
```
R(v) = R_base · (1 + 0.8 · v / v_ref)     v_ref = 120 km/h
```

### Position-Based Multipliers

| Position in platoon | Force multiplier |
|---|---|
| Leader | 0.8× (less exposed from behind) |
| Middle | 1.2× (exposed both sides) |
| Follower | 1.0× (standard) |
| Joining (human) | 0.5× |

### Two-Phase Control

The safety field tracks which **control phase** the human vehicle is in:

#### MERGING Phase
- Applied when the vehicle is converging to its target gap
- Full force magnitude
- High authority ratio → system takes control

#### FOLLOWING Phase
- Applied after the vehicle is integrated into the platoon
- Force reduced quadratically for small errors (comfort)
- Lower authority ratio → human retains control

#### Phase Transition Logic (with hysteresis)

| Transition | Condition | Stability required |
|---|---|---|
| MERGING → FOLLOWING | \|gap_err\| < 15%·desired AND \|Δv\| < 5%·v_target AND \|a\| < 0.5 m/s² | 5 s continuous |
| FOLLOWING → MERGING | \|gap_err\| > 25%·desired OR \|Δv\| > 10%·v_target | Immediate |

The hysteresis prevents chattering near the transition boundary.

---

## Authority Allocator (`longitudinal_authority_allocator.py`)

### Purpose

Computes the authority ratio λ(k) that weights how much system control authority is injected into the Nash game.

### Safety Component (from field force)

```
λ_safety = λ_min + (λ_max − λ_min) · σ(k_steep · (|F| − F_mid))

where σ(x) = 1 / (1 + exp(−x))
```

| Parameter | Value |
|---|---|
| `AUTHORITY_LAMBDA_MIN` | 0.1 (human dominant, safe) |
| `AUTHORITY_LAMBDA_MAX` | 100.0 (system dominant, emergency) |
| `AUTHORITY_FORCE_MIDPOINT` | 400 N |
| `AUTHORITY_K_STEEPNESS` | 0.015 |

### Performance Component (from gap error)

When the vehicle is lagging behind or rushing ahead significantly, λ increases to correct the error faster:

```
Hysteresis thresholds:
  Enter performance mode: |gap_error| > 3 m
  Exit performance mode:  |gap_error| < 1 m
```

### Fusion

```
λ_k = max(λ_safety, λ_perf)
```

Then smoothed with an adaptive EMA:
```
λ_k = α_smooth · λ_target + (1−α_smooth) · λ_prev
```
- Fast smoothing (α=0.12) when gap error > 5 m
- Slow smoothing (α=0.05) when gap error < 2 m

### Authority Interpretation

```
α = λ/(1+λ)    →    system authority weight

λ = 0.1  → α = 0.09  (human almost fully in control)
λ = 1.0  → α = 0.50  (equal authority)
λ = 10   → α = 0.91  (system dominant)
λ = 100  → α = 0.99  (emergency, near-full system takeover)
```

---

## System Reference Generator (`system_reference_generator.py`)

### Purpose

Predicts the autonomous system's **desired future trajectory** (R1) using the Rajamani transitional controller (Chapter 6.7). This trajectory tells the Nash solver where the system *wants* the vehicle to go.

### Three Operating Modes

#### 1. CRUISE Mode
No leader detected within 150 m, or vehicle is at cruise speed:
```
a_des = a_max · (1 − (v / v_target)^4)     v_target = 1.15 · v_platoon_target
```

#### 2. COLLISION AVOIDANCE Mode
Triggered when gap < 2 m or TTC < 1.2 s:
```
a_des = −5.0 m/s²    (emergency braking)
```

#### 3. TRANSITIONAL Mode (normal merging)
Parabola-based velocity targeting (Rajamani Section 6.7.2):

```
v_target = v_leader ± √(2 · a_comfort · |gap_error|)

a_vel_tracking = K_v · (v_target − v_ego)

Near equilibrium (|gap_err| < 5 m):
  a_CTG = K1 · gap_error + K2 · R_dot
  a_des = (1−blend)·a_vel_tracking + blend·a_CTG
```

| Parameter | Value |
|---|---|
| `REFGEN_TIME_HEADWAY` | 1.5 s |
| `REFGEN_STANDSTILL_DISTANCE` | 2.0 m |
| `REFGEN_A_COMFORT` | 1.5 m/s² |
| `REFGEN_K_V` | 0.4 s⁻¹ |
| `REFGEN_K1` | 0.23 s⁻² |
| `REFGEN_K2` | 0.6 s⁻¹ |
| `REFGEN_CATCHUP_FACTOR` | 1.15 |
| `REFGEN_DETECTION_RANGE` | 150 m |
