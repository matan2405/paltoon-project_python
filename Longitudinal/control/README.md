# Control Module — Platoon Formation, Human Driver & Actuator Control

## Overview

This module contains three distinct control layers that operate at different levels of abstraction:

1. **`platoon_control.py`** — Formation-level control: keeps autonomous vehicles in platoon using Rajamani's longitudinal controller
2. **`human_driver.py`** — Driver-level model: simulates human driving behavior using the Intelligent Driver Model (IDM)
3. **`lower_level_controller.py`** — Actuator-level control: converts desired acceleration to throttle/brake commands via PI feedback and feedforward

---

## 1. Platoon Formation Control (`platoon_control.py`)

### Purpose
Manages a multi-vehicle platoon, computing desired acceleration for each vehicle at every timestep.

### Lead Vehicle (Free Road)
The platoon leader accelerates asymptotically toward the target platoon velocity:

```
a_lead = a_max · (1 − (v / v_target)^δ)
```

| Parameter | Value | Source |
|---|---|---|
| `a_max` | 2.5 m/s² | VehicleParameters |
| `δ` (delta) | 4 | `FREE_ROAD_DELTA` |
| `v_target` | 120 km/h (33.3 m/s) | `PLATOON_TARGET_VELOCITY` |

### Follower Vehicles — Rajamani Controller (Ch. 6.4)

Each following vehicle tracks the vehicle ahead using Rajamani's inverse time-headway gap law.

**Desired gap:**
```
s_des = L_vehicle + h · v_follower
```

Where `h = 1.5 s` (time headway, `RAJAMANI_H`) and `L_vehicle` is vehicle length.

**Control law:**
```
a_des = −k1·a_lead − k2·a_follower − k3·Δv − k4·gap_error − k5·v_follower
```

**Gain derivation (Rajamani Eq. 6.15–6.16):**

| Gain | Value | Constraint |
|---|---|---|
| `k1` | −0.12 | k1 < −τ/h (string stability) |
| `k5` | 0.1 | k5 > 0 |
| `k2` | −k1 − h·k1·k5 | derived |
| `k3` | 1/h − k1·k5 | derived |
| `k4` | k5/h | derived |

`τ = 0.1 s` is the actuator time lag (`RAJAMANI_TAU`).

**String stability condition:** k1 < −τ/h ensures disturbances attenuate down the platoon.

### Jerk Limiting

Acceleration rate-of-change is capped per ISO 15622 ACC comfort standard:
```
da/dt ≤ JERK_LIMIT = 2.0 m/s³
```

### Nash Acceleration Override

When the Nash solver is active, the computed shared acceleration overrides the Rajamani command:
```python
if not is_prediction_mode and vehicle.nash_acceleration is not None:
    a_des = vehicle.nash_acceleration
    vehicle.nash_acceleration = None  # mandatory reset
```

---

## 2. Human Driver Model (`human_driver.py`)

### Purpose
Models a human driver using the **Intelligent Driver Model (IDM)**, both for real-time driving and for Nash prediction (predicting the human's future trajectory).

### IDM Algorithm

**Free-road acceleration:**
```
a_free = a_max · (1 − (v/v0)^δ)
```

**Interaction term (car-following):**
```
a_interaction = −a_max · (s* / s)²

where s* = s_min + v·T + v·Δv / (2·√(a_max · b))
```

**Total:**
```
a_IDM = a_free + a_interaction    (clipped to [a_min, a_max])
```

| Parameter | Execution Mode | Planning Mode |
|---|---|---|
| `T` (time headway) | 1.2 s | 0.8 s |
| `b` (comfortable decel) | 2.0 m/s² | 4.0 m/s² |
| `s_min` | 2.0 m | 2.0 m |

**Planning mode** is used when generating the human reference trajectory for Nash: more tolerant parameters to avoid prediction instability.

### Driver Types

| Type | `a_max` | Time headway | Initial x offset |
|---|---|---|---|
| `cautious` | 1.8 m/s² | 1.8 s | +40 m (starts further back) |
| `normal` | 2.5 m/s² | 1.2 s | 0 m |
| `aggressive` | 3.5 m/s² | 0.8 s | −20 m (starts closer) |

Driver type is selected from `NASH_DRIVER_PARAMS` in `config.py`.

### Prediction Interface

```python
states, accels = driver.get_human_acceleration_and_state_sequence(
    Np=20, dt=0.1, leader=optional_vehicle
)
```

Returns Np-step prediction of human states and accelerations (used as R2 reference by Nash solver).

---

## 3. Lower-Level Controller (`lower_level_controller.py`)

### Purpose
Converts a desired acceleration command (from Rajamani or Nash) into physical throttle/brake signals, compensating for vehicle physics.

### Control Architecture

```
a_desired  ──→  Feedforward  ──┐
                                ├──→  F_total  ──→  Actuator Mapping  ──→  (throttle, brake)
a_actual   ──→  PI Feedback  ──┘                         ↓
    ↑                                           Smoothing Filter
    └─────────────────────────────────────────────────────┘
```

### Stage 1: Feedforward

Compensates for resistive forces (cancellation of physics):
```
F_ff = m · a_desired + F_drag + F_rolling + F_grade
```

### Stage 2: PI Feedback

Corrects for model mismatch between desired and actual acceleration:
```
F_fb = Kp · m · (a_des − a_actual) + Ki · m · ∫(a_des − a_actual) dt
```

| Gain | Value |
|---|---|
| `Kp` | 0.4 |
| `Ki` | 0.08 |
| Anti-windup limit | 2.0 m/s²·s |

### Stage 3: Actuator Mapping

```
F_total = F_ff + F_fb

if F_total > +DEADBAND (50 N): → throttle = F_total / F_max_engine
if F_total < -DEADBAND (50 N): → brake    = |F_total| / F_max_brake
else:                           → coasting (engine braking only)
```

### Stage 4: Smoothing Filters

First-order exponential moving average prevents abrupt actuator commands:
```
throttle_cmd = α_throttle · throttle_new + (1−α_throttle) · throttle_prev
```

| Signal | α (smoothing coeff) |
|---|---|
| Throttle | 0.08 |
| Brake | 0.10 |
| Acceleration feedback | 0.06 |

### Engine Braking

When both throttle and brake are zero (coasting), pumping losses + friction produce a retarding torque:
```
T_brake = base_torque · (RPM / 2000)    capped at 120 Nm
```
`base_torque = 60 Nm`.

### Gear Shift Hold

During a transmission gear shift, the PI output is frozen to prevent chasing a transient force discontinuity. This avoids the "jerk on shift" artifact common in naive powertrain models.
