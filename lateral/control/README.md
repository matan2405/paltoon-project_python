# Control Module — Lateral Formation, Human Driver & Lane Change Decision

## Overview

The lateral control module handles three distinct concerns:

1. **`platoon_control.py`** — Formation control for the lateral convoy (Stanley heading + cross-track controller)
2. **`human_driver.py`** — Human lateral behavior model
3. **`mobil_lane_change.py`** — MOBIL model: decides *when* to initiate a lane change

---

## 1. Platoon Formation Control (`platoon_control.py`)

### Purpose
Maintains lateral spacing and alignment within the convoy using a **Stanley controller** for heading and cross-track error correction.

The lateral platoon control operates in concert with the Nash solver: Nash computes the optimal lateral inputs, while the platoon controller provides the baseline formation behavior when Nash authority is low (human is dominant).

### Control Law

**Stanley Controller:**
```
δ = ψ_e + arctan(k_e · e_crosstrack / v)
```

Where:
- `ψ_e` = heading error (vehicle heading vs. lane direction)
- `e_crosstrack` = signed lateral distance to lane centerline
- `k_e` = cross-track gain (from `DRIVER_PARAMS`)
- `v` = longitudinal velocity

### Lane Change Execution

When a lane change is triggered (by MOBIL or mandatory command):
- A smooth reference trajectory is generated (sinusoidal or parabolic path)
- The lateral position target transitions from current lane center to target lane center
- Duration: 3–5 seconds depending on speed and driver type

### Nash Injection
Same pattern as longitudinal:
```python
if not is_prediction_mode and vehicle.nash_acceleration is not None:
    u_lateral = vehicle.nash_acceleration
    vehicle.nash_acceleration = None  # mandatory reset
```

---

## 2. Human Driver (`human_driver.py`)

### Purpose
Models a human driver's lateral behavior — how they track their lane, respond to perturbations, and execute lane changes.

### Behavior Model

The human lateral driver is not purely IDM-based (IDM is longitudinal). Instead, the human lateral behavior is modeled as:
1. **Lane-keeping**: Proportional control to maintain lane center ± noise
2. **Lane-change response**: Following a smooth reference when changing lanes
3. **Reaction to platoon**: Modulated by driver type (cautious, normal, aggressive)

### Driver Types

| Type | Lane-keep stiffness | Lane change aggressiveness | Initial y position |
|---|---|---|---|
| `cautious` | High (slow drift back) | Low (gradual) | Right lane |
| `normal` | Medium | Medium | Right lane |
| `aggressive` | Low (accepts more deviation) | High (fast) | Right lane |

---

## 3. MOBIL Lane Change Decision (`mobil_lane_change.py`)

### Purpose
Decides **when** a lane change is safe and beneficial. Based on Kesting, Treiber & Helbing (2007) "General Lane-Changing Model MOBIL for Car-Following Models."

The Nash solver handles **how** to execute the lane change; MOBIL only decides **if** it should happen.

### MOBIL Criteria

#### Safety Criterion
The new follower (vehicle that will be behind the changing vehicle in the target lane) must not be forced to brake dangerously:
```
ã_new_follower ≥ −b_safe
```

| Parameter | Value | Meaning |
|---|---|---|
| `MOBIL_B_SAFE` | from config | Maximum acceptable deceleration for new follower [m/s²] |

#### Incentive Criterion
The lane change is only executed if the total benefit exceeds a threshold:
```
ã_ego − a_ego + p · (ã_new_follower − a_new_follower) > Δa_threshold + a_bias
```

Where:
- `ã` = acceleration after the lane change (predicted using IDM)
- `a` = current acceleration (before lane change)
- `p` = politeness factor: 0 = egoistic, 1 = fully cooperative

| Parameter | Value | Meaning |
|---|---|---|
| `MOBIL_P` | from config | Politeness factor |
| `MOBIL_A_TH` | from config | Minimum acceleration benefit to change [m/s²] |
| `MOBIL_A_BIAS` | from config | Bias toward right lane (asymmetric traffic rules) |
| `MOBIL_MIN_GAP` | from config | Minimum physical gap required [m] |

#### Mandatory Lane Change
When `mandatory_mode = True`, the incentive criterion is bypassed — only the safety criterion applies. Used for platoon merge maneuvers where the human *must* change lanes.

### IDM for Acceleration Prediction

MOBIL uses IDM internally to predict what accelerations would be before/after the lane change:

```
a_IDM = a_max · (1 − (v/v0)^δ) − a_max · (s*/s)²

s* = s0 + v·T + v·Δv / (2√(a·b))
```

| IDM Parameter | Config Key | Typical Value |
|---|---|---|
| Desired velocity | `MOBIL_IDM_V0` | 33 m/s (120 km/h) |
| Safe time headway | `MOBIL_IDM_T` | 1.5 s |
| Max acceleration | `MOBIL_IDM_A_MAX` | 2.5 m/s² |
| Comfortable decel | `MOBIL_IDM_B` | 2.0 m/s² |
| Min spacing | `MOBIL_IDM_S0` | 2.0 m |
| Exponent | `MOBIL_IDM_DELTA` | 4 |
| Vehicle length | `MOBIL_IDM_L` | 4.3 m |
