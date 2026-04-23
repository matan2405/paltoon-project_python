# Simulation Module — Lateral Orchestrator

## Overview

`simulator.py` contains the lateral `PlatoonSimulation` class — the master orchestrator for the lane-change merging scenario. It is structurally analogous to the longitudinal simulator but manages 2D lateral dynamics and the MOBIL lane change decision process.

---

## Key Differences from Longitudinal Simulator

| Aspect | Longitudinal | Lateral |
|---|---|---|
| State | [x, vx] per vehicle | [x, vx, y, vy] per vehicle |
| Nash rate | 10 Hz (0.1 s) | 20 Hz (0.05 s) |
| Merge direction | Front-to-back (longitudinal gap) | Lane-to-lane (lateral position) |
| Lane change decision | N/A | MOBIL model |
| Human reference | HumanDriver.get_sequence() | HumanReferenceGenerator |
| Results dir | `platoon_sim_kinematic_results/` | `lateral_sim_results_v2/` |

---

## Scenario

The lateral simulation models a single key scenario:

1. Human vehicle starts in the **right lane** (y = 1.75 m) at some longitudinal position
2. Convoy vehicles travel in the **left lane** (y = 0.0 m)
3. MOBIL model detects that a lane change is safe and beneficial (or mandatory mode triggers it)
4. Nash solver guides the human vehicle from right lane to left lane while matching convoy speed
5. Platoon controller accepts the human vehicle into the formation

---

## Per-Step Update

```
Every SIMULATION_DT = 0.01 s (100 Hz):

  1. PlatoonManager.update() → convoy formation control (left lane)
  2. HumanDriver.update()    → lateral behavior (right lane or transitioning)
  3. MOBIL check             → should lane change be triggered?

  Every NASH_CONTROL_DT = 0.05 s (20 Hz):

  4. SystemReferenceGenerator → R1 (convoy position reference, 20 steps)
  5. HumanReferenceGenerator  → R2 (human natural path, 20 steps)
  6. EllipseLateralSafetyField → lateral risk force
  7. LateralAuthorityAllocator → λ(t), check emergency override
  8. Nash solver              → u1*, u2*, u_shared
  9. vehicle.nash_acceleration = u_shared

  10. All vehicles: update()  → bicycle model integration
  11. log_data()
```

---

## Metrics Module

The lateral simulator has an additional `metrics/` directory (not present in longitudinal) with dedicated performance evaluation:
- Lateral error over time
- Lane change completion time
- Comfort metrics (lateral jerk, lateral acceleration profile)
- Authority ratio statistics

---

## Output

Results saved to `lateral_sim_results_v2/`:
- Static plots: lateral position, velocity, authority ratio, safety field force
- Animation: bird's-eye lane change visualization
- Metrics report: quantitative lane change quality assessment
