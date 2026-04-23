# Simulation Module — Master Orchestrator

## Overview

`simulator.py` contains the `PlatoonSimulation` class — the top-level orchestrator that wires together all subsystems: vehicle dynamics, platoon formation control, human driver, and data logging.

---

## Architecture

```
PlatoonSimulation
    ├── PlatoonManager          (controls autonomous vehicles)
    ├── Vehicle × N             (autonomous platoon members)
    ├── Vehicle (human)         (human-driven vehicle)
    ├── HumanDriver             (IDM model for human)
    └── Data Logger             (records all metrics)
```

---

## Simulation Parameters

| Parameter | Value | Source |
|---|---|---|
| Simulation timestep | 0.01 s (100 Hz) | `SIMULATION_DT` |
| Default duration | 120 s | `DEFAULT_SIMULATION_TIME` |
| Merge trigger time | 20 s (configurable) | scenario config |
| Number of platoon vehicles | 4 (configurable) | scenario config |

---

## Initialization Sequence

1. Create N autonomous vehicles in a line with proper initial spacing
2. Create human vehicle at scenario-specified initial position
3. Initialize `PlatoonManager` with autonomous vehicles
4. Initialize `HumanDriver` with human vehicle and driver type
5. Setup data logging structures

---

## Per-Step Update (`update(dt)`)

Each call to `update()` advances the simulation by one timestep:

```
1. PlatoonManager.update(dt)          → Rajamani control for all autonomous vehicles
2. HumanDriver.update(dt)             → IDM acceleration for human vehicle
3. [Nash step if enabled and on schedule]
   a. SystemReferenceGenerator        → predict system trajectory (R1)
   b. HumanDriver.get_sequence()      → predict human trajectory (R2)
   c. EllipseLongitudinalSafetyField  → compute risk force
   d. LongitudinalAuthorityAllocator  → compute λ
   e. ConstrainedNashSolver           → solve for u1, u2, u_shared
   f. vehicle.nash_acceleration = u_shared
4. All vehicles: vehicle.update(dt)   → dynamics integration (RK4)
5. log_data()                         → record all metrics
```

Nash runs every `NASH_CONTROL_DT / SIMULATION_DT` = 10 sim steps.

---

## Merge Sequence

At `t = merge_trigger_time` (default 20 s):

1. Human vehicle is added to the platoon: `PlatoonManager.add_vehicle(human_vehicle)`
2. The human vehicle is inserted by x-position (finds the correct gap in the platoon)
3. Gap tracking extends to include all gaps in the enlarged platoon
4. The human's Nash solver begins active operation

Before merge: human drives independently using IDM only.
After merge: Nash equilibrium shared control is active.

---

## Scenarios

Three built-in scenarios (configured in `main_with_nash.py`):

| Scenario | Human initial x | Human initial speed | Trigger time |
|---|---|---|---|
| 1 — Join Before Platoon | +100 m ahead of platoon | 100 km/h | 25 s |
| 2 — Join Middle | −10 m behind leader | 70 km/h | 20 s |
| 3 — Join After | −100 m behind platoon | 70 km/h | 15 s |

---

## Data Logging

`log_data()` records at each timestep:

| Category | Variables |
|---|---|
| Kinematics | x, y, vx, vy, ax positions for all vehicles |
| Control | throttle, brake, a_desired, a_actual |
| Powertrain | RPM, gear (when hierarchical mode active) |
| Gaps | actual gap, desired gap for each consecutive pair |
| Nash metrics | u1, u2, u_shared, λ, α (when Nash active) |
| Merge metrics | distance to target gap during merge |
| Phase | current phase (MERGING/FOLLOWING) |
| Time | simulation time vector |

---

## Output

The simulation produces raw data arrays that are passed to the visualization module. No files are written directly by `simulator.py` — the main script (`main_with_nash.py`) calls visualization functions after simulation completes.

Results are saved to: `platoon_sim_kinematic_results/`
