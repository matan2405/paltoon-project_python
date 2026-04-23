# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Simulations

**Dependencies** (no requirements.txt; install manually):
```bash
pip install numpy scipy matplotlib cvxpy
```

**Longitudinal simulation** (primary):
```bash
cd Longitudinal
python main_with_nash.py
# Interactive menu: 1=join before, 2=join middle, 3=join after, 4=all, 5=compare Nash vs non-Nash
```

**Lateral simulation**:
```bash
cd lateral
python "Main Application.py"
```

**Syntax-check a file** (no test suite exists):
```bash
python -m py_compile Longitudinal/main_with_nash.py
```

**Results output**: `Longitudinal/platoon_sim_kinematic_results/` (PNG plots, GIF animations, text summaries).

## Architecture Overview

This is a research codebase for vehicle platoon merging using Nash equilibrium-based shared control between an autonomous system (Player 1) and a human driver (Player 2).

### Two Parallel Implementations
- `Longitudinal/` — speed/gap control (primary focus, more complete)
- `lateral/` — lane-change control (parallel structure)

Both share the same conceptual framework: Safety Field → Authority Allocation → Nash Solve → Shared Control.

### Control Pipeline (per Nash step, every 0.1s)

```
main_with_nash.py::nash_control_step()
  1. safety_field.compute_risk_force_from_platoon()     → field_force
  2. authority_allocator.compute_authority_ratio(force)  → λ
  3. system_ref_generator.get_system_acceleration_and_state_sequence() → R1
  4. human_driver.get_human_acceleration_and_state_sequence()          → R2
  5. nash_solver.solve_nash_equilibrium(state, R1, R2, λ) → u1, u2
  6. u_shared = α·u1 + (1-α)·u2   where α = λ/(1+λ)
  7. vehicle.nash_acceleration = u_shared
```

Simulation runs at dt=0.02s (50 Hz); Nash control runs at 0.1s (10 Hz). Multi-rate handled via substeps.

### Nash Acceleration Override Pattern

The result is passed via `vehicle.nash_acceleration`, consumed in `platoon_control.py`:

```python
# platoon_control.py — PlatoonManager.update()
if not is_prediction_mode and not vehicle.nash_acceleration is None:
    a_des = vehicle.nash_acceleration
    vehicle.nash_acceleration = None  # MUST reset — prevents stale reuse
else:
    a_des = rajamani(...)  # Normal Rajamani or free-road control
```

Critical: the check uses `not ... is None` (not `!=`). The flag is only cleared when **not** in prediction mode, which is how human reference prediction avoids interference.

### Authority Ratio (λ)

`λ = 0` → human fully autonomous; `λ → ∞` → system takes control.
`α = λ/(1+λ)` maps this to [0, 1].
Critically, λ scales **Q2** (the human's tracking weight), not the control inputs directly — this preserves the Nash game structure while making the human "want" what the system wants under high risk.

### Nash Solver (Iterative Best Response)

Located in `Longitudinal/nash_solver/longitudinal_constrained_nash_solver.py`:
- Builds prediction matrices H1, H2, U from vehicle state-space matrices (A, B, C)
- Solves alternating: D1 = solve(H11, g1 - H12·D2), D2 = solve(H22, g2 - H21·D1)
- Clips to physical limits: u1 ∈ [-2.5, 2.0] m/s², u2 ∈ [-3.5, 2.5] m/s²
- Converges in ~3–8 iterations for quadratic/linear systems
- Requires `vehicle.use_state_space_model = True`

### Configuration

All parameters centralized in `Longitudinal/config.py` (469 lines). Key sections:
- `SIMULATION_DT = 0.02`, `NASH_dt = 0.1`
- Nash weights: `Q_output = diag([50.0, 200.0])`, `R1 = 50.0`, `R2 = 60.0`
- Safety field params (TTC, headway, gap error, relative velocity, velocity error)
- `HEADLESS_MODE` — set `True` for non-interactive/server environments

### Safety Field Sign Convention

- **Positive force**: repulsive (too close → push away)
- **Negative force**: attractive (too far → pull closer)
- Gap Error and Velocity Error are bidirectional; TTC and Headway are repulsive only
- `follower_weight = 0.5` reduces rear-vehicle contribution to total risk

### State Vector Conventions

```python
# Longitudinal Nash solver
state = [position_x, velocity_x]          # shape (2,)
R1_ref, R2_ref shape: (Np * 2, 1) = (40, 1)   # flattened [pos, vel, pos, vel, ...]

# Lateral Nash solver
state = [x, vx, y, vy]                    # shape (4,)
```

### Phase Detection (Longitudinal, V3.0)

`longitudinal_safety_field.py` tracks two phases:
- **MERGING**: Aggressive safety field for convergence
- **FOLLOWING**: Soft response for steady-state comfort

Transition requires all conditions (gap error, relative velocity, acceleration) to be satisfied for 5 continuous seconds.

## Key Files

| File | Role |
|------|------|
| `Longitudinal/main_with_nash.py` | Entry point; `PlatoonNashSimulation` extends base with Nash pipeline |
| `Longitudinal/config.py` | All parameters — change here first |
| `Longitudinal/nash_solver/longitudinal_constrained_nash_solver.py` | Nash QP solver |
| `Longitudinal/nash_solver/longitudinal_safety_field.py` | Risk field + phase detection |
| `Longitudinal/control/platoon_control.py` | Rajamani controller + Nash override |
| `Longitudinal/vehicle/vehicle.py` | Vehicle dynamics + state-space matrices |

## Common Pitfalls

- After reading `vehicle.nash_acceleration`, always set it to `None`
- Reference trajectories must be shape `(Np*2, 1)` — not `(Np, 2)`
- Gap calculation: `gap = leader.x - follower.x` (positive = safe)
- Nash solver needs `vehicle.use_state_space_model = True` for A/B/C matrices
- Use `NASH_dt` (not `SIMULATION_DT`) when calling `vehicle.get_state_space_matrices(dt)`
