# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

**Dependencies** (no requirements.txt):
```bash
pip install numpy scipy matplotlib cvxpy
```

**Entry point:**
```bash
python main_with_nash.py
# Menu: 1=join before, 2=join middle, 3=join after, 4=all, 5=compare Nash vs non-Nash
```

**Syntax check:**
```bash
python -m py_compile main_with_nash.py
python -m py_compile nash_solver/longitudinal_constrained_nash_solver.py
```

**Results** saved to `platoon_sim_kinematic_results/` — PNG plots, GIF animations, text summaries.

**Headless mode** (no display): set `HEADLESS_MODE = True` in `config.py`.

## Architecture

### Control Pipeline

Every Nash step (0.1s / 10 Hz), `nash_control_step()` in `main_with_nash.py` runs:

```
1. safety_field.compute_risk_force_from_platoon()        → field_force
2. authority_allocator.compute_authority_ratio(force)     → λ
3. system_ref_generator.get_..._sequence()               → R1 (Np steps)
4. human_driver.get_human_..._sequence()                 → R2 (Np steps)
5. nash_solver.solve_nash_equilibrium(state, R1, R2, λ)  → u1, u2
6. α = λ/(1+λ);  u_shared = α·u1 + (1-α)·u2
7. vehicle.nash_acceleration = u_shared
```

Vehicle dynamics run at dt=0.01s (100 Hz). Multi-rate is handled by substeps inside reference generators.

### Nash Acceleration Override

Result is consumed by `platoon_control.py → PlatoonManager.update()`:

```python
if not is_prediction_mode and not vehicle.nash_acceleration is None:
    a_des = vehicle.nash_acceleration
    vehicle.nash_acceleration = None   # MUST reset — prevents stale reuse
else:
    a_des = rajamani(...)   # or free_road_acc() for leader
```

- Uses `not ... is None` (not `!=`)
- Only applies when NOT in prediction mode (prevents interference with human reference rollout)
- Both leader and follower vehicles support this override

### Authority Allocation (Pustilnik V6.0)

λ scales **Q2** (human's tracking weight) inside the Nash cost — not the control output directly:
- `Q2_bar = λ · Q_output` → human "wants" what the system wants under high risk
- After solving equilibrium: `u_shared = α·u1 + (1-α)·u2` (two-stage: game then blend)
- λ ∈ [0.1, 100], α ∈ [0, 1]

### Nash Solver (V6.0 — Pustilnik α formulation)

`nash_solver/longitudinal_constrained_nash_solver.py`:
- Iterative best response: alternates `D1 = solve(H11, g1 - H12·D2)` and vice versa
- Clips per iteration: u1 ∈ [-3.5, 2.5] m/s², u2 ∈ [-4.0, 3.0] m/s²
- Converges in ~3–8 iterations; max 15 iterations
- Prediction matrices H1, H2, U are precomputed at init from `vehicle.get_state_space_matrices(dt=NASH_CONTROL_DT)`
- Requires `vehicle.use_state_space_model = True`

Cost structure:
```
J1 = ||z - r1||²_Q  + R1||u1||² + S1||u2||²
J2 = α||z - r2||²_Q + R2||u2||² + α·S2||u1||²
```

### Safety Field + Phase Detection (V3.0)

`nash_solver/longitudinal_safety_field.py`:
- Five force components: TTC, Headway, Gap Error, Relative Velocity, Velocity Error
- Gap Error and Velocity Error are **bidirectional**: positive = repulsive, negative = attractive
- Two phases: **MERGING** (aggressive) and **FOLLOWING** (soft comfort)
- Phase transition requires all conditions met for 5 continuous seconds (hysteresis)
- `compute_risk_force_from_platoon()` evaluates both leader and follower risk

### Hierarchical Control

When `use_hierarchical=True`, the upper level plans with the double-integrator and outputs `a_desired`. The lower-level controller (`control/lower_level_controller.py`) converts it to throttle/brake via:
1. Feedforward: `F_ff = m·a_des + F_drag + F_rolling`
2. PI feedback on acceleration error
3. Actuator mapping F → throttle/brake

### Vehicle Motion Models

Selected via `vehicle.set_motion_model(use_kinematic, use_state_space, use_hierarchical)`:
- **State-space** (default): double integrator with ZOH discretization — required for Nash
- **Kinematic**: fast, no engine dynamics
- **Hierarchical**: planning with double integrator, execution with full powertrain
- **Complex**: full engine (Audi TT torque curve), 6-speed transmission, aero drag

## Key Files

| File | Role |
|------|------|
| `main_with_nash.py` | Entry point; `PlatoonNashSimulation` extends base with Nash pipeline |
| `config.py` | All parameters — change here first |
| `nash_solver/longitudinal_constrained_nash_solver.py` | Nash QP solver V6.0 |
| `nash_solver/longitudinal_safety_field.py` | Risk field + phase detection V3.0 |
| `nash_solver/longitudinal_authority_allocator.py` | λ computation V5.2 |
| `nash_solver/system_reference_generator.py` | Rajamani Ch. 6.7 transitional reference |
| `control/platoon_control.py` | Rajamani controller + Nash override |
| `control/human_driver.py` | IDM-based human model with Np-step prediction |
| `vehicle/vehicle.py` | Vehicle dynamics + `get_state_space_matrices()` |

## Configuration Reference

All parameters in `config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SIMULATION_DT` | 0.01 s | Vehicle update rate (100 Hz) |
| `NASH_CONTROL_DT` | 0.1 s | Nash update rate (10 Hz) |
| `NASH_NP` | 20 | Prediction horizon (2 s) |
| `NASH_NU` | 10 | Control horizon (1 s) |
| `Q_pos` | 2500 | Position tracking weight |
| `Q_vel` | 50 | Velocity tracking weight |
| `R1 = R2` | 800 | Control effort weights |
| `h` | 1.5 s | Desired time headway |

## Common Pitfalls

- After reading `vehicle.nash_acceleration`, reset it to `None` — stale value persists otherwise
- Reference trajectories must be shape `(Np*2, 1)` — not `(Np, 2)`
- Always pass `dt=NASH_CONTROL_DT` (not `SIMULATION_DT`) to `get_state_space_matrices()`
- Gap sign: `gap = leader.x - follower.x` — positive means safe
- Force sign: positive = repulsive, negative = attractive
- Nash solver needs `use_state_space=True` at vehicle initialization
