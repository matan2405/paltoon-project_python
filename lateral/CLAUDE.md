# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

**Dependencies** (no requirements.txt):
```bash
pip install numpy scipy matplotlib cvxpy
```

**Entry point:**
```bash
python main_with_menu.py
# Menu: 1=join before, 2=join middle, 3=join after, 4=all, 5=run with selection, 6=exit
```

**Results** saved to `convoy_simulation_results/` (or `lateral_sim_results_v2/`) — PNG plots, GIF animations.

**Headless mode**: `config.py` sets `matplotlib.use('Agg')` automatically when interactive backend fails.

## Architecture

### Control Pipeline

Every Nash step (0.05 s / 20 Hz), `nash_control_step()` in `simulator.py` runs:

```
1. safety_field.compute_risk_force()                             → field_force
2. authority_allocator.compute_authority_ratio(force)            → λ
3. system_ref_generator.generate()                               → R1 (Np steps, [y, ψ])
4. human_ref_generator.generate()                                → R2 (Np steps, [y, ψ])
5. nash_solver.solve(current_state, R1, R2, λ)                  → δ1, δ2
6. δ_shared = δ1 + δ2   (NOT α·δ1 + (1-α)·δ2 — see GNE note)
7. vehicle.update_dynamics(dt, δ_shared)
```

Vehicle dynamics run at dt=0.01s (100 Hz). Nash runs at 0.05s (20 Hz) — simulator reuses the last Nash result for intermediate dynamics steps.

### Key Difference from Longitudinal: Non-Normalized GNE

The lateral module uses **Generalized Nash Equilibrium (Pustilnik & Borrelli 2025)** solved with CVXPY (OSQP), not iterative best response. Control outputs are **added** (`δ_shared = δ1 + δ2`), not blended. Authority allocation is embedded inside the QP via the Pustilnik α scaling in cost matrices — there is no external blending step.

Cost structure:
```
J1 = ||z - r1||²_Q1  + R1||u1||² + S1||u2||²
J2 = α||z - r2||²_Q1 + R2||u2||² + α·S2||u1||²
where α = λ/(1+λ)
```

### Dual Propagation Vehicle Model

`vehicle/vehicle.py` maintains **two parallel state vectors** updated by the same steering input δ:
- **Body-frame** `[y, ẏ, ψ, ψ̇]` — realistic dynamics, used for actual position tracking
- **Error-model** `[e1, ė1, e2, ė2]` where e1=y_error, e2=ψ_error — read directly by Nash solver

`get_state_vector()` returns the error-model state; `get_state_space_matrices()` returns A_error_d, B_error_d, C. Always use `NASH_CONTROL_DT` (not `SIMULATION_DT`) when building matrices.

### Phase State Machine

`nash_solver/lateral_safety_field.py` implements:
```
CRUISE → GAP_SEARCH (0.5 s) → LANE_CHANGE → LANE_KEEPING → FOLLOWING
```
Each phase triggers different behavior in the reference generators. FOLLOWING entry requires 5 s of stability.

### Reference Generators (V4.0)

Both system and human references use polynomial trajectories:
- **System** (`system_reference_generator.py`): 5th-order polynomial, T_lc × 1.5 (slower/safer)
- **Human** (`human_reference_generator.py`): 3rd-order polynomial, base T_lc (faster)

V4.0 adds **soft LANE_KEEPING transition** — captures current (y, ψ) on phase entry and generates a settling trajectory to the target, preventing reference discontinuities.

Heading reference is feedforward from `ψ_ref = dy/dt / vx` during lane change (previously always zero — caused heading fights).

### Driver Personality Propagation

All components receive `driver_type` at scenario setup (`'cautious'`, `'normal'`, `'aggressive'`):
- Stanley gains `k_e`, `k_psi` in `HumanDriver`
- Trajectory duration and max heading angle in reference generators
- MOBIL politeness factor in `mobil_lane_change.py`

Nash game weights (R1, R2, S1, S2, Q_y, Q_psi) are **fixed** — driver type does NOT modify the Nash solver cost matrices.

### Authority Allocator (V3.0 — aligned with Longitudinal V5.2)

Two authority sources, fused by `max()`:
1. **Safety** — sigmoid on `risk_force` magnitude
2. **Performance** — hysteresis on `|y_error|` (mirrors `gap_error` in longitudinal):
   - Enter performance mode when `|y_error| > AUTHORITY_ENTER_THRESHOLD` (1.0 m)
   - Exit when `|y_error| < AUTHORITY_EXIT_THRESHOLD` (0.3 m)

Adaptive smoothing: `alpha_fast` when `|y_error|` is large, `alpha_base` otherwise.
All thresholds live in `config.py` under `AUTHORITY_*`.

### Human Driver: Stanley Controller

`control/human_driver.py` uses the Stanley path-tracking law:
```
δ = -k_e · y_error - k_psi · ψ_error
```
with low-pass filtering. Driver type sets gains (normal: k_e=0.005, k_psi=0.5).

### MOBIL Lane Change Decision

`control/mobil_lane_change.py` gates the merge. Before the Nash controller activates, MOBIL checks:
1. **Safety**: new follower won't have `a < -b_safe`
2. **Incentive**: lane change improves ego + weighted gain of neighbors

## Key Files

| File | Role |
|------|------|
| `main_with_menu.py` | Entry point, scenario/driver selection, output |
| `config.py` | All parameters — change here first |
| `simulation/simulator.py` | `LateralSimulation` class, main step loop |
| `nash_solver/lateral_constrained_nash_solver.py` | CVXPY-based GNE solver V6.0 |
| `nash_solver/lateral_safety_field.py` | Phase state machine + risk force |
| `nash_solver/lateral_authority_allocator.py` | λ computation V2.2 |
| `nash_solver/system_reference_generator.py` | 5th-order polynomial reference V4.0 |
| `nash_solver/human_reference_generator.py` | 3rd-order polynomial reference V4.0 |
| `control/human_driver.py` | Stanley controller |
| `control/mobil_lane_change.py` | MOBIL lane-change decision |
| `vehicle/vehicle.py` | 2-DOF bicycle model, dual propagation |

## Configuration Reference

All parameters in `config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SIMULATION_DT` | 0.01 s | Vehicle dynamics rate (100 Hz) |
| `NASH_CONTROL_DT` | 0.05 s | Nash rate (20 Hz) |
| `NASH_NP` | 20 | Prediction horizon (1 s) |
| `NASH_NU` | 10 | Control horizon (0.5 s) |
| `LANE_WIDTH` | 3.5 m | Standard highway lane |
| `PLATOON_LANE_Y` | 0.0 m | Platoon travels here |
| `HUMAN_INITIAL_LANE_Y` | 3.5 m | Human starts here |
| `NOMINAL_VELOCITY` | 20.0 m/s | ~72 km/h |
| `NASH_Q_Y` | 10.0 | Lateral position tracking |
| `NASH_Q_PSI` | 10000.0 | Heading tracking (high for stability) |
| `NASH_R1 = NASH_R2` | 1 000 000 | Control effort weights |
| `NASH_S1 = NASH_S2` | 200 000 | Cooperation coupling weights |

## Common Pitfalls

- Lateral module uses GNE (additive outputs) not iterative best response (blended outputs) — do not confuse the two merge strategies
- Q_psi >> Q_y intentionally — heading must stabilize before lateral position
- `get_state_space_matrices()` returns error-model matrices, not body-frame — ensure consistent usage
- Reference generators must be reset/re-initialized when phase changes
- Scenarios: join_before (x=100m, t_merge=3s), join_middle (x=35m, t_merge=3s), join_after (x=-20m, t_merge=5s)
- Driver type must be propagated to ALL components: Stanley, reference generators, MOBIL — but NOT Nash weights (fixed)
