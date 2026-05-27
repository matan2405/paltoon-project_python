# Unified Nash Shared-Control Module

The `unified/` module is the canonical implementation of the Nash equilibrium shared-control algorithm. It is used directly by the Python simulation and serves as the reference design for the Unity HITL simulator.

## Module Map

```
unified/
├── config.py               — all parameters (single source of truth)
├── control/
│   └── coordinator.py      — Nash control loop, R1/R2 generators, authority allocation
├── nash_solver/
│   ├── longitudinal_constrained_nash_solver.py
│   └── lateral_constrained_nash_solver.py
├── vehicle/
│   └── vehicle_6d.py       — 6-DOF Belousov EOM + ZOH discretisation
└── simulation/
    └── simulator.py        — scenario runner
```

---

## Human Reference (R2) — Design Rationale

R2 is the human player's desired trajectory in the Nash game. Its design follows two principles: it should reflect measured human behaviour, and it should be evaluated at the current state and held constant over the prediction horizon (ZOH).

### Foundation — Li & Song (2019)

Li & Song (Section 5.2) derive R2 empirically:

> *"The average path among the vehicle paths of 5 experiments under driver alone driving condition is chosen as the human driver's target path."*

Two implementation consequences:

1. **R2 should capture current human intent**, not a pre-planned trajectory.
2. **R2 is held constant over the horizon** (ZOH) — Li & Song use Np = 100 with a fixed R2; re-evaluating it at the current state each Nash step is the closest rolling equivalent.

### ZOH Justification — Zhang & Sun (2024)

The AR(1) temporal lengthscale of human acceleration measured on HighD is **ℓ = 1.51 s**. Since ℓ ≈ horizon (Np · dt = 2 s), the human's acceleration is correlated across the full horizon — evaluating IDM once at k=0 and holding it forward is statistically equivalent to rolling integration. This independently confirms ZOH for the longitudinal model.

---

## Longitudinal R2 (`_long_hum_ref`)

### Algorithm

```
1. Evaluate IDM once at current state (x, vx, gap, Δv)       ← ZOH
2. Propagate double integrator for Np steps with constant a₀
3. Add SE-kernel GP residual per step (Python only)           ← stochastic noise layer
```

### IDM Parameters

HighD-calibrated (Zhang & Sun 2024, passenger cars, ~68 km/h steady-state following). T and b are driver-dependent properties, not speed-dependent, making them applicable at highway merge speeds.

| Parameter | Value | Source |
|---|---|---|
| Time headway T | 2.585 s | HighD, Zhang & Sun 2024 |
| Comfortable decel b | 1.649 m/s² | HighD, Zhang & Sun 2024 |
| Minimum gap s₀ | 3.416 m | HighD, Zhang & Sun 2024 |

### Phase Logic

| Phase | Leader used? | Rationale |
|---|---|---|
| APPROACH / GAP_SEARCH | No | Gap is large; free-road acceleration is cooperative with R1 |
| MERGE / FOLLOWING | Yes | Gap-aware IDM; R2 models the human's intent to maintain a safe gap |

---

## Lateral R2 (`_lat_hum_ref`)

### Algorithm (three layers)

```
Layer 1 — psiDes (desired heading, ZOH)
  Python:  psiDes = clip(atan2(target_y − y,  vx·Np·dt), ±ψ_max)
  Unity:   psiDes = atan(lf · δ_G29 / vx)                         ← real G29 input

Layer 2 — Neuromuscular IIR  (Swain & Rath 2023, Eq. 7)
  J·ψ̈ + B·ψ̇ + K·ψ = K·ψ_des   →   Euler-discretised IIR
  Converts desired heading → executed heading with muscle-tendon delay.

Layer 3 — GP residual  (Python only, Zhang & Sun 2024)
  SE-kernel GP models stochastic deviation around mean path.
  psi_gp ~ GP(0, σ_k²·exp(−|Δt|²/(2ℓ²)))
```

### Why proportional heading (Layer 1, Python)?

In Python there is no real driver. The proportional formula `atan2(Δy, horizon)` gives the heading that, held constant, brings the vehicle from its current y to target_y over the prediction horizon. This is:
- **Adaptive**: re-computed at each Nash step from the current y, not locked to a polynomial entry state
- **Consistent with ZOH**: a single scalar psiDes, constant for all Np steps
- **Consistent with Li & Song**: R2 reflects where the driver is now, not where they planned to be at merge entry

### Parameter Sources

| Parameter | Value | Source |
|---|---|---|
| Max heading ψ_max | 2.6° (0.04538 rad) | Lee & Olsen 2004 — 8,667 LCs at ~97 km/h |
| Neuro J | 0.0037 | Swain & Rath 2023, D1 neutral profile |
| Neuro B | 0.1363 | Swain & Rath 2023 |
| Neuro K | 1.1742 | Swain & Rath 2023 |
| GP σ_k (lat) | 0.018 rad | Zhang & Sun 2024, Table I |
| GP ℓ (lat) | 0.65 s | Zhang & Sun 2024, Table I |

The 2.6° limit is applicable because Lee & Olsen measured at ~97 km/h, matching the simulation's merge speed of ≥100 km/h.

---

## Python vs Unity — Intentional Differences

| Component | Python (simulation) | Unity (HITL) | Intentional? |
|---|---|---|---|
| R2 long — input | IDM at current state (ZOH) | IDM at current state (ZOH) | same |
| R2 long — GP | SE-kernel GP residual | none | yes — HITL needs no synthetic noise |
| R2 lat — psiDes | `atan2(Δy, horizon)` (synthetic) | `atan(lf·δ/vx)` (real G29) | yes — HITL measures real intent |
| R2 lat — IIR | Swain & Rath 2023 | Swain & Rath 2023 | same |
| R2 lat — GP | SE-kernel GP | none | yes — same as longitudinal |
| FOLLOWING λ cap | `LONG_AUTHORITY_LAMBDA_MAX_FOLLOWING` | `LongLambdaMaxFollowing` | same |
| 4-phase state machine | APPROACH→GAP_SEARCH→MERGE→FOLLOWING | identical transitions | same |

The GP layer exists in Python to simulate the stochastic variability around the mean driver path (as in Li & Song's 5-experiment average). In Unity, the real driver's variability is already present in the G29 measurements — adding a synthetic GP layer on top would double-count the noise.
