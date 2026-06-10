# NashPlatoon — Unity HITL Implementation

Unity implementation of the Nash equilibrium shared-control algorithm for the Human-In-The-Loop (HITL) simulator. The design mirrors `unified/` (Python) exactly — same phases, same authority allocator, same R1 generators — with one deliberate difference: R2 is derived from real G29 inputs, not a synthetic driver model.

## Module Map

```
NashPlatoon/
├── Config/
│   ├── ReferenceGeneratorConfig.cs   — all R1/R2 parameters (single source of truth)
│   └── AuthorityConfig.cs            — authority allocator parameters
├── Control/
│   ├── NashCoordinator.cs            — Nash control loop + 4-phase state machine
│   ├── LongitudinalReferenceGen.cs   — R1/R2 longitudinal reference generators
│   └── LateralReferenceGen.cs        — R1/R2 lateral reference generators
├── NashSolver/
│   ├── LongitudinalNashSolver.cs     — DMPC-IBR solver, 2D state [x, vx]
│   └── LateralNashSolver.cs          — DMPC-IBR solver, 4D state [y, vy, ψ, ψ̇]
└── Authority/
    └── AuthorityAllocator.cs         — dynamic λ computation
```

---

## Human Reference (R2) — Design Rationale

### Foundation — Li & Song (2019)

Li & Song (Section 5.2) derive R2 empirically:

> *"The average path among the vehicle paths of 5 experiments under driver alone driving condition is chosen as the human driver's target path."*

In Unity HITL the real driver IS present — R2 = their measured current intent, which is the closest possible realisation of Li & Song's design. No synthetic model is needed.

Two implementation consequences:

1. **R2 uses real-time G29 measurements**, not a heuristic formula.
2. **R2 is held constant over the prediction horizon (ZOH)** — evaluate once at the current state, propagate Np steps.

### ZOH Justification — Zhang & Sun (2024)

The AR(1) temporal lengthscale of human acceleration from HighD is **ℓ = 1.51 s** (Zhang & Sun 2024). Since ℓ ≈ horizon (Np · dt = 2 s), the current acceleration is correlated across the full window. Evaluating once and holding is statistically sufficient — and consistent with Li & Song (their R2 is fixed for Np = 100 steps).

---

## Longitudinal R2 (`LongitudinalReferenceGen.HumanRef`)

### Old implementation (replaced)

```csharp
float uHuman = throttle * IdmAMax * 2f - brake * 8f;   // ad-hoc, no physics
```

Problems: arbitrary scaling, no gap awareness, IDM parameters in config were unused.

### Current implementation

```
1. Evaluate IDM once at current state (x, vx, gap, Δv)   ← ZOH
2. Propagate double integrator for Np steps with constant a₀
```

The real G29 throttle/brake pedals tell us the driver's *current* effort. IDM evaluated at the current state predicts *why* the driver is pressing the pedal — the gap-state drives the intent, not the instantaneous pedal position (which is transient). This is the same logic Li & Song use when they take the average of 5 driver-alone runs: what matters is the equilibrium behaviour at the current state.

### IDM Parameters (HighD, Zhang & Sun 2024)

T and b are driver-dependent properties, not speed-dependent, so HighD values (~68 km/h steady-state) apply at highway merge speed (≥100 km/h).

| Parameter | Field | Value | Source |
|---|---|---|---|
| Time headway | `IdmPlanT` | 2.585 s | HighD, Zhang & Sun 2024 |
| Comfortable decel | `IdmPlanB` | 1.649 m/s² | HighD, Zhang & Sun 2024 |
| Minimum gap | `IdmS0` | 3.416 m | HighD, Zhang & Sun 2024 |

### Phase Logic

| Phase | Leader used? | Rationale |
|---|---|---|
| Approach / GapSearch | No | Gap is large; free-road IDM gives a₀ > 0, cooperative with R1 |
| Merge / Following | Yes | Gap-aware IDM; R2 models intent to maintain safe following distance |

---

## Lateral R2 (`LateralReferenceGen.HumanRef`)

### Algorithm (two layers)

```
Layer 1 — psiDes (desired heading, ZOH)
  psiDes = atan(lf · δ_G29 / vx)           ← real G29 steering angle, computed once

Layer 2 — Neuromuscular IIR  (Swain & Rath 2023, Eq. 7)
  J·ψ̈ + B·ψ̇ + K·ψ = K·ψ_des   →   Euler-discretised IIR
  Converts desired heading → executed heading with muscle-tendon delay.
  Clamp: |ψ_exec| ≤ R2MaxHeadingDeg (2.6°)
```

No GP layer — the real driver's stochastic variability is already present in the G29 signal.

### Why the IIR clamp matters

The old clamp was ±0.3 rad (17.2°) — physically unrealistic. Lee & Olsen (2004) measured 8,667 lane changes at ~97 km/h and found a mean maximum heading angle of **2.6°**. At our merge speed of ≥100 km/h the same limit applies directly. The 6× reduction (17.2° → 2.6°) prevents the solver from chasing physically impossible R2 headings.

### Parameter Sources

| Parameter | Field | Value | Source |
|---|---|---|---|
| Max heading ψ_max | `R2MaxHeadingDeg` | 2.6° | Lee & Olsen 2004, 8,667 LCs ~97 km/h |
| Neuro J | `NeuroJ` | 0.0037 | Swain & Rath 2023, D1 neutral profile |
| Neuro B | `NeuroB` | 0.1363 | Swain & Rath 2023 |
| Neuro K | `NeuroK` | 1.1742 | Swain & Rath 2023 |

---

## Python vs Unity — Intentional Differences

| Component | Python (simulation) | Unity (HITL) | Intentional? |
|---|---|---|---|
| R2 long — psiDes source | IDM at current state | IDM at current state | same |
| R2 long — GP layer | SE-kernel GP residual | none | yes — HITL driver provides real noise |
| R2 lat — psiDes source | `atan2(Δy, horizon)` (synthetic) | `atan(lf·δ/vx)` (real G29) | yes — HITL measures real steering intent |
| R2 lat — IIR | Swain & Rath 2023 | Swain & Rath 2023 | same |
| R2 lat — GP layer | SE-kernel GP | none | yes — same rationale |
| FOLLOWING λ cap | `LONG_AUTHORITY_LAMBDA_MAX_FOLLOWING` | `LongLambdaMaxFollowing` | same |
| 4-phase state machine | APPROACH→GAP_SEARCH→MERGE→FOLLOWING | identical transitions | same |

The GP layer exists in Python to simulate the stochastic variability that Li & Song observed across 5 driver experiments. In Unity, that variability comes directly from the real driver — adding a synthetic GP on top would double-count it.

---

## Phase State Machine (`NashCoordinator.UpdatePhase`)

```
APPROACH ──(MOBIL approved)──► GAP_SEARCH ──(0.5 s elapsed)──► MERGE
                                                                    │
                                               IsMergeComplete()    │
                                               y-error < 0.3 m     │
                                               ψ < 0.05 rad         │
                                               gap-error < 20%     ▼
                                                              FOLLOWING
```

`NashModeActive` is an output flag set after each Nash solve — it is not a replacement for the phase machine.

---

## References

| Paper | Usage |
|---|---|
| Li & Song 2019 | R2 = measured human intent; Nash GNE framework |
| Zhang & Sun 2024 | HighD IDM parameters (T, b, s₀); ZOH justification (AR(1) ℓ=1.51 s) |
| Pustilnik & Borrelli 2025 | Non-normalised GNE cost formulation |
| Swain & Rath 2023 | Neuromuscular IIR (Eq. 7); lateral authority sigmoid (Eq. 15) |
| Lee & Olsen 2004 | Max heading at lane change: 2.6° at ~97 km/h |
| Rajamani Ch. 6.7 | R1 longitudinal transitional reference |
| Gu & Dolan | R1 lateral 5th-order polynomial |
