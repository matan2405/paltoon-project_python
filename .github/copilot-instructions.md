# Platoon Merging Simulation with Nash Equilibrium Control

## Project Overview
This is a research codebase for simulating vehicle platoon merging with **Nash equilibrium-based shared control** between autonomous systems and human drivers. The project implements **bidirectional safety fields** and **game-theoretic authority allocation** for safe and comfortable merging maneuvers.

### Two Main Implementations
1. **Longitudinal/** - Platoon merging in longitudinal direction (primary focus)
2. **lateral/** - Convoy merging in lateral direction (lane changing)

Both share the same Nash equilibrium framework but differ in dynamics and safety field implementations.

## Architecture

### Core Control Loop (Nash Equilibrium)
The system follows this cycle for every simulation step:

```
1. Safety Field Assessment → 2. Authority Allocation → 3. Trajectory Planning → 4. Nash Solving → 5. Shared Control Application
```

**Key insight**: The Nash solver computes equilibrium between two players (autonomous controller & human driver), weighted by a dynamic authority ratio (λ) that increases with risk.

### Component Interactions

#### `main_with_nash.py` (Entry Point)
- Extends `PlatoonSimulation` with Nash control via `PlatoonNashSimulation` class
- Coordinates all subsystems through `nash_control_step()`
- Implements three scenarios: join before/middle/after platoon
- **Critical method**: `update_with_nash()` - calls Nash control if vehicle is merging or joined

#### Nash Control Pipeline
```python
# From nash_control_step() in main_with_nash.py
field_force = safety_field.compute_risk_force_from_platoon()  # Step 1
lambda_k = authority_allocator.compute_authority_ratio(field_force)  # Step 2
accel_seq_controller, state_seq_controller = system_ref_generator.get_system_acceleration_and_state_sequence()  # Step 3a
accel_seq_human, state_seq_human = human_driver.get_human_acceleration_and_state_sequence()  # Step 3b
u1_opt, u2_opt = nash_solver.solve_nash_equilibrium(state, R1_ref, R2_ref, lambda_k)  # Step 4
u_shared = alpha * u1_opt + (1 - alpha) * u2_opt  # Step 5 (alpha = λ/(1+λ))
```

#### Safety Field (Bidirectional)
Located in `nash_solver/longitudinal_safety_field.py`:
- **Five components**: TTC, Headway, Gap Error, Relative Velocity, Velocity Error
- **Bidirectional forces**: Gap Error and Velocity Error can be attractive (negative) or repulsive (positive)
- **Sign convention**: Positive error (deficiency) → Negative force (attractive), Negative error (excess) → Positive force (repulsive)
- `compute_risk_force_from_platoon()` evaluates both leader and follower risks
- `get_leader_and_follower()` integrates with `PlatoonManager` to find adjacent vehicles
- Returns detailed `breakdown` dict with individual force components

#### Vehicle Dynamics
The codebase supports **three motion models** (see `vehicle/vehicle.py`):
1. **Complex dynamics** (default off) - Full powertrain simulation
2. **Kinematic bicycle model** - Simplified geometric model
3. **State-space bicycle model** (preferred) - Linear A/B/C matrices for Nash solver

**Important**: Nash solver requires state-space model for prediction matrices. Set `use_state_space=True` when initializing.

#### Platoon Control - Nash Acceleration Override
`control/platoon_control.py` implements:
- **Rajamani controller** for following vehicles (uses desired time headway h=1.5s)
- **Free road acceleration** for leader vehicle (IDM-like behavior with delta=4 exponent)
- **Critical pattern**: `PlatoonManager.update()` checks for Nash override:

```python
# ACTUAL CODE PATTERN - for both leader and followers
if not is_prediction_mode and not vehicle.nash_acceleration is None:
    a_des = vehicle.nash_acceleration
    vehicle.nash_acceleration = None  # Reset for next iteration
    # Apply acceleration...
else:
    # Normal Rajamani or free road acceleration
```

**Why this matters**: 
- The check is `not vehicle.nash_acceleration is None` (not using `hasattr`)
- Must reset to `None` after reading to prevent reuse
- Only applies when NOT in prediction mode
- Works for both leader (free road) and followers (Rajamani)

### Data Flow Pattern
1. **Initialization**: `PlatoonSimulation.__init__()` creates vehicles, human driver, platoon manager
2. **Nash Extension**: `PlatoonNashSimulation.__init__()` adds Nash components (safety field, authority allocator, solver)
3. **Simulation Loop**: `update_with_nash()` → `nash_control_step()` → sets `vehicle.nash_acceleration` → platoon control applies it
4. **Vehicle Updates**: Each vehicle calls `update_dynamics()` with its assigned acceleration

## Key Conventions

### Coordinate System
- **X-axis**: Longitudinal (forward direction, increasing)
- **Y-axis**: Lateral (lane position)
- **Longitudinal**: Y=0 for all platoon vehicles (single lane)
- **Lateral**: Right lane (y=3.5m), Left lane (y=0.0m)

### State Vector Format
```python
# For Nash solver (longitudinal)
state = [position_x, velocity_x]  # Shape: (2,)
# Reference trajectories are flattened: (Np * 2, 1)

# For lateral Nash solver
state = [x, vx, y, vy]  # Shape: (4,)
```

### Time Steps
- **Simulation dt**: 0.02s (50 Hz) - defined in `config.py` as `SIMULATION_DT`
- **Control dt (Nash)**: 0.1s (10 Hz) - multi-rate control
- **Prediction horizon (Np)**: 20 steps (2 seconds ahead)
- **Control horizon (Nu)**: 10 steps (1 second ahead)

### Nash Solver Parameters
Located in `main_with_nash.py` and `nash_solver/longitudinal_nash_solver.py`:
```python
Q_output = np.diag([50.0, 200.0])  # Position/velocity tracking weights
R1 = 50.0   # Controller effort cost (smoother = higher)
R2 = 60.0   # Human effort cost (prefer less aggressive)
```
**Tuning tip**: Increase R1/R2 for smoother control, increase Q for tighter tracking.

## DMPC-Based Nash Equilibrium Solution

### Mathematical Framework
The Nash solver uses **Distributed Model Predictive Control (DMPC)** to find the equilibrium between two players (autonomous system and human driver). This is based on the game-theoretic approach from **Li et al. (2019)** "Shared control with a novel dynamic authority allocation strategy based on game theory and driving safety field".

#### Why "Non-Cooperative" Game?

**Important clarification**: The game is **non-cooperative** in the game-theoretic sense, even though the players exhibit cooperative behavior:

**Non-Cooperative Definition:**
- Each player optimizes **their own cost function independently**
- No binding agreement or shared cost function
- Nash equilibrium: Neither player can improve their cost by unilateral change

**This is NOT Cooperative Game** where players would minimize a single combined cost: `J_total = J1 + J2`

**The Key Insight from Li et al.:**
```
"Cooperative behavior emerges from non-cooperative game through authority allocation"

- When risk is LOW (λ ≈ 0): Q2 ≈ 0 → Human follows own preferences
- When risk is HIGH (λ → ∞): Q2 → ∞ → Human's cost aligns with System's
- Result: Smooth transition from human autonomy to system control
```

This preserves human agency (non-cooperative structure) while achieving cooperation when needed (via dynamic weighting).

#### Two-Player Game Structure
**Players:**
- Player 1: Autonomous controller (wants safe, efficient platoon integration)
- Player 2: Human driver (has own preferences and comfort zones)

**State-Space Prediction Model:**
```python
# Discrete-time state-space (obtained from vehicle.get_state_space_matrices())
x[k+1] = A·x[k] + B1·u1[k] + B2·u2[k]
z[k] = C·x[k]

# For longitudinal:
# x = [position, velocity]ᵀ  (nx=2)
# z = [position, velocity]ᵀ  (nz=2)  
# u1, u2 = acceleration commands (nu=1)
```

**Prediction over horizon Np:**
```python
Z = U·x0 + H1·D1 + H2·D2

where:
  Z = [z[1]; z[2]; ...; z[Np]]     # Predicted outputs (Np×nz, 1)
  D1 = [u1[0]; u1[1]; ...; u1[Nu-1]]  # Player 1 controls (Nu, 1)
  D2 = [u2[0]; u2[1]; ...; u2[Nu-1]]  # Player 2 controls (Nu, 1)
  
  U = [C·A; C·A²; ...; C·A^Np]     # Free response matrix
  H1, H2 = prediction matrices built from A, B1, B2, C
```

#### Cost Functions (Quadratic)
```python
# Player 1 (System) - wants to track system reference R1
J1(D1, D2) = ||R1 - Z||²_Q1 + ||D1||²_R1
           = (R1 - U·x0 - H1·D1 - H2·D2)ᵀ·Q1·(R1 - U·x0 - H1·D1 - H2·D2) + D1ᵀ·R1·D1

# Player 2 (Human) - wants to track human reference R2  
J2(D1, D2) = ||R2 - Z||²_Q2 + ||D2||²_R2
           = (R2 - U·x0 - H1·D1 - H2·D2)ᵀ·Q2·(R2 - U·x0 - H1·D1 - H2·D2) + D2ᵀ·R2·D2

where:
  Q1 = diag(Q_output, Q_output, ..., Q_output)  # Np repetitions
  Q2 = λ·Q1  # Scaled by authority ratio!
  R1 = R1·I  # Nu×Nu identity scaled
  R2 = R2·I
```

**Critical insight**: Q2 is scaled by λ (authority ratio), not the control inputs. This means:
- High risk (large λ) → System reference tracking becomes more important in human's cost
- Low risk (small λ) → Human can follow own preferences more freely

#### Nash Equilibrium Conditions
At equilibrium (D1*, D2*), neither player can improve their cost by unilateral deviation:
```
∂J1/∂D1 = 0  →  H11·D1 + H12·D2 = g1
∂J2/∂D2 = 0  →  H21·D1 + H22·D2 = g2

where:
  H11 = H1ᵀ·Q1·H1 + R1·I
  H12 = H1ᵀ·Q1·H2
  H21 = H2ᵀ·Q2·H1
  H22 = H2ᵀ·Q2·H2 + R2·I
  
  e1 = R1 - U·x0  (system tracking error)
  e2 = R2 - U·x0  (human tracking error)
  g1 = H1ᵀ·Q1·e1
  g2 = H2ᵀ·Q2·e2
```

### Iterative Best Response Algorithm
The solver uses **alternating optimization** to find Nash equilibrium:

```python
# Initialize
D1 = zeros(Nu)
D2 = zeros(Nu)

# Iterate until convergence
for iteration in range(15):
    D1_prev = D1
    D2_prev = D2
    
    # Player 1 best response (given D2)
    D1 = solve(H11, g1 - H12·D2_prev)
    D1 = clip(D1, u1_min, u1_max)  # [-2.5, 2.0] m/s²
    
    # Player 2 best response (given D1)  
    D2 = solve(H22, g2 - H21·D1)
    D2 = clip(D2, u2_min, u2_max)  # [-3.5, 2.5] m/s²
    
    # Check convergence
    if ||D1 - D1_prev|| < 1e-4 and ||D2 - D2_prev|| < 1e-4:
        break

# Extract first control action (receding horizon)
u1_optimal = D1[0]
u2_optimal = D2[0]
```

**Convergence**: Typically converges in 3-8 iterations due to quadratic costs and linear dynamics.

### Implementation in Code

#### 1. Build Prediction Matrices (`_build_prediction_matrices`)
```python
# Called once during initialization
A, B1, C = vehicle.get_state_space_matrices(dt=0.1)  # Use CONTROL dt, not simulation dt!
B2 = B1.copy()  # Symmetric players

for i in range(Np):
    U[i*nz:(i+1)*nz, :] = C @ A^(i+1)
    
    for j in range(min(i+1, Nu)):
        H1[i*nz:(i+1)*nz, j] = C @ A^(i-j) @ B1
        H2[i*nz:(i+1)*nz, j] = C @ A^(i-j) @ B2
```

#### 2. Solve Nash Equilibrium (`solve_nash_equilibrium`)
```python
def solve_nash_equilibrium(x0, R1_ref, R2_ref, lambda_k):
    # Flatten references to (Np*nz,) vectors
    r1 = R1_ref.flatten()
    r2 = R2_ref.flatten()
    
    # Predict free response (no control)
    z_free = (U @ x0).flatten()
    
    # Tracking errors
    e1 = r1 - z_free
    e2 = r2 - z_free
    
    # Build cost matrices (KEY: Scale Q2 by λ)
    Q1_bar = kron(I_Np, Q_output)
    Q2_bar = kron(I_Np, Q_output * lambda_k)  # ← Authority ratio effect!
    
    R1_bar = R1 * I_Nu
    R2_bar = R2 * I_Nu
    
    # Nash coupling matrices
    H11 = H1ᵀ @ Q1_bar @ H1 + R1_bar
    H12 = H1ᵀ @ Q1_bar @ H2
    H21 = H2ᵀ @ Q2_bar @ H1
    H22 = H2ᵀ @ Q2_bar @ H2 + R2_bar
    
    g1 = H1ᵀ @ Q1_bar @ e1
    g2 = H2ᵀ @ Q2_bar @ e2
    
    # Regularization for numerical stability
    H11 += 1e-4 * I
    H22 += 1e-4 * I
    
    # Iterative best response (see algorithm above)
    # ... (15 iterations with clipping)
    
    return u1_optimal, u2_optimal
```

#### 3. Apply Shared Control
```python
# In main_with_nash.py
u1_opt, u2_opt = nash_solver.solve_nash_equilibrium(state, R1_ref, R2_ref, lambda_k)

# Blend according to authority
alpha = lambda_k / (1.0 + lambda_k)
u_shared = alpha * u1_opt + (1 - alpha) * u2_opt

# Apply to vehicle
vehicle.nash_acceleration = u_shared
```

### Key Design Choices (Critical Understanding)

#### 1. Why scale Q2 by λ instead of control inputs?

**What we DO (correct, from Li et al.):**
```python
Q2_bar = λ · Q_output  # Scale the tracking weight
J2 = ||R2 - Z||²_Q2 + ||D2||²_R2
```

**What we DON'T do (would be wrong):**
```python
# ❌ WRONG: Directly weight the equilibrium
u_shared = λ·u1 + (1-λ)·u2  # This breaks Nash structure!
```

**Why this matters:**
- **Q2 scaling changes the optimization problem itself**: Human starts to "want" what system wants
- **Direct weighting would bypass the game**: No equilibrium, just linear combination
- **Game-theoretic property preserved**: Nash equilibrium still exists and is unique
- **Physical interpretation**: λ modulates how much human "cares" about system goals vs. own goals

**Mathematical insight:**
```
When λ → ∞:
  Q2 → ∞ · Q_output
  → Tracking error R2 - Z becomes infinitely important
  → Human must minimize ||R2 - Z|| above all else
  → If R2 ≈ R1, then human and system converge to same control

When λ → 0:
  Q2 → 0
  → Human doesn't care about tracking, only control effort
  → J2 ≈ ||D2||²_R2 → D2 → 0 (do nothing)
```

**After Nash equilibrium, THEN we blend:**
```python
# Get optimal controls from Nash
u1*, u2* = solve_nash_equilibrium(...)

# NOW apply authority weighting
alpha = λ / (1 + λ)
u_shared = alpha · u1* + (1 - alpha) · u2*
```

This is a **two-stage process**: (1) Game theory finds equilibrium, (2) Authority allocation blends.

#### 2. Why iterative best response instead of direct solution?

**Iterative approach (what we use):**
```python
for iteration in range(15):
    D1 = solve(H11, g1 - H12·D2_prev)  # System's best response
    D2 = solve(H22, g2 - H21·D1)       # Human's best response
    # Clip to constraints
    # Check convergence
```

**Direct approach (alternative):**
```python
# Solve coupled system simultaneously
[H11  H12] @ [D1]  =  [g1]
[H21  H22]   [D2]     [g2]
```

**Why iterative is better:**
1. **Numerical stability**: Each step solves smaller, better-conditioned system
2. **Constraint handling**: Can clip each player's control independently
3. **Convergence monitoring**: Can track and diagnose convergence issues
4. **Physical interpretation**: Mimics real alternating decision-making
5. **Regularization**: Can add different regularization per player

**Convergence guarantee**: For quadratic costs + linear dynamics, contraction mapping theorem ensures convergence (typically 3-8 iterations).

#### 3. Why receding horizon (only use first control)?

**What we do:**
```python
# Plan for Np steps (e.g., 20 × 0.1s = 2 seconds)
D1 = [u1[0], u1[1], ..., u1[Nu-1]]  # Entire sequence

# But only execute first action
u1_optimal = D1[0]  

# Next iteration: Replan from new state
```

**Why not use entire sequence?**
1. **Model uncertainty**: Predictions drift from reality over time
2. **Disturbances**: Unexpected events require replanning  
3. **Computational efficiency**: Don't need perfect long-term plan
4. **Closed-loop feedback**: Incorporates new measurements each step
5. **Standard MPC practice**: Proven effective in practice

**Trade-off:**
- Longer Np: Better anticipation of future, but more computation
- Shorter Np: Faster computation, but myopic behavior
- Current choice (Np=20, Nu=10): 2-second lookahead, 1-second control authority

### Debugging Nash Solver (Detailed Analysis)

The solver prints detailed diagnostics at each call. Here's how to interpret them:

#### Normal Output Example:
```
🔍 DEBUG Nash Solver:
   x0 (current) = [150.5  33.33]  # position=150.5m, velocity=33.33m/s (120km/h)
   r1[0:4] (system) = [165.0  33.33  180.0  33.33]  # System wants: +14.5m, same speed
   r2[0:4] (human) = [160.0  35.0  175.0  35.0]     # Human wants: +9.5m, speed up
   e1[0:4] = [14.5  0.0  29.5  0.0]    # System tracking errors
   e2[0:4] = [9.5  1.67  24.5  1.67]   # Human tracking errors
   λ=1.500 → Q2_scale=1.500  # Moderate risk → Q2 weighted 1.5×
   🔧 Standard Nash: ||H11||=1234, ||H22||=1567
   Grads: g1=45.2, g2=38.7
   Cond: H11=2.34e+03, H22=3.12e+03  # Well-conditioned
   Iter 0: d1=[1.85 1.65 1.45], d2=[0.65 0.55 0.45]
   ✅ Converged in 5 iterations
   🔍 Check: v=120.0km/h
      Sys Δv=+0.0 → u1=+1.85  # System doesn't want velocity change, but gives positive accel (gap closing)
      Hum Δv=+1.67 → u2=+0.65 # Human wants to speed up → positive accel ✓ Consistent
   ✅ Nash: u1=+1.85, u2=+0.65
```

#### Red Flags and Solutions:

**🚩 RED FLAG 1: High Condition Numbers**
```
Cond: H11=5.67e+08, H22=3.21e+09  # ← BAD! Singular/near-singular
```

**What it means:**
- Matrices H11, H22 are nearly singular (not invertible)
- Small numerical errors → huge errors in solution
- `solve()` may fail or give nonsense results

**Common causes:**
1. **Zero or tiny weights**: `R1 = 0` or `Q_output = [0, 0]`
2. **Inconsistent horizons**: Nu > Np (control beyond prediction)
3. **Bad state-space matrices**: A, B1, B2 not properly computed

**Solutions:**
```python
# Increase regularization
reg_factor = 1e-3  # Instead of 1e-4

# Check weights are reasonable
assert R1 > 0 and R2 > 0
assert np.all(np.diag(Q_output) > 0)

# Verify state-space matrices
A, B1, C = vehicle.get_state_space_matrices(dt=0.1)
print(f"A eigenvalues: {np.linalg.eigvals(A)}")  # Should be < 1 for stability
```

---

**🚩 RED FLAG 2: No Convergence in 15 Iterations**
```
Iter 0: d1=[1.5 ...], d2=[0.5 ...]
Iter 5: d1=[2.1 ...], d2=[-0.8 ...]
Iter 10: d1=[1.8 ...], d2=[0.2 ...]
Iter 14: d1=[1.9 ...], d2=[0.1 ...]
⚠️ Did not converge!
```

**What it means:**
- Iterative best response is oscillating or diverging
- No stable Nash equilibrium found (shouldn't happen with quadratic costs!)

**Common causes:**
1. **Infeasible references**: R1, R2 physically impossible to track
   ```python
   # Example: Want to be at x=200m in 0.1s from x=0m
   # → Requires infinite acceleration!
   ```
2. **Conflicting objectives**: System and human want opposite things with huge weights
3. **Numerical instability**: Condition numbers too high

**Solutions:**
```python
# Check reference feasibility
max_position_change = max_velocity * dt * Np
assert abs(R1_ref[0] - x0[0]) < max_position_change

# Reduce weight conflicts
if lambda_k > 10:  # Very high authority
    lambda_k = 10  # Cap it

# Increase iteration limit or tolerance
for iteration in range(30):  # Instead of 15
    # ...
    if conv1 < 1e-3 and conv2 < 1e-3:  # Looser tolerance
        break
```

---

**🚩 RED FLAG 3: Inconsistent Control Signs**
```
🔍 Check: v=80.0km/h
   ⚠️ System wants to speed up but gives negative acceleration!
   Sys Δv=+5.0 → u1=-1.2  # ← Inconsistent!
```

**What it means:**
- Nash solver output contradicts the tracking goal
- Logic error in reference generation or solver

**Common causes:**
1. **Wrong reference format**: Position/velocity swapped
   ```python
   # Wrong: R1_ref = [velocity, position, velocity, position, ...]
   # Right: R1_ref = [position, velocity, position, velocity, ...]
   ```

2. **Free response not subtracted**: Should be `e = R - z_free`, not `e = R - z`

3. **Sign error in dynamics**: B matrices have wrong sign
   ```python
   # Check: Positive acceleration → increasing velocity
   A, B1, C = vehicle.get_state_space_matrices(dt)
   x_test = [0, 10]  # 10 m/s
   x_next = A @ x_test + B1 * 1.0  # +1 m/s² accel
   assert x_next[1] > x_test[1]  # Velocity should increase
   ```

**Solutions:**
```python
# Add sanity check in solver
if abs(e1[1]) > 0.1:  # Want velocity change > 0.1 m/s
    sign_desired = np.sign(e1[1])
    sign_actual = np.sign(u1_optimal)
    if sign_desired != sign_actual:
        print(f"⚠️ Warning: Control sign mismatch!")
        # Optional: flip sign or use fallback

# Verify reference generation
print(f"Current velocity: {x0[1]*3.6:.1f} km/h")
print(f"Desired velocity (step 1): {R1_ref[1]*3.6:.1f} km/h")
print(f"Direction: {'speed up' if R1_ref[1] > x0[1] else 'slow down'}")
```

---

**🚩 RED FLAG 4: Extreme Control Values**
```
✅ Nash: u1=+25.0, u2=-18.0  # ← Unrealistic accelerations!
```

**What it means:**
- Controls exceed physical limits (even after clipping)
- Likely missing or wrong constraint limits

**Common causes:**
1. **Wrong clip limits**: 
   ```python
   # Longitudinal should be:
   d1_seq = np.clip(d1_seq, -2.5, 2.0)   # m/s²
   # NOT lateral limits:
   d1_seq = np.clip(d1_seq, -0.3, 0.3)   # rad/s (steering)
   ```

2. **Cost weights too small**: Control effort R1, R2 → 0
   ```python
   # If R1 = 0.001, system doesn't care about effort
   # → Will use huge accelerations to track perfectly
   ```

**Solutions:**
```python
# Set appropriate limits per application
if application == "longitudinal":
    self.u1_min, self.u1_max = -2.5, 2.0  # m/s²
    self.u2_min, self.u2_max = -3.5, 2.5
elif application == "lateral":
    self.u1_min, self.u1_max = -0.3, 0.3  # rad/s
    self.u2_min, self.u2_max = -0.3, 0.3

# Ensure control cost is significant
assert self.R1 >= 1.0 and self.R2 >= 1.0
```

---

**🚩 RED FLAG 5: LinAlgError Exception**
```
❌ Nash solver: Singular matrix, using fallback
```

**What it means:**
- `scipy.linalg.solve()` failed due to singular matrix
- Falling back to zero controls (safe but suboptimal)

**Common causes:**
- All the causes from RED FLAG 1 (condition numbers)
- Plus: Actual singularity in prediction matrices H1, H2

**Solutions:**
```python
# Check prediction matrices are not zero
assert np.linalg.norm(self.H1) > 1e-6
assert np.linalg.norm(self.H2) > 1e-6

# Verify state-space controllability
from scipy.linalg import ctrb
Wc = ctrb(A, B1.reshape(-1, 1))
rank = np.linalg.matrix_rank(Wc)
print(f"Controllability rank: {rank}/{A.shape[0]}")
# Should be full rank for complete controllability

# Last resort: Use pseudoinverse
try:
    d1_seq = solve(H11, g1 - H12 @ d2_prev)
except np.linalg.LinAlgError:
    d1_seq = np.linalg.lstsq(H11, g1 - H12 @ d2_prev, rcond=None)[0]
```

---

#### Good Output Indicators (Green Flags ✅)

1. **Condition numbers < 1e5**: Well-conditioned problem
2. **Converges in < 10 iterations**: Efficient solution
3. **Consistent control signs**: Logic matches physics
4. **Reasonable control magnitudes**: Within ±3 m/s² for longitudinal
5. **Smooth convergence**: `||D1 - D1_prev||` decreases monotonically

## Running Simulations

### Main Entry Points
```powershell
# Longitudinal platoon merging with Nash
cd Longitudinal
python main_with_nash.py
# Choose scenario: 1 (join before), 2 (join middle), 3 (join after), 4 (all), 5 (compare without Nash)

# Original without Nash (comparison)
python main_without_nash.py

# Lateral convoy merging
cd ..\lateral
python "Main Application.py"
```

### Scenario Configuration
Scenarios defined in `run_scenario_with_nash()`:
```python
'Scenario 1: Join Before Platoon (Nash)': {
    'initial_x': 100.0, 'initial_y': -2.0, 
    'target_speed': 100.0, 'join_trigger_time': 25.0
}
'Scenario 2: Join Middle of Platoon (Nash)': {
    'initial_x': -10.0, 'initial_y': -2.0,
    'target_speed': 70.0, 'join_trigger_time': 20.0
}
'Scenario 3: Join After Platoon (Nash)': {
    'initial_x': -100.0, 'initial_y': -2.0,
    'target_speed': 70.0, 'join_trigger_time': 15.0
}
```

## Critical Implementation Details

### Nash Acceleration Application Flow
```python
# 1. In main_with_nash.py - nash_control_step() calculates shared control
nash_result = self.nash_control_step(self.human_vehicle)
shared_accel = nash_result['shared_input']

# 2. Store on vehicle for platoon manager to use
self.human_vehicle.nash_acceleration = shared_accel
self.human_vehicle.a_desired = shared_accel

# 3. In platoon_control.py - PlatoonManager.update() applies it
if not is_prediction_mode and not follower.nash_acceleration is None:
    a_des = follower.nash_acceleration
    follower.nash_acceleration = None  # MUST reset
```

### Multi-Rate Control
Human driver and system reference generators use **substeps** to match finer simulation dt:
```python
dt_sim = 0.02  # Simulation time step
substeps_per_control = max(1, int(np.round(dt / dt_sim)))  # 0.1s / 0.02s = 5 substeps
```
This ensures smooth predictions despite coarser Nash control updates.

### Authority Ratio Interpretation
```python
lambda_k = authority_allocator.compute_authority_ratio(field_force)
alpha = lambda_k / (1.0 + lambda_k)  # System authority
# alpha > 0.5: System dominant (SYSTEM TAKES CONTROL)
# alpha < 0.5: Human dominant
```
**Range**: λ ∈ [0, ∞), α ∈ [0, 1]

### Debugging Output
Simulation prints detailed Nash data every second (when `int(self.time) != int(self.time - self.dt)`):
```
🎯 Nash: u_sh=X.XX m/s², u_sys=X.XX, u_human=X.XX
   📍 Gap_L=XX.Xm, Gap_F=XX.Xm, v=XX.Xkm/h
   🛡️ Risk_L=XX.XN, Risk_F=XX.XN, Total=XX.XN
```

And every 5 seconds:
```
🧠 Nash Data: λ=X.XX, Shared=X.XX
   🛡️ Risks: Leader=XX.XN, Follower=XX.XN, Total=XX.XN
   🤝 Coop=XX, Opp=XX
```

## Visualization

### Outputs
Results saved to `simulation/platoon_sim_kinematic_results/`:
- **Static plots**: Comprehensive analysis via `create_comprehensive_plots()` (positions, velocities, gaps, Nash data)
- **Animations**: GIF files via `create_platoon_animation()` showing merging maneuver
- **Scenario summary**: Text report via `create_detailed_scenario_summary()` with execution time and statistics

### Animation Control
Set `HEADLESS_MODE` in `config.py`:
- `False`: Interactive Qt5Agg/TkAgg backend (recommended)
- `True`: Non-interactive Agg backend (server environments)

Functions use `plt.ioff()` before animation and `plt.ion()` after to control interactivity.

## Common Pitfalls

1. **Nash acceleration reset is critical**:
   ```python
   vehicle.nash_acceleration = None  # MUST do this after reading
   ```
   Otherwise, stale acceleration is reused indefinitely.

2. **Prediction mode flag matters**:
   ```python
   if not is_prediction_mode and not vehicle.nash_acceleration is None:
   ```
   During prediction (e.g., in human model), Nash acceleration is ignored to prevent interference.

3. **State-space model must be enabled** for Nash solver to work properly:
   ```python
   vehicle.use_state_space_model = True  # Required for A/B/C matrices
   ```

4. **Reference trajectory shape**: Must be `(Np * 2, 1)` flattened format:
   ```python
   R1_ref = R1_ref_trajectory.reshape(self.Np * 2, 1)
   ```

5. **Gap calculation sign**: `gap = leader.x - follower.x` (positive = safe distance).

6. **Force sign convention**: Positive = repulsive (push away), Negative = attractive (pull closer).

## Testing & Validation

No formal test suite exists. Validation is done through:
1. **Scenario runs**: Compare Nash vs non-Nash behavior (choice 5 in menu)
2. **Visual inspection**: Check animations for smooth merging
3. **Metrics from `print_nash_analysis()`**:
   - Cooperation vs opposition moments
   - Average/max authority ratio
   - Average/max safety field forces (total, leader, follower)
   - Risk contribution percentages
4. **Console output**: Real-time debug prints every 1-5 seconds

## Dependencies
Core scientific stack (install manually, no requirements.txt):
```python
numpy       # Array operations, linear algebra
scipy       # ODE integration (odeint), Nash solver (solve)
matplotlib  # Visualization and animation
```

## Code Style Notes
- **Emojis in print statements** (🚗, 🧠, 🛡️, 🎯, 📍, 🤝) for visual clarity in console output
- **Extensive inline comments** explaining physics/control theory concepts
- **Hebrew characters** in some file paths (OneDrive directory names) and debug output
- **Mixed naming conventions**: camelCase (VehicleModel) and snake_case (platoon_manager) - no strict standard
- **Commented-out code**: Alternative implementations kept for reference (e.g., kinematic vs state-space models)
- **Progress indicators**: Prints every 2000 iterations with percentage complete

## Safety Field Parameters Reference
From `LongitudinalSafetyFieldParams` in `longitudinal_safety_field.py`:
```python
# TTC (Time-To-Collision) - exponential
ttc_critical = 2.0      # [s] Critical threshold
ttc_weight = 60.0       # Force weight
ttc_decay_rate = 2.0    # Exponential decay

# Headway - exponential  
headway_critical = 1.5  # [s] Critical threshold
headway_weight = 40.0   
headway_decay_rate = 1.5

# Gap Error - BIDIRECTIONAL
gap_error_weight = 100.0
gap_error_threshold = 2.0  # [m] Dead zone ±2m
gap_error_decay_rate = 1.5

# Relative Velocity - safety only
rel_vel_weight = 40.0
rel_vel_threshold = 2.0  # [m/s] Minimum to consider
rel_vel_decay_rate = 1.0

# Velocity Error - BIDIRECTIONAL
velocity_error_weight = 50.0
velocity_error_threshold = 1.0  # [m/s] Dead zone ±1 m/s
velocity_error_decay_rate = 1.2

# Force limits
max_force = 1000.0              # [N] Max repulsive
max_attractive_force = 300.0    # [N] Max attractive
follower_weight = 0.5           # Weight multiplier for follower risk
```

## Quick Development Commands

```powershell
# Run single scenario
cd Longitudinal
python main_with_nash.py  # Choose 1, 2, or 3

# Run all scenarios with Nash
python main_with_nash.py  # Choose 4

# Compare with/without Nash
python main_with_nash.py  # Choose 5

# Clean results directory
Remove-Item .\simulation\platoon_sim_kinematic_results\* -Recurse -Force

# Check for errors in key files
python -m py_compile main_with_nash.py
python -m py_compile control\platoon_control.py
python -m py_compile nash_solver\longitudinal_nash_solver.py
```
