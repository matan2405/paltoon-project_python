# Platoon Merging Simulation with Nash Equilibrium Control

## Project Overview

A research codebase for simulating vehicle platoon merging with **Nash equilibrium-based shared control** between autonomous systems and human drivers. The project implements **bidirectional safety fields** and **game-theoretic authority allocation** for safe and comfortable merging maneuvers.

### Two Main Implementations

1. **Longitudinal/** - Platoon merging in longitudinal direction (primary focus)
2. **lateral/** - Convoy merging in lateral direction (lane changing)

Both implementations share the same Nash equilibrium framework but differ in dynamics and safety field implementations.

## Architecture

### Core Control Loop (Nash Equilibrium)

The system follows this cycle for every simulation step:

```
1. Safety Field Assessment → 2. Authority Allocation → 3. Trajectory Planning → 4. Nash Solving → 5. Shared Control Application
```

**Key insight**: The Nash solver computes equilibrium between two players (autonomous controller & human driver), weighted by a dynamic authority ratio (λ) that increases with risk.

### Core Components

#### Vehicle Dynamics Model
The codebase supports **three motion models**:
1. **Complex dynamics** - Full powertrain simulation
2. **Kinematic bicycle model** - Simplified geometric model
3. **State-space bicycle model** (preferred) - Linear A/B/C matrices for Nash solver

**Important**: Nash solver requires state-space model for prediction matrices. Set `use_state_space=True` when initializing.

#### Safety Field (Bidirectional)
Located in `nash_solver/longitudinal_safety_field.py`:
- **Five components**: TTC, Headway, Gap Error, Relative Velocity, Velocity Error
- **Bidirectional forces**: Gap Error and Velocity Error can be attractive (negative) or repulsive (positive)
- **Sign convention**: Positive error (deficiency) → Negative force (attractive), Negative error (excess) → Positive force (repulsive)

#### DMPC-Based Nash Equilibrium Solution

The model uses **Distributed Model Predictive Control (DMPC)** to find the equilibrium between two players. This is based on the game-theoretic approach from **Li et al. (2019)** "Shared control with a novel dynamic authority allocation strategy based on game theory and driving safety field".

**Why "Non-Cooperative" Game?**

The game is **non-cooperative** in the game-theoretic sense, even though the players exhibit cooperative behavior:

- Each player optimizes **their own cost function independently**
- No binding agreement or shared cost function
- Nash equilibrium: Neither player can improve their cost by unilateral change

**Key Insight from Li et al.:**
```
"Cooperative behavior emerges from non-cooperative game through authority allocation"

- When risk is LOW (λ ≈ 0): Q2 ≈ 0 → Human follows own preferences
- When risk is HIGH (λ → ∞): Q2 → ∞ → Human's cost aligns with System's
- Result: Smooth transition from human autonomy to system control
```

#### Cost Functions (Quadratic)
```python
# Player 1 (System) - wants to track system reference R1
J1(D1, D2) = ||R1 - Z||²_Q1 + ||D1||²_R1

# Player 2 (Human) - wants to track human reference R2  
J2(D1, D2) = ||R2 - Z||²_Q2 + ||D2||²_R2

where:
  Q2 = λ·Q1  # Scaled by authority ratio!
```

**Critical insight**: Q2 is scaled by λ (authority ratio), not the control inputs.

## Installation and Running

### Prerequisites
```bash
pip install numpy scipy matplotlib
```

### Running the Project

**Longitudinal Simulation:**
```powershell
cd Longitudinal
python main_with_nash.py
```

**Lateral Simulation:**
```powershell
cd lateral
python "Main Application.py"
```

### Scenario Configuration

Scenarios defined in `run_scenario_with_nash()`:

- **Scenario 1: Join Before Platoon**
  - Initial position: x=100m, y=-2m
  - Target speed: 100 km/h
  - Trigger time: 25s

- **Scenario 2: Join Middle of Platoon**
  - Initial position: x=-10m, y=-2m
  - Target speed: 70 km/h
  - Trigger time: 20s

- **Scenario 3: Join After Platoon**
  - Initial position: x=-100m, y=-2m
  - Target speed: 70 km/h
  - Trigger time: 15s

## Conventions and Parameters

### Coordinate System
- **X-axis**: Longitudinal (forward direction, increasing)
- **Y-axis**: Lateral (lane position)
- **Longitudinal**: Y=0 for all platoon vehicles (single lane)
- **Lateral**: Right lane (y=3.5m), Left lane (y=0.0m)

### State Vector Format
```python
# For Nash solver (longitudinal)
state = [position_x, velocity_x]  # Shape: (2,)

# For lateral Nash solver
state = [x, vx, y, vy]  # Shape: (4,)
```

### Time Steps
- **Simulation dt**: 0.02s (50 Hz)
- **Control dt (Nash)**: 0.1s (10 Hz) - multi-rate control
- **Prediction horizon (Np)**: 20 steps (2 seconds ahead)
- **Control horizon (Nu)**: 10 steps (1 second ahead)

### Nash Solver Parameters
```python
Q_output = np.diag([50.0, 200.0])  # Position/velocity tracking weights
R1 = 50.0   # Controller effort cost (smoother = higher)
R2 = 60.0   # Human effort cost (prefer less aggressive)
```

**Tuning tip**: Increase R1/R2 for smoother control, increase Q for tighter tracking.

## Critical Implementation Details

### Nash Acceleration Application Flow

```python
# 1. In main_with_nash.py - nash_control_step() calculates shared control
nash_result = self.nash_control_step(self.human_vehicle)
shared_accel = nash_result['shared_input']

# 2. Store on vehicle for platoon manager to use
self.human_vehicle.nash_acceleration = shared_accel

# 3. In platoon_control.py - PlatoonManager.update() applies it
if not is_prediction_mode and not follower.nash_acceleration is None:
    a_des = follower.nash_acceleration
    follower.nash_acceleration = None  # MUST reset!
```

### Authority Ratio Interpretation
```python
lambda_k = authority_allocator.compute_authority_ratio(field_force)
alpha = lambda_k / (1.0 + lambda_k)  # System authority
# alpha > 0.5: System dominant (SYSTEM TAKES CONTROL)
# alpha < 0.5: Human dominant
```

**Range**: λ ∈ [0, ∞), α ∈ [0, 1]

## Visualization

### Outputs
Results saved to `simulation/platoon_sim_kinematic_results/`:
- **Static plots**: Comprehensive analysis via `create_comprehensive_plots()`
- **Animations**: GIF files via `create_platoon_animation()` showing merging maneuver
- **Scenario summary**: Text report via `create_detailed_scenario_summary()`

### Animation Control
Set `HEADLESS_MODE` in `config.py`:
- `False`: Interactive Qt5Agg/TkAgg backend (recommended)
- `True`: Non-interactive Agg backend (server environments)

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
   During prediction, Nash acceleration is ignored to prevent interference.

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

## Testing and Validation

No formal test suite exists. Validation is done through:
1. **Scenario runs**: Compare Nash vs non-Nash behavior
2. **Visual inspection**: Check animations for smooth merging
3. **Metrics from `print_nash_analysis()`**: Cooperation vs opposition moments, average/max authority ratio
4. **Console output**: Real-time debug prints every 1-5 seconds

## Dependencies

Core scientific stack:
```bash
numpy       # Array operations, linear algebra
scipy       # ODE integration, Nash solver
matplotlib  # Visualization and animation
```

## Code Style Notes

- **Emojis in print statements** (🚗, 🧠, 🛡️, 🎯, 📍, 🤝) for visual clarity in console output
- **Extensive inline comments** explaining physics/control theory concepts
- **Hebrew characters** in some file paths (OneDrive directory names) and debug output
- **Mixed naming conventions**: camelCase (VehicleModel) and snake_case (platoon_manager)

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

## Safety Field Parameters Reference

From `LongitudinalSafetyFieldParams`:
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

# Relative Velocity - safety only
rel_vel_weight = 40.0
rel_vel_threshold = 2.0  # [m/s] Minimum to consider

# Velocity Error - BIDIRECTIONAL
velocity_error_weight = 50.0
velocity_error_threshold = 1.0  # [m/s] Dead zone ±1 m/s

# Force limits
max_force = 1000.0              # [N] Max repulsive
max_attractive_force = 300.0    # [N] Max attractive
follower_weight = 0.5           # Weight multiplier for follower risk
```

## License and Contributions

Research project. For questions and suggestions, please contact the project maintainers.

---

**Note**: This project is intended for research and educational purposes only.
