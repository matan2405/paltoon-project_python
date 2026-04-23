# Unity Vehicle Platoon Simulator

A real-time 3D simulation of a vehicle platoon using Unity, implementing the Rajamani string-stable platoon control algorithm. Includes a Python script for post-simulation data analysis.

## What It Does

- Simulates a leader vehicle + followers maintaining safe inter-vehicle gaps at highway speeds
- Uses the **Rajamani** platoon control law for string-stable gap regulation
- Models **Audi TT Coupé 2.0 TFSI** vehicle dynamics (bicycle model with real engine/transmission)
- Supports **Logitech G29 steering wheel** and keyboard inputs
- Exports telemetry to CSV for analysis in Python

## Structure

```
simulator_unity/
├── Assets/          # Unity project: C# scripts, scenes, prefabs
├── VehicleData/     # Exported CSV data + Python analysis script
├── Packages/        # Unity package dependencies
└── ProjectSettings/ # Unity project configuration
```

## Running

### Unity Simulation

1. Open `simulator_unity/` in Unity Editor
2. Press **Play** to start the simulation
3. Use keyboard or Logitech G29 to interact
4. Exit Play mode to export CSV data to `VehicleData/`

### Python Analysis

After running a simulation:

```bash
pip install numpy matplotlib pandas
cd VehicleData
python bicycle_model_python.py
```

Generates plots for:
- Per-vehicle: steering, longitudinal force, lateral position, yaw, velocities, 2D trajectory
- Platoon-level: actual vs desired gaps, gap errors, velocity differences

## Control Algorithm

Followers use the **Rajamani** model (constant time headway):

```
a_des = -k1·a_leader - k2·a_follower - k3·e_dot - k4·e - k5·v
```

where `h = 1.5 s` (desired time headway) and gains are derived analytically from h and k1, k5. Acceleration is clamped to ±2.5 m/s².

The leader tracks a target velocity using free-road acceleration.

## Vehicle Model

Physical parameters match the Python simulation modules (Audi TT 2.0 TFSI, 1305 kg). The bicycle model runs in Unity's game loop and includes:
- Engine torque curve (real turbo spool behavior)
- 6-speed S-tronic transmission with RPM-based and speed-based shift criteria
- Aerodynamic drag, rolling resistance
- Brake system with weight transfer

## Data Format

| File | Contents |
|------|----------|
| `Inputs_N.csv` | Time, longitudinal force Fx [N], steering angle Lambda [rad] |
| `Outputs_N.csv` | Time, position XYZ, velocity Vx, acceleration Ax, lateral state Y/YDot, yaw Psi/PsiDot, engine RPM |
| `platoon_data.csv` | Time, actual/desired gaps for each pair, target velocity |

Data is exported when Play mode ends (`OnDestroy()` callback).

## Relationship to Python Simulation Modules

This Unity simulator implements the **baseline Rajamani controller only** — it does not include Nash equilibrium shared control. It serves as:
- A 3D visualization and validation environment
- A hardware-in-the-loop testbed with real steering wheel input
- A reference for vehicle parameters and control gains used in the Python modules (`Longitudinal/`, `lateral/`)

The Python modules extend this foundation with game-theoretic shared control (Nash equilibrium) and human driver modeling.
