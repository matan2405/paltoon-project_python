# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What's in This Folder

This folder contains two independent parts:
1. **Unity C# project** (`Assets/`) — real-time 3D vehicle platoon simulation
2. **Python analysis script** (`VehicleData/bicycle_model_python.py`) — post-simulation data analysis

There is **no real-time Python↔Unity communication**. Data flows one-way: Unity exports CSV files on scene exit → Python reads and plots them.

## Running the Python Analysis

**Dependencies:**
```bash
pip install numpy matplotlib pandas
```

**Run analysis after a Unity simulation:**
```bash
cd VehicleData
python bicycle_model_python.py
```

Reads CSV files from `VehicleData/` and generates matplotlib plots. No arguments required — file paths are hardcoded relative to the script.

## Unity Project

Open the project root (`simulator_unity/`) in **Unity Editor**. No build step is needed for development; press Play in the editor to run.

**Required Unity version**: Check `ProjectSettings/ProjectVersion.txt`.

## Architecture

### C# Scripts (`Assets/Scripts/`)

| Script | Role |
|--------|------|
| `AdvancedBicycleModel.cs` | Vehicle physics: kinematic + dynamic bicycle model, engine torque curve, transmission, braking |
| `PlatoonManager.cs` | Rajamani string-stable platoon control (leader + followers) |
| `VehicleDataVisualizer.cs` | Per-vehicle telemetry recorder → `Inputs_N.csv`, `Outputs_N.csv` |
| `PlatoonDataVisualizer.cs` | Platoon-level gap recorder → `platoon_data.csv` |
| `VehicleParameters.cs` | Audi TT Coupé 2.0 TFSI specs (shared with Python modules) |
| `Powertrain.cs` | Engine torque curves, gear ratios |
| `Transmission.cs` | S-tronic shift logic, ratio blending |
| `BrakeSystem.cs` | Brake force with weight transfer |
| `SteeringSystem.cs` | Ackermann steering geometry |
| `Game_Manager.cs` | Dashboard UI (speedometer, RPM) |
| `VehicleInputs.cs` | Keyboard and Logitech G29 steering wheel input |

### Platoon Control Algorithm (Rajamani)

Implemented in `PlatoonManager.cs`. Follower acceleration:
```
a_des = -k1·a_leader - k2·a_follower - k3·e_dot - k4·e - k5·v

where: h = 1.5 s (time headway), k1 = -0.12, k5 = 0.1
       k2 = -k1 - h·k1·k5
       k3 = 1/h - k1·k5
       k4 = k5/h
       |a_des| clamped to 2.5 m/s²
```
Leader uses free-road acceleration (proportional to velocity error from target speed).

### CSV Data Format

**`Inputs_N.csv`** (per vehicle N):
```
Time, Fx, Lambda
```
- `Fx`: longitudinal force [N]
- `Lambda`: steering angle [rad]

**`Outputs_N.csv`** (per vehicle N):
```
Time, PositionX, PositionY, PositionZ, Vx, Ax, Y, YDot, Psi, PsiDot, EngineRpm
```

**`platoon_data.csv`**:
```
Time, actual_gap1, actual_gap2, des_gap1, des_gap2, TargetVelocity
```

Data is written in Unity's `OnDestroy()` callback. To export mid-simulation, call `SaveDataToCSV()` manually.

### Python Analysis Script

`VehicleData/bicycle_model_python.py` provides two functions:
- `plot_simulation_data_Player(N)` — 6-panel dynamics plot for vehicle N (steering, force, lateral position, yaw, velocities, trajectory)
- `plot_simulation_data_Platoon()` — inter-vehicle gap analysis (actual vs desired, gap error, velocity differences)

## Key Differences from Python Simulation Modules

| Aspect | Unity (C#) | Python (Longitudinal / Lateral) |
|--------|-----------|-------------------------------|
| Control | Rajamani only | Rajamani + Nash equilibrium |
| Real-time | Yes (Unity game loop) | No (batch simulation) |
| Visualization | 3D with camera, dashboard | matplotlib 2D plots + GIF |
| Nash/shared control | Not implemented | Core feature |
| Human driver model | Keyboard / G29 wheel | IDM / Stanley |

## Common Pitfalls

- CSV files are only written when Unity's `OnDestroy()` fires — exiting Play mode or closing the application
- `bicycle_model_python.py` expects CSV files in `VehicleData/` with exact column names; column order matters
- Vehicle numbering in CSV filenames (N) starts from the index Unity assigns — check scene setup if files are missing
- Unity coordinate system: Z = forward (not X as in Python modules)
