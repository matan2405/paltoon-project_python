# Visualization Module — Plots and Animations

## Overview

This module produces all visual outputs from the platoon simulation: comprehensive static analysis plots and animated GIF visualizations of the merging maneuver.

| File | Role |
|---|---|
| `plots.py` | Static multi-panel analysis plots (PNG) |
| `animation.py` | Time-animated bird's-eye + time-history visualization (GIF) |

---

## Static Plots (`plots.py`)

### `create_comprehensive_plots(sim_data, results_dir)`

Generates a 9-subplot figure covering the full simulation:

| Subplot | Content |
|---|---|
| 1 | Vehicle positions vs. time |
| 2 | Vehicle velocities vs. time (with target speed line) |
| 3 | Inter-vehicle gaps — actual vs. desired |
| 4 | Gap tracking errors (actual − desired) |
| 5 | 2D trajectory (top view with lane markings) |
| 6 | String stability: velocity differences between consecutive vehicles |
| 7 | Vehicle accelerations vs. time |
| 8 | Human vehicle lateral position (Y) — shows lane change |
| 9 | Platoon size vs. time (shows merge event) |

### `create_nash_analysis_plots(sim_data, results_dir)`

Game-theoretic metrics:
- Authority ratio λ(t) and α(t) = λ/(1+λ)
- System vs. human control inputs u1(t) and u2(t)
- Shared control u_shared(t)
- Cooperation vs. opposition moments
- Safety field force magnitude vs. time

### `create_hierarchical_control_plots(sim_data, results_dir)`

Powertrain analysis (only meaningful when `motion_model = 'hierarchical'`):
- Throttle position [0–1]
- Brake pressure [0–1]
- Engine RPM
- Gear number
- Actual vs. desired acceleration

### `create_detailed_scenario_summary(sim_data, results_dir)`

Formatted text report including:
- Scenario name and parameters
- Total simulation time
- Merge success indicator and merge time
- Min/max/mean gaps during following phase
- Min/max/mean authority ratio
- Number of Nash cooperation vs. opposition moments
- Final platoon velocity and spacing errors

---

## Animation (`animation.py`)

### `create_platoon_animation(sim_data, results_dir)`

Creates an animated visualization with **4 panels**:

```
┌─────────────────────────────────────┐
│   Bird's-Eye View (top-down road)   │
├─────────────────────────────────────┤
│   Velocity Trace (time-history)     │
├─────────────────────────────────────┤
│   Gap History (all vehicle pairs)   │
├─────────────────────────────────────┤
│   Status: time | platoon size | ... │
└─────────────────────────────────────┘
```

#### Panel 1 — Spatial View
- Road with lane markings
- Color-coded vehicle rectangles (Red = Human, Blue = Platoon)
- Speed label above each vehicle [km/h]
- Gap distance annotated between consecutive vehicles
- Vertical marker at merge trigger time (on time panels)

#### Panel 2 — Velocity Trace
- Time-history of all vehicle velocities
- Target speed as horizontal dashed line
- Current frame marked with vertical line

#### Panel 3 — Gap History
- Inter-vehicle gaps vs. time
- Desired gap as dashed reference line
- Color-coded per vehicle pair

#### Panel 4 — Status Bar
- Current simulation time
- Platoon size
- Human vehicle status (JOINING / IN PLATOON)

### Animation Settings

| Parameter | Value |
|---|---|
| Total frames | 150 (subsampled) |
| Frame interval | 50 ms (20 fps) |
| Output format | GIF |

---

## Backend Configuration

Controlled by `HEADLESS_MODE` in `config.py`:

| Mode | Backend | Behavior |
|---|---|---|
| `HEADLESS_MODE = False` | Qt5Agg or TkAgg | Interactive display + file save |
| `HEADLESS_MODE = True` | Agg | File save only (no window) |

The backend is selected automatically at startup in `config.py` by trying Qt5Agg → TkAgg → Agg in order.

---

## Output Files

All files saved to `RESULTS_DIR` (= `platoon_sim_kinematic_results/`):

```
platoon_sim_kinematic_results/
    comprehensive_plots_<timestamp>.png
    nash_analysis_<timestamp>.png
    hierarchical_control_<timestamp>.png
    scenario_summary_<timestamp>.txt
    animation_<timestamp>.gif
```
