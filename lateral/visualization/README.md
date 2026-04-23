# Visualization Module — Lateral Plots and Animations

## Overview

The lateral visualization module produces plots and animations specialized for the lane-change merging scenario. The key focus is on lateral position, safety field forces, authority ratio, and lane change quality.

---

## Static Plots (`plots.py`)

### Lateral-Specific Subplots

In addition to the longitudinal subplots (velocity, acceleration, etc.), the lateral module adds:

| Subplot | Content |
|---|---|
| Lateral positions (y) | All vehicles' y-position vs. time, with lane center lines at y=0 and y=1.75 |
| Lateral velocity (vy) | Lateral velocity during lane change |
| 2D bird's-eye trajectory | Full (x, y) path showing lane change arc |
| Safety field force | F_x and F_y components vs. time |
| Authority ratio | λ(t) and α(t) = λ/(1+λ) |
| System vs. human inputs | u1(t), u2(t), u_shared(t) — lateral accelerations |
| MOBIL decision timeline | When lane change was triggered |

---

## Animation (`animation.py`)

### 4-Panel Layout

```
┌──────────────────────────────────────┐
│  Bird's-Eye View (x-y with lanes)    │
├──────────────────────────────────────┤
│  Lateral Position y(t)               │
├──────────────────────────────────────┤
│  Authority Ratio λ(t)                │
├──────────────────────────────────────┤
│  Status: time | phase | driver type  │
└──────────────────────────────────────┘
```

### Bird's-Eye View Details

- Two lanes shown: right lane (y=1.75m) and left lane (y=0.0m)
- Lane boundaries and centerlines drawn
- Vehicles color-coded: Red = human, Blue = convoy
- Lane change trajectory arc shown as dotted line
- Safety field ellipse drawn around human vehicle (visualizes risk zone)

---

## Comparison Plots

The lateral module supports **with/without Nash comparison**:
- Side-by-side lane change trajectory
- Lateral error over time (Nash vs. no-Nash)
- Comfort metrics comparison (lateral jerk)

---

## Output Directory

`lateral_sim_results_v2/`

```
lateral_sim_results_v2/
    lateral_analysis_<timestamp>.png
    nash_comparison_<timestamp>.png
    animation_<timestamp>.gif
    metrics_report_<timestamp>.txt
```

---

## Backend

Same as longitudinal: TkAgg on Windows, Agg fallback. Controlled by `HEADLESS_MODE` in `lateral/config.py`.

> Note: On Windows, lateral config tries `TkAgg` first (not Qt5Agg like longitudinal).
