"""
Post-simulation analysis for the Unity platoon simulator.

Reads from:
  VehicleData/data/ego_vehicle.csv      — ego full 6-DOF state
  VehicleData/data/vehicle_*.csv        — platoon vehicle longitudinal state
  VehicleData/data/platoon_gaps.csv     — inter-vehicle gaps + phase
  VehicleData/data/ego_nash.csv         — Nash coordinator telemetry

Produces 5 figures in VehicleData/figures/:
  1_velocities.png       — longitudinal speeds of all vehicles
  2_gaps.png             — actual vs desired gaps per slot
  3_accelerations.png    — longitudinal accelerations
  4_nash_control.png     — Nash u1/u2/u_shared + authority λ (long + lat)
  5_ego_lateral.png      — ego lateral dynamics (y, psi, vy, delta)

Run from the simulator_unity/VehicleData directory:
  python run_analysis.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR  = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load CSVs ─────────────────────────────────────────────────────────────────

def load(filename, required=True):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing: {path}")
        return None
    return pd.read_csv(path)

ego   = load('ego_vehicle.csv')
gaps  = load('platoon_gaps.csv')
nash  = load('ego_nash.csv', required=False)

# Load all platoon vehicle CSVs (vehicle_Car1.csv, vehicle_Car2.csv, etc.)
platoon_files = sorted(glob.glob(os.path.join(DATA_DIR, 'vehicle_*.csv')))
platoon = {os.path.basename(f).replace('vehicle_', '').replace('.csv', ''): pd.read_csv(f)
           for f in platoon_files}

t_ego  = ego['Time'].values
t_gaps = gaps['Time'].values

# Phase colours for background shading
PHASE_COLORS = {
    'Approach':  '#8888ff',
    'GapSearch': '#ffaa00',
    'Merge':     '#ff4444',
    'Following': '#44cc44',
}

def shade_phases(ax, t, phases):
    if phases is None:
        return
    prev_p, prev_t = phases.iloc[0], t[0]
    for i in range(1, len(t)):
        if phases.iloc[i] != prev_p or i == len(t) - 1:
            ax.axvspan(prev_t, t[i], alpha=0.07,
                       color=PHASE_COLORS.get(prev_p, 'gray'), lw=0)
            prev_p, prev_t = phases.iloc[i], t[i]

phase_col = gaps['phase'] if 'phase' in gaps.columns else None

# ── Fig 1: Velocities ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
shade_phases(ax, t_gaps, phase_col)
for name, df in platoon.items():
    ax.plot(df['Time'], df['vx'] * 3.6, lw=1, label=name)
ax.plot(t_ego, ego['vx'] * 3.6, 'k--', lw=2, label='Ego')
if 'target_vx_kmh' in gaps.columns:
    ax.plot(t_gaps, gaps['target_vx_kmh'], ':', color='gray', lw=1, label='Target')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Speed (km/h)')
ax.set_title('Longitudinal Velocity')
ax.legend(fontsize=8); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '1_velocities.png'), dpi=120)
plt.close()
print('Saved 1_velocities.png')

# ── Fig 2: Gaps ───────────────────────────────────────────────────────────────
gap_actual_cols = [c for c in gaps.columns if c.startswith('actual_gap_')]
gap_des_cols    = [c for c in gaps.columns if c.startswith('des_gap_')]
n_slots = len(gap_actual_cols)

if n_slots > 0:
    fig, axes = plt.subplots(n_slots, 1, figsize=(12, 3 * n_slots), sharex=True)
    if n_slots == 1:
        axes = [axes]
    for i, (ac, dc) in enumerate(zip(gap_actual_cols, gap_des_cols)):
        ax = axes[i]
        shade_phases(ax, t_gaps, phase_col)
        act = gaps[ac].values.astype(float)
        des = gaps[dc].values.astype(float)
        ax.plot(t_gaps, act, 'b-',  lw=1.2, label='Actual')
        ax.plot(t_gaps, des, 'b--', lw=0.8, label='Desired')
        ax.fill_between(t_gaps, act, des, alpha=0.12, color='blue')
        ax.set_ylabel(f'Gap slot {i} (m)')
        ax.legend(fontsize=8); ax.grid(True)
    axes[0].set_title('Inter-Vehicle Gaps (bumper-to-bumper)')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '2_gaps.png'), dpi=120)
    plt.close()
    print('Saved 2_gaps.png')

# ── Fig 3: Accelerations ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
shade_phases(ax, t_gaps, phase_col)
for name, df in platoon.items():
    ax.plot(df['Time'], df['ax'], lw=1, label=name)
ax.plot(t_ego, ego['ax'], 'k--', lw=2, label='Ego')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Acceleration (m/s²)')
ax.set_title('Longitudinal Acceleration')
ax.legend(fontsize=8); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '3_accelerations.png'), dpi=120)
plt.close()
print('Saved 3_accelerations.png')

# ── Fig 4: Nash control ───────────────────────────────────────────────────────
if nash is not None:
    t_n = nash['Time'].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Nash Shared Control — Ego')

    # (a) Longitudinal inputs
    ax = axes[0, 0]
    shade_phases(ax, t_gaps, phase_col)
    ax.plot(t_n, nash['u_long'],  'k-',  lw=1.5, label='u_shared')
    ax.plot(t_n, nash['u1_long'], 'b--', lw=1,   label='u1 (system)')
    ax.plot(t_n, nash['u2_long'], color='orange', ls='--', lw=1, label='u2 (human)')
    ax.set_ylabel('Long ctrl (m/s²)'); ax.set_title('(a) Longitudinal Nash inputs')
    ax.legend(fontsize=7); ax.grid(True)

    # (b) Long authority λ + gap error
    ax = axes[0, 1]
    ax2 = ax.twinx()
    shade_phases(ax, t_gaps, phase_col)
    ax.semilogy(t_n, np.clip(nash['lambda_long'], 1e-4, None), 'r-', label='λ_long')
    ax2.plot(t_n, nash['gap_err'], color='gray', alpha=0.7, label='gap error (m)')
    ax.set_ylabel('λ_long (log)', color='r')
    ax2.set_ylabel('Gap error (m)', color='gray')
    ax.set_title('(b) Long authority & gap error'); ax.grid(True)

    # (c) Lateral inputs
    ax = axes[1, 0]
    shade_phases(ax, t_gaps, phase_col)
    ax.plot(t_n, np.degrees(nash['delta_lat']), 'k-',  lw=1.5, label='δ_shared')
    ax.plot(t_n, np.degrees(nash['u1_lat']),    'b--', lw=1,   label='u1_lat (system)')
    ax.plot(t_n, np.degrees(nash['u2_lat']),    color='orange', ls='--', lw=1, label='u2_lat (human)')
    ax.set_ylabel('Steering (deg)'); ax.set_title('(c) Lateral Nash inputs')
    ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('Time (s)')

    # (d) Lat authority λ + DSF force
    ax = axes[1, 1]
    ax2 = ax.twinx()
    shade_phases(ax, t_gaps, phase_col)
    ax.semilogy(t_n, np.clip(nash['lambda_lat'], 1e-4, None), color='purple', label='λ_lat')
    ax2.plot(t_n, nash['lat_force'], color='gray', alpha=0.7, label='lat DSF force (N)')
    ax.set_ylabel('λ_lat (log)', color='purple')
    ax2.set_ylabel('Lat DSF force (N)', color='gray')
    ax.set_title('(d) Lat authority & DSF force'); ax.grid(True); ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '4_nash_control.png'), dpi=120)
    plt.close()
    print('Saved 4_nash_control.png')
else:
    print('Skipped 4_nash_control.png (ego_nash.csv not found)')

# ── Fig 5: Ego lateral dynamics ───────────────────────────────────────────────
# Sign convention: Unity world frame — right = positive X, CCW yaw = positive ψ.
# posX, psi, lambda are stored raw from Unity.
# vy/yDot follow Belousov (y = x0 - posX, left = +), so they are negated here
# to match Unity direction (right = positive).
if 'psi' in ego.columns:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Ego Lateral Dynamics  [Unity sign: right = positive]')

    # (a) Lateral position: posX raw from Unity (right = +)
    ax = axes[0, 0]
    shade_phases(ax, t_gaps, phase_col)
    pos_x = ego['posX'].values if 'posX' in ego.columns else np.zeros_like(t_ego)
    ax.plot(t_ego, pos_x, 'b-')
    ax.set_ylabel('posX (m)  [right = +]'); ax.set_title('(a) Lateral position')
    ax.axhline(0, color='k', ls=':', lw=0.8); ax.grid(True)

    # (b) Yaw angle: psi raw from Unity (CCW = +, i.e. left turn = +)
    ax = axes[0, 1]
    shade_phases(ax, t_gaps, phase_col)
    ax.plot(t_ego, np.degrees(ego['psi']), color='purple')
    ax.set_ylabel('ψ (deg)  [CCW/left = +]'); ax.set_title('(b) Yaw angle')
    ax.axhline(0, color='k', ls=':', lw=0.8); ax.grid(True)

    # (c) Lateral velocity:
    #   vy   = body-frame lateral (sNew[3] from bicycle model equations)
    #   yDot = world-frame  = -(vx·sinψ + vy·cosψ)  (differs from vy when ψ ≠ 0)
    # Both stored in Belousov sign (left = +) → negate to Unity (right = +)
    ax = axes[1, 0]
    shade_phases(ax, t_gaps, phase_col)
    if 'vy' in ego.columns:
        ax.plot(t_ego, -ego['vy'],   'g-',  lw=1.5, label='vy  body-frame (slip)')
    if 'yDot' in ego.columns:
        ax.plot(t_ego, -ego['yDot'], 'm--', lw=1.2, label='ẏ  world-frame = −(vx·sinψ + vy·cosψ)')
    ax.set_ylabel('Lateral velocity (m/s)  [right = +]')
    ax.set_title('(c) Lateral velocity')
    ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('Time (s)')

    # (d) Steering: front wheel angle delta [deg] (left axis) and steering norm (right axis)
    # Note: ego_vehicle.csv column 'lambda' = angles.average = Ackermann avg wheel angle [rad]
    ax = axes[1, 1]
    ax2d = ax.twinx()
    shade_phases(ax, t_gaps, phase_col)
    if 'lambda' in ego.columns:
        ax.plot(t_ego, np.degrees(ego['lambda']), 'r-', label='δ front wheel angle (deg)')
    if 'steering' in ego.columns:
        ax2d.plot(t_ego, ego['steering'], 'b--', lw=0.8, label='steering input (norm)')
        ax2d.set_ylabel('steering input (norm)', color='blue')
        ax2d.tick_params(axis='y', labelcolor='blue')
    ax.set_ylabel('δ front wheel angle (deg)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title('(d) Steering input & wheel angle')
    ax.axhline(0, color='k', ls=':', lw=0.8); ax.grid(True); ax.set_xlabel('Time (s)')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2d.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '5_ego_lateral.png'), dpi=120)
    plt.close()
    print('Saved 5_ego_lateral.png')

# ── Summary stats ─────────────────────────────────────────────────────────────
print('\n── Summary ─────────────────────────────────────────────')
print(f'  Ego speed range:  {ego["vx"].min()*3.6:.1f} .. {ego["vx"].max()*3.6:.1f} km/h')
if 'psi' in ego.columns:
    print(f'  Max |yaw|:        {np.degrees(np.abs(ego["psi"])).max():.2f} deg')
if n_slots > 0:
    for i, ac in enumerate(gap_actual_cols):
        g = gaps[ac].dropna()
        if len(g):
            print(f'  Gap slot {i}:       {g.min():.1f} .. {g.max():.1f} m  (mean {g.mean():.1f})')
if nash is not None:
    print(f'  λ_long range:     {nash["lambda_long"].min():.3f} .. {nash["lambda_long"].max():.1f}')
    print(f'  λ_lat range:      {nash["lambda_lat"].min():.3f} .. {nash["lambda_lat"].max():.1f}')
    gmax = nash['gap_err'].abs().max()
    ti   = nash.loc[nash['gap_err'].abs().idxmax(), 'Time']
    print(f'  Max |gap error|:  {gmax:.1f} m at t={ti:.1f}s')
print('────────────────────────────────────────────────────────')
