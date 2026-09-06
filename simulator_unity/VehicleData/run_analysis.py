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
print('\n-- Summary ------------------------------------------------------')
print(f'  Ego speed range:  {ego["vx"].min()*3.6:.1f} .. {ego["vx"].max()*3.6:.1f} km/h')
if 'psi' in ego.columns:
    print(f'  Max |yaw|:        {np.degrees(np.abs(ego["psi"])).max():.2f} deg')
if n_slots > 0:
    for i, ac in enumerate(gap_actual_cols):
        g = gaps[ac].dropna()
        if len(g):
            print(f'  Gap slot {i}:       {g.min():.1f} .. {g.max():.1f} m  (mean {g.mean():.1f})')
if nash is not None:
    print(f'  lam_long range:   {nash["lambda_long"].min():.3f} .. {nash["lambda_long"].max():.1f}')
    print(f'  lam_lat range:    {nash["lambda_lat"].min():.3f} .. {nash["lambda_lat"].max():.1f}')
    gmax = nash['gap_err'].abs().max()
    ti   = nash.loc[nash['gap_err'].abs().idxmax(), 'Time']
    print(f'  Max |gap error|:  {gmax:.1f} m at t={ti:.1f}s')
print('-' * 56)

# ── Regulatory compliance checks — per phase ──────────────────────────────────
# Phase → relevant standards:
#   GapSearch : ISO 15622 s6.4 (long. decel/jerk/accel limits, ACC-like control begins)
#   Merge     : ISO 15622 s6.4 + R79 s5.6.4.4 (lat accel/jerk) + R79 s5.6.4.7 (TTC)
#               + R79 s5.6.4.6.5 (lane-change duration)
#   Following : ISO 15622 s6.4 + ISO 15622 s6.2.3.1 (THW) + ISO 11270 s5.4 (LKAS)
# Approach phase is driven by the human only — excluded from all checks.

def _moving_avg(arr, dt, window_s):
    k = max(1, int(round(window_s / dt)))
    kernel = np.ones(k) / k
    return np.convolve(np.abs(arr), kernel, mode='same')

def _ego_slice(phase_name):
    """Return (t_array, ax_array, vx_array, psi_array) masked to a single phase."""
    if phase_col is None:
        return None, None, None, None
    pmask = phase_col.isin([phase_name]).values
    if not pmask.any():
        return None, None, None, None
    t_ph  = t_gaps[pmask]
    t0, t1 = t_ph[0], t_ph[-1]
    emask = (t_ego >= t0) & (t_ego <= t1)
    t_e   = t_ego[emask]
    ax_e  = ego['ax'].values[emask]
    vx_e  = ego['vx'].values[emask]
    psi_e = ego['psi'].values[emask] if 'psi' in ego.columns else None
    return t_e, ax_e, vx_e, psi_e

def _gaps_slice(phase_name):
    """Return (t_g, gap_arrays_dict, vx_interp) for a single phase."""
    if phase_col is None:
        return None, None, None
    pmask = phase_col.isin([phase_name]).values
    if not pmask.any():
        return None, None, None
    t_g    = t_gaps[pmask]
    vx_g   = np.interp(t_g, t_ego, ego['vx'].values)
    gap_d  = {ac: gaps[ac].values[pmask].astype(float) for ac in gap_actual_cols}
    return t_g, gap_d, vx_g

ok   = 'PASS'
fail = 'FAIL'
na   = 'N/A '
dt_ego = float(np.median(np.diff(t_ego))) if len(t_ego) > 1 else 0.02

def _section(title):
    print(f'\n-- {title} ' + '-' * max(0, 56 - len(title)))

def _long_iso15622(ax_e, t_e=None, nash_phase=None):
    # ISO 15622 s6.4: decel avg 2s, jerk avg 1s, accel peak.
    # Nash outputs at 0.1 s but ego CSV logs at dt_ego (~0.01 s): differentiating the
    # step-change at each Nash tick produces apparent spike jerk of ~(delta_u/0.01) that
    # inflates the 1-s moving average.  When nash CSV is available, compute jerk at the
    # Nash rate (0.1 s) instead — that is the physically meaningful rate of command change.
    avg2s = _moving_avg(-ax_e, dt_ego, 2.0).max()
    s1 = ok if avg2s <= 3.5 else fail
    print(f'  ISO 15622 s6.4  Long. decel  avg 2s <= 3.5 m/s2:  worst = {avg2s:.2f} m/s2  -> {s1}')

    # Jerk and accel: prefer Nash-rate (0.1 s) to avoid step-change logging artifacts.
    # ego_vehicle.csv logs at ~0.01 s; each Nash tick causes a step in ax that, when
    # differentiated at 0.01 s, gives ~100x amplified apparent jerk.
    if nash is not None and t_e is not None and nash_phase is not None and \
       'u_long' in nash.columns and 'Time' in nash.columns:
        t0_ph, t1_ph = t_e[0], t_e[-1]
        nmask = (nash['Time'].values >= t0_ph) & (nash['Time'].values <= t1_ph)
        if nmask.any():
            n_ax_full = nash['u_long'].values[nmask]
            n_t_full  = nash['Time'].values[nmask]
            # ego_nash.csv logs at 100 Hz but Nash updates at ~10 Hz.
            # Detect actual Nash steps: rows where u_long changes value.
            # Use those rows only so dt reflects the true command rate.
            step_idx = np.where(np.abs(np.diff(n_ax_full)) > 1e-6)[0] + 1
            step_idx = np.concatenate([[0], step_idx])   # always include first row
            n_ax  = n_ax_full[step_idx]
            n_t   = n_t_full[step_idx]
            dt_nash = float(np.median(np.diff(n_t))) if len(n_t) > 1 else 0.1
            jerk_n  = np.gradient(n_ax, dt_nash)
            wjerk   = _moving_avg(jerk_n, dt_nash, 1.0).max()
            wacc    = n_ax.max()
            s2 = ok if wjerk <= 2.5 else fail
            s3 = ok if wacc  <= 2.0 else fail
            print(f'  ISO 15622 s6.4  Long. jerk   avg 1s <= 2.5 m/s3:  worst = {wjerk:.2f} m/s3  -> {s2}  (Nash-rate ~{dt_nash:.2f}s, {len(n_t)} steps)')
            print(f'  ISO 15622 s6.4  Long. accel  peak   <= 2.0 m/s2:  worst = {wacc:.2f} m/s2   -> {s3}  (Nash-rate u_long)')
            return
    # Fallback: differentiate high-rate ego CSV (may over-report due to Nash step changes)
    jerk  = np.gradient(ax_e, dt_ego)
    wjerk = _moving_avg(jerk, dt_ego, 1.0).max()
    wacc  = ax_e.max()
    s2 = ok if wjerk <= 2.5 else fail
    s3 = ok if wacc  <= 2.0 else fail
    print(f'  ISO 15622 s6.4  Long. jerk   avg 1s <= 2.5 m/s3:  worst = {wjerk:.2f} m/s3  -> {s2}  [high-rate, may over-report]')
    print(f'  ISO 15622 s6.4  Long. accel  peak   <= 2.0 m/s2:  worst = {wacc:.2f} m/s2   -> {s3}  [high-rate]')

def _lat_r79_iso11270(ax_e, psi_e, vx_e, t_e=None, is_merge=True):
    # Correct lateral acceleration metric: a_lat = vx * psi_dot  (world-frame, felt by occupant).
    # vx^2 * delta / L  is the steady-state cornering formula — it gives the centripetal
    # acceleration only when the vehicle is tracking a constant-radius arc.  During a lane
    # change (transient), vx*psi_dot and vx^2*delta/L diverge by up to 16x (verified on data:
    # delta=0.038 rad at vx=28 m/s gives bm=11.35 m/s^2 but vx*psi_dot=0.70 m/s^2 at that instant).
    # Therefore all lateral accel checks use vx*psi_dot from ego_vehicle.csv.
    #
    # Jerk is computed at Nash-rate (0.1 s) using delta_lat steps from ego_nash.csv to avoid
    # spike amplification from differentiating the high-rate ego CSV.  a_lat at Nash-ticks is
    # interpolated from vx*psi_dot at those timestamps.  Falls back to full-rate if Nash CSV
    # unavailable.
    #
    # is_merge=True:  R79 s5.6.4.4 lane-change framing (full + excl. 1st-s onset).
    # is_merge=False: ISO 11270 s5.4 LKAS steady-state framing (full phase only).

    psi_dot  = np.gradient(psi_e, dt_ego)
    a_lat    = vx_e * psi_dot                      # [m/s^2], signed, high-rate
    wlong    = (-ax_e).max()
    s4 = ok if wlong <= 3.0 else fail

    wlat_full  = np.abs(a_lat).max()
    frac3_full = (np.abs(a_lat) > 3.0).mean() * 100

    # Jerk: prefer Nash-rate u1_lat steps to avoid step-change artifacts.
    # u1_lat (system lateral command) changes only at each Nash solve tick (~0.1 s),
    # unlike delta_lat which is the integrated wheel angle updated every frame.
    jerk_rate_note = '[high-rate, may over-report]'
    lat_jerk = np.gradient(a_lat, dt_ego)
    wjlat_full = _moving_avg(lat_jerk, dt_ego, 0.5).max()
    n_t = None; a_lat_n = None; dt_nash = 0.1   # will be overwritten if Nash available
    if nash is not None and t_e is not None and \
       'u1_lat' in nash.columns and 'Time' in nash.columns:
        t0_ph, t1_ph = t_e[0], t_e[-1]
        nmask = (nash['Time'].values >= t0_ph) & (nash['Time'].values <= t1_ph)
        if nmask.any():
            n_u1_full = nash['u1_lat'].values[nmask]
            n_t_full  = nash['Time'].values[nmask]
            step_idx  = np.where(np.abs(np.diff(n_u1_full)) > 1e-7)[0] + 1
            step_idx  = np.concatenate([[0], step_idx])
            if len(step_idx) > 2:
                n_t     = n_t_full[step_idx]
                dt_nash = float(np.median(np.diff(n_t))) if len(n_t) > 1 else 0.1
                # a_lat at Nash timestamps: interpolate from high-rate vx*psi_dot
                a_lat_n = np.interp(n_t, t_e, a_lat)
                jerk_n  = np.gradient(a_lat_n, dt_nash)
                wjlat_full = _moving_avg(jerk_n, dt_nash, 0.5).max()
                jerk_rate_note = f'Nash-rate ~{dt_nash:.2f}s, {len(n_t)} steps'

    if is_merge:
        # R79 s5.6.4.4: report full phase AND post-onset (excl. first 1 s human turning onset).
        # Post-onset jerk also computed at Nash-rate using the same n_t / a_lat_n arrays if
        # available, to avoid double-counting step-change artifacts.
        wlat_post = wlat_full; wjlat_post = wjlat_full; frac3_post = frac3_full
        if t_e is not None and len(t_e) > 0:
            t_skip = t_e[0] + 1.0
            post_mask = t_e >= t_skip
            if post_mask.sum() > 10:
                a_lat_post  = a_lat[post_mask]
                wlat_post   = np.abs(a_lat_post).max()
                frac3_post  = (np.abs(a_lat_post) > 3.0).mean() * 100
                # Jerk post-onset at Nash-rate if available, else high-rate fallback
                if jerk_rate_note.startswith('Nash-rate'):
                    post_nash_mask = n_t >= t_skip
                    if post_nash_mask.sum() > 2:
                        jerk_post_n = np.gradient(a_lat_n[post_nash_mask], dt_nash)
                        wjlat_post  = _moving_avg(jerk_post_n, dt_nash, 0.5).max()
                    # else keep wjlat_post = wjlat_full
                else:
                    jerk_post   = np.gradient(a_lat_post, dt_ego)
                    wjlat_post  = _moving_avg(jerk_post, dt_ego, 0.5).max()

        s1f = ok if wlat_full <= 1.0 else fail
        s2f = ok if wlat_full <= 3.0 else fail
        s3f = ok if wjlat_full <= 5.0 else fail
        s1p = ok if wlat_post <= 1.0 else fail
        s2p = ok if wlat_post <= 3.0 else fail
        s3p = ok if wjlat_post <= 5.0 else fail

        print(f'  R79  s5.6.4.4(a) Lat accel (total, full phase) <= 1.0 m/s2:  worst = {wlat_full:.2f} m/s2  -> {s1f}  ({frac3_full:.1f}% of phase >3 m/s2)')
        print(f'  R79/ISO11270     Lat accel (total, full phase) <= 3.0 m/s2:  worst = {wlat_full:.2f} m/s2  -> {s2f}')
        print(f'  R79  s5.6.4.4(a) Lat accel (excl. 1st s onset) <= 1.0 m/s2: worst = {wlat_post:.2f} m/s2  -> {s1p}')
        print(f'  R79/ISO11270     Lat accel (excl. 1st s onset) <= 3.0 m/s2:  worst = {wlat_post:.2f} m/s2  -> {s2p}  ({frac3_post:.1f}% >3)')
        print(f'  ISO 11270 s5.4   Lat jerk avg 0.5s (full)  <= 5.0 m/s3:    worst = {wjlat_full:.2f} m/s3  -> {s3f}  ({jerk_rate_note})')
        print(f'  ISO 11270 s5.4   Lat jerk avg 0.5s (excl.)  <= 5.0 m/s3:   worst = {wjlat_post:.2f} m/s3  -> {s3p}  ({jerk_rate_note})')
        print(f'  ISO 11270 s5.4   Long. decel (LKAS) <= 3.0 m/s2:           worst = {wlong:.2f} m/s2  -> {s4}')
    else:
        # ISO 11270 s5.4 LKAS steady-state (Following phase): no lane-change framing
        s1 = ok if wlat_full <= 1.0 else fail
        s2 = ok if wlat_full <= 3.0 else fail
        s3 = ok if wjlat_full <= 5.0 else fail
        print(f'  ISO 11270 s5.4   Lat accel (LKAS steady-state) <= 1.0 m/s2: worst = {wlat_full:.2f} m/s2  -> {s1}  ({frac3_full:.1f}% >3 m/s2)')
        print(f'  ISO 11270 s5.4   Lat accel (LKAS steady-state) <= 3.0 m/s2: worst = {wlat_full:.2f} m/s2  -> {s2}')
        print(f'  ISO 11270 s5.4   Lat jerk   avg 0.5s           <= 5.0 m/s3: worst = {wjlat_full:.2f} m/s3  -> {s3}')
        print(f'  ISO 11270 s5.4   Long. decel (LKAS)            <= 3.0 m/s2: worst = {wlong:.2f} m/s2  -> {s4}')

def _thw(phase_name):
    # THW threshold context:
    #   ISO 15622 s6.2.3.1 (ACC, single-vehicle):     THW >= 0.8 s
    #   ISO 22179 s6.2 (FSRA, cooperative)  :         THW >= 0.6 s
    #   SAE J3016 + CACC research (SARTRE/PATH/     :  THW >= 0.3 s (V2V-enabled platoon)
    #     Ensemble, Rajamani 2012)
    # This work targets cooperative platooning (V2V-implicit via shared PlatoonManager),
    # so the CACC threshold (0.3 s) is the applicable standard for the Following phase.
    # ISO 15622 is reported informationally as it is the closest widely-known baseline.
    _, gap_d, vx_g = _gaps_slice(phase_name)
    if gap_d is None:
        print(f'  CACC / SAE J3016  THW >= 0.3 s:  {na}')
        return
    for i, ac in enumerate(gap_actual_cols):
        g = gap_d[ac]
        mask = (vx_g > 3.0) & np.isfinite(g) & (g > 0)
        if not mask.any():
            print(f'  CACC / SAE J3016  THW slot {i} >= 0.3 s:  {na}')
            continue
        min_thw   = (g[mask] / vx_g[mask]).min()
        status_pl  = ok if min_thw >= 0.3 else fail  # platooning target
        ref_note   = f'min = {min_thw:.2f} s  (ACC single-vehicle limit, not applicable to platoon)'
        print(f'  CACC / SAE J3016  THW slot {i} >= 0.3 s (platooning):  min = {min_thw:.2f} s  -> {status_pl}')
        print(f'    [info] ISO 15622 s6.2.3.1  THW >= 0.8 s (ACC baseline, N/A for platoon):  {ref_note}')

def _ttc_merge(phase_name):
    # R79 s5.6.4.7 critical situation: TTC check during lane change.
    # Critical if an approaching vehicle from behind would need > 3 m/s2 decel.
    # Proxy: min TTC between ego and Car1 (slot 0) during Merge phase.
    _, gap_d, vx_g = _gaps_slice(phase_name)
    if gap_d is None or not gap_actual_cols:
        print(f'  R79 s5.6.4.7   TTC ego->leader during merge:  {na}')
        return
    g    = gap_d[gap_actual_cols[0]]
    # Relative velocity: ego approaching leader (positive = closing)
    t_ph = t_gaps[phase_col.isin([phase_name]).values]
    vx_leader = None
    for _name, df in platoon.items():
        vx_leader = np.interp(t_ph, df['Time'].values, df['vx'].values)
        break   # use first platoon vehicle (Car1) as leader proxy
    if vx_leader is None:
        print(f'  R79 s5.6.4.7   TTC ego->leader:  {na} (no platoon vehicle)')
        return
    rel_v = vx_g - vx_leader          # positive = closing
    with np.errstate(divide='ignore', invalid='ignore'):
        ttc = np.where(rel_v > 0.1, g / rel_v, 1e6)
    min_ttc = ttc.min()
    # R79 §5.6.4.7: critical if approaching rear vehicle needs >3 m/s2.
    # Simple TTC threshold: < 4 s is the boundary used by many ADAS standards.
    status = ok if min_ttc >= 4.0 else fail
    print(f'  R79 s5.6.4.7   Min TTC ego->leader >= 4.0 s:  min = {min_ttc:.2f} s  -> {status}')

def _merge_duration():
    # R79 s5.6.4.6.5: lane change manoeuvre < 5 s (M1/N1).
    # Measured from first lateral movement onset until ego centre reaches platoon lane centre.
    # Platoon lane centre = y_err == 0, i.e. ego y == platoonLaneY.
    # Proxy using ego_nash.csv y_err column: start = first |y_err| drop below |y_err_start|*0.9
    # end   = first |y_err| < 0.2 m (within 20 cm of platoon lane centre).
    if phase_col is None:
        print(f'  R79 s5.6.4.6.5  Lane change duration < 5 s:  {na}')
        return
    pmask = phase_col.isin(['Merge']).values
    if not pmask.any():
        print(f'  R79 s5.6.4.6.5  Lane change duration < 5 s:  {na}')
        return
    t_merge = t_gaps[pmask]
    t0_m, t1_m = t_merge[0], t_merge[-1]

    # Use y_err from ego_nash if available — it is signed dist to platoon lane [m]
    if nash is not None and 'y_err' in nash.columns and 'Time' in nash.columns:
        nmask = (nash['Time'].values >= t0_m) & (nash['Time'].values <= t1_m)
        if nmask.any():
            n_t   = nash['Time'].values[nmask]
            y_err = np.abs(nash['y_err'].values[nmask])

            # Start: Merge phase entry = ego is at its own lane centre (y_err = full lane width).
            # y_err is logged as 0 during GapSearch; at Merge entry it jumps to ~3.5 m.
            # Find the first sample with |y_err| > 0.5 m — this is the true start of travel.
            started = np.where(y_err > 0.5)[0]
            if len(started) == 0:
                dur = t1_m - t0_m
                print(f'  R79 s5.6.4.6.5  Lane change duration < 5.0 s:  {dur:.2f} s (no clear start, full Merge)  -> INFORMATIONAL')
                return
            t_start   = n_t[started[0]]
            y_err_start = y_err[started[0]]

            # End: first sample where |y_err| < 0.2 m — ego centre at platoon lane centre.
            arrived = np.where((n_t >= t_start) & (y_err < 0.2))[0]
            if len(arrived) == 0:
                dur = t1_m - t_start
                print(f'  R79 s5.6.4.6.5  Lane change duration < 5.0 s:  {dur:.2f} s (never reached platoon lane centre)  -> INFORMATIONAL')
                return
            t_arrive = n_t[arrived[0]]
            dur = t_arrive - t_start
            status = ok if dur < 5.0 else fail
            print(f'  R79 s5.6.4.6.5  Lane change duration < 5.0 s:  {dur:.2f} s  -> {status}  (own lane t={t_start:.1f}s y_err={y_err_start:.2f}m -> platoon lane t={t_arrive:.1f}s)')
            return

    # Fallback: full Merge phase duration (includes gap-matching time, not just lateral travel)
    dur = t1_m - t0_m
    print(f'  R79 s5.6.4.6.5  Lane change duration < 5.0 s:  {dur:.2f} s (full Merge phase, no y_err data)  -> INFORMATIONAL')

print('\n== Regulatory compliance by phase (UN-ECE R79 / ISO 15622 / ISO 11270) ==')

# ── GapSearch ─────────────────────────────────────────────────────────────────
_section('GapSearch phase  [ISO 15622 s6.4]')
t_e, ax_e, vx_e, psi_e = _ego_slice('GapSearch')
if ax_e is None or len(ax_e) == 0:
    print('  No GapSearch data.')
else:
    _long_iso15622(ax_e, t_e=t_e, nash_phase='GapSearch')

# ── Merge ─────────────────────────────────────────────────────────────────────
_section('Merge phase  [ISO 15622 s6.4 | R79 s5.6.4.4 | R79 s5.6.4.6.5 | R79 s5.6.4.7]')
t_e, ax_e, vx_e, psi_e = _ego_slice('Merge')
if ax_e is None or len(ax_e) == 0:
    print('  No Merge data.')
else:
    _long_iso15622(ax_e, t_e=t_e, nash_phase='Merge')
    if psi_e is not None:
        _lat_r79_iso11270(ax_e, psi_e, vx_e, t_e=t_e)
    else:
        print(f'  R79/ISO 11270 lateral checks:  {na} (no psi column)')
    _merge_duration()
    _ttc_merge('Merge')

# ── Following ─────────────────────────────────────────────────────────────────
# Cooperative platooning (V2V-implicit): THW target 0.3 s per CACC/SAE J3016.
# Long. cruising limits (accel/jerk/decel) still apply per ISO 15622 s6.4.
# ISO 15622 s6.2.3.1 (ACC THW >= 0.8 s) reported informationally only.
_section('Following phase  [ISO 15622 s6.4 | CACC/SAE J3016 (THW) | ISO 11270 s5.4]')
t_e, ax_e, vx_e, psi_e = _ego_slice('Following')
if ax_e is None or len(ax_e) == 0:
    print('  No Following data.')
else:
    _long_iso15622(ax_e, t_e=t_e, nash_phase='Following')
    _thw('Following')
    if psi_e is not None:
        _lat_r79_iso11270(ax_e, psi_e, vx_e, t_e=t_e, is_merge=False)
    else:
        print(f'  ISO 11270 lateral checks:  {na} (no psi column)')

print('=' * 59)
