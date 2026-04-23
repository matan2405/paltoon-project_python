"""
unified/visualization/plotter.py — Static plots and GIF animation for unified simulation.

Produces:
  1. Longitudinal overview  — positions, velocities, acceleration, gap, λ
  2. Lateral overview       — y, psi, steering angle, lateral force, λ_lat
  3. Bird's-eye animation   — GIF of all vehicles on road
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

from unified.visualization.ieee_style import (
    apply_ieee_style, label_subplots, shade_phases, save_fig,
    IEEE_TALL_DOUBLE, IEEE_DOUBLE_COL,
)
apply_ieee_style()

from unified.config import (
    RESULTS_DIR, VEHICLE_COLORS,
    PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y, LANE_WIDTH,
    VEHICLE_LENGTH, PLATOON_VEHICLE_LENGTH,
    PLATOON_TIME_GAP, PLATOON_STANDSTILL_DISTANCE, PLATOON_TARGET_VELOCITY,
    LONG_SAFETY_MAX_REPULSIVE_FORCE,
)


# =============================================================================
# Helpers
# =============================================================================

def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _safe_arr(data: Dict, key: str) -> np.ndarray:
    """Return data[key] as a numpy array, or empty array if missing."""
    v = data.get(key, [])
    return np.asarray(v) if len(v) else np.array([])


def _phase_spans(time: np.ndarray, phase: list):
    """Return list of (t_start, t_end, phase_name) for coloring background spans."""
    spans = []
    if not len(phase):
        return spans
    cur_phase = phase[0]
    t_start   = time[0]
    for i, p in enumerate(phase[1:], 1):
        if p != cur_phase:
            spans.append((t_start, time[i - 1], cur_phase))
            cur_phase = p
            t_start   = time[i]
    spans.append((t_start, time[-1], cur_phase))
    return spans


_PHASE_COLORS = {
    'APPROACH':  '#d0e8ff',
    'MERGE':     '#fff3cd',
    'FOLLOWING': '#d4edda',
}


def _add_phase_bg(ax, time, phase):
    """Shade phase regions — no labels (avoids duplicate legend entries)."""
    for t0, t1, pname in _phase_spans(time, list(phase)):
        ax.axvspan(t0, t1, alpha=0.18,
                   color=_PHASE_COLORS.get(pname, '#eeeeee'))


def _phase_legend(fig):
    """Single shared phase legend placed below the figure."""
    handles = [patches.Patch(facecolor=c, alpha=0.55, label=p, edgecolor='none')
               for p, c in _PHASE_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(_PHASE_COLORS),
               fontsize=7, bbox_to_anchor=(0.5, 0.0), frameon=False)


# =============================================================================
# Figure 1: Longitudinal overview
# =============================================================================

def plot_longitudinal(data: Dict, title_suffix: str = '', save: bool = True):
    """6-panel longitudinal figure (IEEE double-column, 3×2).

    (a) Vehicle positions  — platoon + ego x(t)
    (b) Velocity profiles  — platoon + ego vx(t) [km/h] + target line
    (c) Front gap          — actual vs desired Δx_front(t)
    (d) Rear gap           — actual vs desired Δx_rear(t) [from locked_follower]
    (e) Safety field       — F_leader / F_follower / F_total (EMA) breakdown
    (f) Nash shared control — u₁/u₂/u_shared [m/s²] + λ(t) on twinx
    """
    _ensure_results_dir()
    time = _safe_arr(data, 'time')
    if not len(time):
        print("[Plotter] No longitudinal data to plot.")
        return

    phase = data.get('phase', [])
    n_pv  = len(data.get('platoon_x', []))

    fig, axes = plt.subplots(3, 2, figsize=(IEEE_TALL_DOUBLE[0], IEEE_TALL_DOUBLE[1] * 1.4),
                             sharex=True)
    fig.suptitle(f'Longitudinal Overview{title_suffix}', fontsize=9)

    from unified.visualization.ieee_style import IEEE_COLORS

    # ── (a) Positions ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    _add_phase_bg(ax, time, phase)
    for i in range(n_pv):
        ax.plot(time, np.asarray(data['platoon_x'][i]),
                color=VEHICLE_COLORS[i % len(VEHICLE_COLORS)],
                lw=0.9, label=f'$P_{{{i+1}}}$')
    ax.plot(time, _safe_arr(data, 'ego_x'), color='k', lw=1.4, label='Ego')
    ax.set_ylabel(r'$x$ [m]')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.2)

    # ── (b) Velocities ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    _add_phase_bg(ax, time, phase)
    for i in range(n_pv):
        ax.plot(time, np.asarray(data['platoon_vx'][i]) * 3.6,
                color=VEHICLE_COLORS[i % len(VEHICLE_COLORS)],
                lw=0.9, label=f'$P_{{{i+1}}}$')
    ax.plot(time, _safe_arr(data, 'ego_vx') * 3.6, color='k', lw=1.4, label='Ego')
    ax.axhline(PLATOON_TARGET_VELOCITY * 3.6, color='gray', lw=0.6,
               ls=':', alpha=0.8, label='Target')
    ax.set_ylabel(r'$v_x$ [km/h]')
    ax.legend(fontsize=6, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.2)

    # ── (c) Front gap ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    _add_phase_bg(ax, time, phase)
    gap = _safe_arr(data, 'ego_gap_to_leader')
    des = _safe_arr(data, 'desired_gap')
    if len(gap):
        ax.plot(time, gap, color=IEEE_COLORS['blue'], lw=1.2, label='Actual')
    if len(des):
        ax.plot(time, des, color=IEEE_COLORS['orange'], lw=0.9, ls='--', label='Desired')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_ylabel(r'$\Delta x_\mathrm{front}$ [m]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (d) Rear gap (to locked_follower) ────────────────────────────────────
    ax = axes[1, 1]
    _add_phase_bg(ax, time, phase)
    gap_rear = _safe_arr(data, 'ego_rear_gap')
    des_rear = _safe_arr(data, 'desired_gap')   # same desired gap policy
    if len(gap_rear):
        valid = np.isfinite(gap_rear)
        if valid.any():
            ax.plot(time[valid], gap_rear[valid],
                    color=IEEE_COLORS['red'], lw=1.2, label='Actual (rear)')
    if len(des_rear):
        ax.plot(time, des_rear, color=IEEE_COLORS['orange'], lw=0.9, ls='--',
                label='Desired')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_ylabel(r'$\Delta x_\mathrm{rear}$ [m]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (e) Safety field force breakdown ─────────────────────────────────────
    ax = axes[2, 0]
    _add_phase_bg(ax, time, phase)
    f_total    = _safe_arr(data, 'long_force')
    f_leader   = _safe_arr(data, 'long_force_leader')
    f_follower = _safe_arr(data, 'long_force_follower')
    if len(f_total):
        ax.plot(time, f_total,    color=IEEE_COLORS['blue'],   lw=1.4,
                label=r'$F_\mathrm{total}$ (EMA)')
    if len(f_leader):
        ax.plot(time, f_leader,   color=IEEE_COLORS['red'],    lw=0.9, ls='--',
                label=r'$F_\mathrm{lead}$ (raw)')
    if len(f_follower):
        ax.plot(time, f_follower, color=IEEE_COLORS['green'],  lw=0.9, ls=':',
                label=r'$F_\mathrm{foll}$ (raw)')
    ax.axhline(0, color='gray', lw=0.4)
    ax.axhline( LONG_SAFETY_MAX_REPULSIVE_FORCE, color='gray', lw=0.5,
                ls=':', alpha=0.5)
    ax.axhline(-LONG_SAFETY_MAX_REPULSIVE_FORCE * 0.25, color='gray', lw=0.5,
                ls=':', alpha=0.5)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$F_\mathrm{long}$ [N]')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (f) Nash shared control + λ ──────────────────────────────────────────
    ax = axes[2, 1]
    _add_phase_bg(ax, time, phase)
    u_sh = _safe_arr(data, 'u_long')
    u1   = _safe_arr(data, 'u1_long')
    u2   = _safe_arr(data, 'u2_long')
    if len(u_sh):
        ax.plot(time, u_sh, color=IEEE_COLORS['red'],    lw=1.4,
                label=r'$u_\mathrm{shared}$')
    if len(u1):
        ax.plot(time, u1, color=IEEE_COLORS['blue'],   lw=0.8, ls='--',
                label=r'$u_1$ sys')
    if len(u2):
        ax.plot(time, u2, color=IEEE_COLORS['orange'], lw=0.8, ls='--',
                label=r'$u_2$ hum')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$a_x$ [m/s$^2$]')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2)

    lam = _safe_arr(data, 'long_lambda')
    if len(lam):
        ax2 = ax.twinx()
        ax2.semilogy(time, np.maximum(lam, 1e-3),
                     color=IEEE_COLORS['purple'], lw=0.8, ls=':', alpha=0.8,
                     label=r'$\lambda$')
        ax2.set_ylabel(r'$\lambda$', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.spines['top'].set_visible(False)

    label_subplots(axes.flat)
    _phase_legend(fig)
    fig.tight_layout(pad=0.5, rect=(0, 0.06, 1, 1))

    if save:
        path = os.path.join(RESULTS_DIR, f'unified_longitudinal{title_suffix}.pdf')
        save_fig(fig, path)
    plt.show()


# =============================================================================
# Figure 2: Lateral overview
# =============================================================================

def plot_lateral(data: Dict, title_suffix: str = '', save: bool = True):
    """4-panel lateral figure (IEEE double-column, 2×2).

    (a) Lateral position y(t) — ego trajectory + lane targets
    (b) Steering inputs δ(t)  — u₁/u₂/δ_shared [deg]
    (c) Yaw angle ψ(t) + lateral velocity vy(t) on twinx
    (d) Lateral authority λ_lat(t) + DSF force on twinx
    """
    _ensure_results_dir()
    time = _safe_arr(data, 'time')
    if not len(time):
        return

    phase = data.get('phase', [])
    fig, axes = plt.subplots(2, 2, figsize=IEEE_TALL_DOUBLE, sharex=True)
    fig.suptitle(f'Lateral Overview{title_suffix}', fontsize=9)

    from unified.visualization.ieee_style import IEEE_COLORS

    # ── (a) Lateral position ─────────────────────────────────────────────────
    ax = axes[0, 0]
    _add_phase_bg(ax, time, phase)
    ax.plot(time, _safe_arr(data, 'ego_y'), color='k', lw=1.4, label='Ego')
    ax.axhline(PLATOON_LANE_Y,       color=IEEE_COLORS['blue'],  ls='--', lw=0.8, label='Target')
    ax.axhline(HUMAN_INITIAL_LANE_Y, color=IEEE_COLORS['green'], ls=':',  lw=0.8, label='Initial')
    ax.set_ylabel(r'$y$ [m]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (b) Steering inputs ──────────────────────────────────────────────────
    ax = axes[0, 1]
    _add_phase_bg(ax, time, phase)
    delta_sh = _safe_arr(data, 'delta_lat')
    u1_lat   = _safe_arr(data, 'u1_lat')
    u2_lat   = _safe_arr(data, 'u2_lat')
    if len(delta_sh):
        ax.plot(time, np.degrees(delta_sh), color=IEEE_COLORS['red'],    lw=1.4,
                label=r'$\delta_\mathrm{shared}$')
    if len(u1_lat):
        ax.plot(time, np.degrees(u1_lat), color=IEEE_COLORS['blue'],   lw=0.8, ls='--',
                label=r'$u_1$ sys')
    if len(u2_lat):
        ax.plot(time, np.degrees(u2_lat), color=IEEE_COLORS['orange'], lw=0.8, ls='--',
                label=r'$u_2$ hum')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_ylabel(r'$\delta$ [deg]')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (c) Yaw angle + lateral velocity ────────────────────────────────────
    ax = axes[1, 0]
    _add_phase_bg(ax, time, phase)
    psi = _safe_arr(data, 'ego_psi')
    if len(psi):
        ax.plot(time, np.degrees(psi), color='k', lw=1.2, label=r'$\psi$')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\psi$ [deg]')
    ax.grid(True, alpha=0.2)

    ego_y_arr = _safe_arr(data, 'ego_y')
    if len(ego_y_arr) > 1:
        dt_val = float(time[1] - time[0]) if len(time) > 1 else 0.01
        vy_world = np.gradient(ego_y_arr, dt_val)
        ax2 = ax.twinx()
        ax2.plot(time, vy_world, color=IEEE_COLORS['sky'], lw=0.8, ls='--',
                 label=r'$\dot{y}$ (world)')
        ax2.set_ylabel(r'$\dot{y}$ [m/s]', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.spines['top'].set_visible(False)
        lines_a, lbl_a = ax.get_legend_handles_labels()
        lines_b, lbl_b = ax2.get_legend_handles_labels()
        ax.legend(lines_a + lines_b, lbl_a + lbl_b, fontsize=6)
    else:
        ax.legend(fontsize=7)

    # ── (d) Lateral λ + DSF force ────────────────────────────────────────────
    ax = axes[1, 1]
    _add_phase_bg(ax, time, phase)
    lam_lat = _safe_arr(data, 'lat_lambda')
    if len(lam_lat):
        ax.semilogy(time, np.maximum(lam_lat, 1e-3),
                    color=IEEE_COLORS['purple'], lw=1.2, label=r'$\lambda_\mathrm{lat}$')
    ax.axhline(1.0, color='k', lw=0.5, ls=':', alpha=0.6)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\lambda_\mathrm{lat}$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    lat_f = _safe_arr(data, 'lat_force')
    if len(lat_f):
        ax3 = ax.twinx()
        ax3.plot(time, lat_f, color=IEEE_COLORS['red'], lw=0.8, ls='--', alpha=0.7,
                 label=r'$F_\mathrm{DSF}$')
        ax3.set_ylabel(r'$F$ [N]', fontsize=7)
        ax3.tick_params(labelsize=6)
        ax3.spines['top'].set_visible(False)

    label_subplots(axes.flat)
    _phase_legend(fig)
    fig.tight_layout(pad=0.5, rect=(0, 0.06, 1, 1))

    if save:
        path = os.path.join(RESULTS_DIR, f'unified_lateral{title_suffix}.pdf')
        save_fig(fig, path)
    plt.show()


# =============================================================================
# Figure 3: Platoon formation analysis
# =============================================================================

def plot_platoon_analysis(data: Dict, title_suffix: str = '', save: bool = True):
    """4-panel platoon formation quality figure (IEEE double-column, 2×2).

    (a) Inter-vehicle gaps — all consecutive pairs, actual vs desired
    (b) Velocity synchronisation — ego−platoon mean speed difference [km/h]
    (c) Combined authority — λ_long and λ_lat on the same semilogy axes
    (d) DSF forces — longitudinal and lateral safety field magnitudes
    """
    _ensure_results_dir()
    time = _safe_arr(data, 'time')
    if not len(time):
        return

    phase = data.get('phase', [])
    gaps, desired, gap_labels = _compute_platoon_gaps(data)

    fig, axes = plt.subplots(2, 2, figsize=IEEE_TALL_DOUBLE, sharex=True)
    fig.suptitle(f'Platoon Formation Analysis{title_suffix}', fontsize=9)

    from unified.visualization.ieee_style import IEEE_COLORS
    _gap_colors = [IEEE_COLORS['blue'], IEEE_COLORS['green'],
                   IEEE_COLORS['red'],  IEEE_COLORS['purple']]

    # ── (a) Inter-vehicle gaps ───────────────────────────────────────────────
    ax = axes[0, 0]
    _add_phase_bg(ax, time, phase)
    for i, (g, des, lbl) in enumerate(zip(gaps, desired, gap_labels)):
        t_g = time[:len(g)]
        c   = _gap_colors[i % len(_gap_colors)]
        ax.plot(t_g, g,   color=c, lw=1.1, label=f'Gap {lbl}')
        ax.plot(t_g, des, color=c, lw=0.7, ls='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_ylabel(r'$\Delta x$ [m]')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (b) Velocity synchronisation ────────────────────────────────────────
    ax = axes[0, 1]
    _add_phase_bg(ax, time, phase)
    ego_v = _safe_arr(data, 'ego_vx')
    n_pv  = len(data.get('platoon_vx', []))
    if n_pv and len(ego_v):
        platoon_mean = np.mean([np.asarray(data['platoon_vx'][i]) for i in range(n_pv)], axis=0)
        v_diff = (ego_v - platoon_mean) * 3.6
        ax.plot(time, v_diff, color=IEEE_COLORS['red'], lw=1.2,
                label=r'$\Delta v = v_\mathrm{ego} - \bar{v}_\mathrm{platoon}$')
        ax.fill_between(time, v_diff, alpha=0.08, color=IEEE_COLORS['red'])
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_ylabel(r'$\Delta v$ [km/h]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (c) Authority ratios ─────────────────────────────────────────────────
    ax = axes[1, 0]
    _add_phase_bg(ax, time, phase)
    lam_long = _safe_arr(data, 'long_lambda')
    lam_lat  = _safe_arr(data, 'lat_lambda')
    if len(lam_long):
        ax.semilogy(time, np.maximum(lam_long, 1e-3),
                    color=IEEE_COLORS['blue'],   lw=1.1, label=r'$\lambda_\mathrm{long}$')
    if len(lam_lat):
        ax.semilogy(time, np.maximum(lam_lat, 1e-3),
                    color=IEEE_COLORS['orange'], lw=1.1, ls='--', label=r'$\lambda_\mathrm{lat}$')
    ax.axhline(1.0, color='k', lw=0.5, ls=':', alpha=0.6)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\lambda$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (d) DSF forces — longitudinal breakdown + lateral ────────────────────
    ax = axes[1, 1]
    _add_phase_bg(ax, time, phase)
    f_long    = _safe_arr(data, 'long_force')
    f_leader  = _safe_arr(data, 'long_force_leader')
    f_follower= _safe_arr(data, 'long_force_follower')
    f_lat     = _safe_arr(data, 'lat_force')
    # F_total (filtered) — thick line
    if len(f_long):
        ax.plot(time, f_long, color=IEEE_COLORS['blue'], lw=1.4,
                label=r'$F_\mathrm{long}=F_\mathrm{lead}+F_\mathrm{foll}$')
    # F_leader (raw, unfiltered) — dashed
    if len(f_leader):
        ax.plot(time, f_leader, color=IEEE_COLORS['red'], lw=0.8, ls='--',
                label=r'$F_\mathrm{lead}$ (raw)')
    # F_follower (raw, unfiltered) — dotted
    if len(f_follower):
        ax.plot(time, f_follower, color=IEEE_COLORS['green'], lw=0.8, ls=':',
                label=r'$F_\mathrm{foll}$ (raw)')
    if len(f_lat):
        ax.plot(time, f_lat, color=IEEE_COLORS['orange'], lw=1.0, ls='-.',
                label=r'$F_\mathrm{lat}$')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$F_\mathrm{DSF}$ [N]')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    label_subplots(axes.flat)
    _phase_legend(fig)
    fig.tight_layout(pad=0.5, rect=(0, 0.06, 1, 1))

    if save:
        path = os.path.join(RESULTS_DIR, f'unified_platoon_analysis{title_suffix}.pdf')
        save_fig(fig, path)
    plt.show()


# =============================================================================
# Figure 4: Bird's-eye GIF animation
# =============================================================================

def animate_birdseye(data: Dict,
                     title_suffix: str = '',
                     fps: int = 20,
                     speedup: float = 10.0,
                     save: bool = True) -> Optional[str]:
    """Produce a 4-panel bird's-eye GIF animation.

    Layout
    ------
    Top row   : Bird's-eye road view (full width) with gap labels + info box
    Bottom row: Velocities | Inter-vehicle gaps | Live status

    Parameters
    ----------
    speedup : float
        Display speed relative to real-time (e.g. 4.0 = 4× faster than real time).
    """
    _ensure_results_dir()
    time = _safe_arr(data, 'time')
    if not len(time):
        return None

    ego_x   = _safe_arr(data, 'ego_x')
    ego_y   = _safe_arr(data, 'ego_y')
    ego_psi = _safe_arr(data, 'ego_psi')
    ego_vx  = _safe_arr(data, 'ego_vx')
    n_pv    = len(data.get('platoon_x', []))

    # Subsample to match desired fps and speedup
    dt_sim    = float(time[1] - time[0]) if len(time) > 1 else 0.01
    step_size = max(1, round(speedup / (fps * dt_sim)))
    idx       = np.arange(0, len(time), step_size)

    L = VEHICLE_LENGTH
    W = 1.8  # vehicle width [m]

    road_min_y = -LANE_WIDTH * 0.5
    road_max_y =  LANE_WIDTH * 2.5
    # gap labels placed just below the bottom road edge (outside the road shading)
    gap_text_y = road_min_y - 0.4

    # Pre-compute gaps (full resolution) for bottom panels
    gaps_data, desired_data, gap_labels = _compute_platoon_gaps(data)

    _GAP_ACT_COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    _GAP_DES_COLORS = ['tab:orange', 'tab:olive', 'tab:pink', 'tab:gray',  'tab:cyan']

    # ── Figure layout ─────────────────────────────────────────────────────────
    # height_ratios: top panel is kept short so the equal-aspect road strip
    # doesn't waste too much whitespace; bottom panels take the majority.
    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 3.0], hspace=0.45, wspace=0.32)
    ax_bird = fig.add_subplot(gs[0, :])   # top: bird's eye (all 3 cols)
    ax_vel  = fig.add_subplot(gs[1, 0])   # bottom-left:  velocities
    ax_gap  = fig.add_subplot(gs[1, 1])   # bottom-mid:   gaps
    ax_stat = fig.add_subplot(gs[1, 2])   # bottom-right: live status
    fig.suptitle(f"Unified Simulation — Bird's-Eye  {title_suffix}", fontsize=11)

    def _draw_frame(i_frame: int):
        i = idx[i_frame]

        # ── Bird's-eye panel ─────────────────────────────────────────────────
        ax_bird.clear()

        # Road background
        ax_bird.axhspan(road_min_y, road_max_y, color='#cccccc', alpha=0.3)
        ax_bird.axhline(PLATOON_LANE_Y,       color='white', linewidth=2, linestyle='--')
        ax_bird.axhline(HUMAN_INITIAL_LANE_Y, color='white', linewidth=2, linestyle='--')

        # x-window: show ALL vehicles (like Longitudinal animation)
        all_x_pos = [float(ego_x[i])] + [float(data['platoon_x'][j][i]) for j in range(n_pv)]
        x_min = min(all_x_pos) - 15.0
        x_max = max(all_x_pos) + 30.0
        # ── equal aspect: adjustable='box' respects both xlim/ylim with no warning ──
        ax_bird.set_xlim(x_min, x_max)
        ax_bird.set_ylim(road_min_y - 2.0, road_max_y + 1.0)
        ax_bird.set_aspect('equal', adjustable='box')

        # Platoon vehicles (axis-aligned — no yaw)
        all_vehs: list = []   # (x_centre, label) for gap annotations
        for j in range(n_pv):
            pv_x_j = float(data['platoon_x'][j][i])
            col    = VEHICLE_COLORS[j % len(VEHICLE_COLORS)]
            rect   = patches.Rectangle(
                (pv_x_j - L / 2, PLATOON_LANE_Y - W / 2), L, W,
                linewidth=1.0, edgecolor='black', facecolor=col, alpha=0.8)
            ax_bird.add_patch(rect)
            ax_bird.text(pv_x_j, PLATOON_LANE_Y, f'P{j + 1}',
                         ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            all_vehs.append((pv_x_j, f'P{j + 1}'))

        # Ego vehicle — rotated polygon so yaw is correct in data space
        yaw = float(ego_psi[i]) if len(ego_psi) > i else 0.0
        ex, ey = float(ego_x[i]), float(ego_y[i])
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Four corners in local frame (front = +x), rotated into world frame
        loc = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
        world = loc @ np.array([[cy, sy], [-sy, cy]]) + np.array([ex, ey])
        ego_poly = patches.Polygon(world, closed=True,
                                   facecolor='black', edgecolor='white',
                                   linewidth=1.5, alpha=0.9, zorder=5)
        ax_bird.add_patch(ego_poly)
        # Heading arrow (yellow) from centre toward nose — makes small yaw visible
        nose_x = ex + (L / 2 + 0.6) * cy
        nose_y = ey + (L / 2 + 0.6) * sy
        ax_bird.annotate('', xy=(nose_x, nose_y), xytext=(ex, ey),
                         arrowprops=dict(arrowstyle='->', color='gold',
                                         lw=2.0, mutation_scale=10),
                         zorder=7)
        ax_bird.text(ex, ey, 'Ego',
                     ha='center', va='center', fontsize=6, fontweight='bold',
                     color='white', zorder=8)
        all_vehs.append((ex, 'Ego'))

        # Gap labels between consecutive vehicles (sorted front→back)
        all_vehs.sort(key=lambda v: v[0], reverse=True)
        for k in range(len(all_vehs) - 1):
            x_front, lbl_f = all_vehs[k]
            x_back,  lbl_b = all_vehs[k + 1]
            # bumper-to-bumper: rear of front vehicle to front of rear vehicle
            gap_m = (x_front - L / 2) - (x_back + L / 2)
            if gap_m > 0:
                x_mid = (x_front + x_back) / 2
                involves_ego = ('Ego' in (lbl_f, lbl_b))
                txt_color  = 'darkorange'   if involves_ego else 'darkblue'
                box_color  = 'lightyellow'  if involves_ego else 'lightcyan'
                ax_bird.text(x_mid, gap_text_y, f'{gap_m:.1f} m',
                             ha='center', va='top', fontsize=7,
                             color=txt_color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color,
                                       alpha=0.9, edgecolor=txt_color, linewidth=0.5))

        # Info box (top-right corner)
        phase_label = data['phase'][i] if isinstance(data['phase'], (list, np.ndarray)) else ''
        vx_kmh  = float(ego_vx[i]) * 3.6 if len(ego_vx) > i else 0.0
        psi_deg = np.degrees(yaw)
        g_lead  = _safe_arr(data, 'ego_gap_to_leader')
        gap_str = f"{float(g_lead[i]):.1f} m" if len(g_lead) > i else '—'
        info_txt = (f"t = {float(time[i]):.1f} s\n"
                    f"Phase : {phase_label}\n"
                    f"X     = {ex:.1f} m\n"
                    f"Y     = {ey:.3f} m\n"
                    f"\u03c8     = {psi_deg:.1f}\u00b0\n"
                    f"vx    = {vx_kmh:.1f} km/h\n"
                    f"gap\u2192leader = {gap_str}")
        ax_bird.text(0.985, 0.97, info_txt,
                     transform=ax_bird.transAxes, fontsize=7,
                     va='top', ha='right', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                               alpha=0.88, edgecolor='gray'))

        ax_bird.set_xlabel('x [m]', fontsize=8)
        ax_bird.set_ylabel('y [m]', fontsize=8)
        ax_bird.grid(True, alpha=0.2)

        # ── Velocity panel ────────────────────────────────────────────────────
        ax_vel.clear()
        for j in range(n_pv):
            vx_j = np.asarray(data['platoon_vx'][j])[:i + 1]
            ax_vel.plot(time[:i + 1], vx_j * 3.6,
                        color=VEHICLE_COLORS[j % len(VEHICLE_COLORS)],
                        linewidth=0.9, label=f'P{j + 1}')
        if len(ego_vx):
            ax_vel.plot(time[:i + 1], ego_vx[:i + 1] * 3.6,
                        'k-', linewidth=1.4, label='Ego')
        ax_vel.axhline(PLATOON_TARGET_VELOCITY * 3.6, color='gray',
                       linestyle='--', linewidth=0.8, label='Target')
        ax_vel.set_xlim(time[0], time[-1])
        ax_vel.set_ylabel('Speed [km/h]', fontsize=7)
        ax_vel.set_xlabel('Time [s]', fontsize=7)
        ax_vel.set_title('Velocities', fontsize=8)
        ax_vel.legend(fontsize=5, ncol=3)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.tick_params(labelsize=6)

        # ── Gaps panel ────────────────────────────────────────────────────────
        ax_gap.clear()
        for k, (g, des, lbl) in enumerate(zip(gaps_data, desired_data, gap_labels)):
            n_g = min(i + 1, len(g))
            ax_gap.plot(time[:n_g], g[:n_g],
                        color=_GAP_ACT_COLORS[k % len(_GAP_ACT_COLORS)],
                        linewidth=0.9, label=f'Gap {lbl}')
            ax_gap.plot(time[:n_g], des[:n_g],
                        color=_GAP_DES_COLORS[k % len(_GAP_DES_COLORS)],
                        linewidth=0.7, linestyle='--', label=f'Des {lbl}')
        ax_gap.axhline(0, color='gray', linewidth=0.5)
        ax_gap.set_xlim(time[0], time[-1])
        ax_gap.set_ylabel('Gap [m]', fontsize=7)
        ax_gap.set_xlabel('Time [s]', fontsize=7)
        ax_gap.set_title('Inter-vehicle Gaps', fontsize=8)
        ax_gap.legend(fontsize=5, ncol=2)
        ax_gap.grid(True, alpha=0.3)
        ax_gap.tick_params(labelsize=6)

        # ── Live status panel ─────────────────────────────────────────────────
        ax_stat.clear()
        phase_color = _PHASE_COLORS.get(str(phase_label), '#eeeeee')
        ax_stat.set_facecolor(phase_color)
        ax_stat.set_xlim(0, 1); ax_stat.set_ylim(0, 1)
        ax_stat.axis('off')
        status_lines = [
            f"t = {float(time[i]):.2f} s",
            "",
            f"Phase:  {phase_label}",
            "",
            f"Ego X:  {ex:.1f} m",
            f"Ego Y:  {ey:.3f} m",
            f"Ego \u03c8:  {psi_deg:.2f}\u00b0",
            f"Ego vx: {vx_kmh:.1f} km/h",
            "",
            f"gap\u2192leader: {gap_str}",
        ]
        for k, line in enumerate(status_lines):
            ax_stat.text(0.08, 0.93 - k * 0.09, line,
                         transform=ax_stat.transAxes,
                         fontsize=8, va='top', family='monospace',
                         color='#1a1a1a')
        ax_stat.set_title('Live Status', fontsize=8)

    anim = FuncAnimation(fig, _draw_frame, frames=len(idx),
                         interval=1000.0 / fps, blit=False)

    if save:
        path = os.path.join(RESULTS_DIR, f"unified_animation{title_suffix}.gif")
        try:
            writer = PillowWriter(fps=fps)
            anim.save(path, writer=writer, dpi=50)
            print(f"[Plotter] Saved: {path}")
        except Exception as exc:
            import traceback as _tb
            print(f"[Plotter] GIF save failed: {type(exc).__name__}: {exc}")
            _tb.print_exc()
            path = None
    else:
        plt.show()
        path = None

    plt.close(fig)
    return path


def _compute_platoon_gaps(data: Dict):
    """Return (gaps, desired_gaps, labels) for all consecutive vehicle pairs + ego.

    When ego is inserted between two platoon vehicles (middle / front scenarios),
    the direct P_i → P_{i+1} gap is replaced by two gaps:
      P_i → Ego  (ego_gap_to_leader, already logged)
      Ego → P_{i+1}  (ego_x - platoon_x[i+1] - L, computed here)
    This avoids the misleading ~2× gap that appears when the plotter naively
    computes P1.x - P2.x for non-adjacent vehicles.
    """
    L  = PLATOON_VEHICLE_LENGTH
    h  = PLATOON_TIME_GAP
    d0 = PLATOON_STANDSTILL_DISTANCE
    n_pv = len(data.get('platoon_x', []))
    gaps, desired, labels = [], [], []

    ego_x        = np.asarray(data.get('ego_x', []))
    ego_vx       = np.asarray(data.get('ego_vx', []))
    ego_join_time = data.get('ego_join_time')
    time         = np.asarray(data.get('time', []))

    # Detect which original pair (i-1, i) has ego inserted between them.
    # Use ego position at join time: ego_x > platoon_x[i] means ego is ahead of P_{i+1}.
    ego_inserted_before = -1   # platoon index that ego sits ahead of after joining
    if (ego_join_time is not None and len(ego_x) > 0 and len(time) > 0 and n_pv >= 2):
        join_idx = int(np.searchsorted(time, ego_join_time))
        join_idx = min(join_idx, len(ego_x) - 1)
        ex = float(ego_x[join_idx])
        for i in range(1, n_pv):
            px_lead   = float(data['platoon_x'][i - 1][join_idx])
            px_follow = float(data['platoon_x'][i][join_idx])
            if px_lead > ex > px_follow:
                ego_inserted_before = i   # ego sits between P[i-1] and P[i]
                break

    g_leader = _safe_arr(data, 'ego_gap_to_leader')
    d_leader = _safe_arr(data, 'desired_gap')

    for i in range(1, n_pv):
        if i == ego_inserted_before and len(ego_x) > 0:
            # Replace direct P[i-1]→P[i] gap with P[i-1]→Ego + Ego→P[i]
            # P[i-1] → Ego: use ego_gap_to_leader (already logged correctly)
            if len(g_leader):
                gaps.append(g_leader)
                desired.append(d_leader)
                labels.append(f'P{i}→Ego')

            # Ego → P[i]: compute from stored positions
            x_follower = np.asarray(data['platoon_x'][i])
            v_follower = np.asarray(data['platoon_vx'][i])
            n = min(len(ego_x), len(x_follower))
            gaps.append(ego_x[:n] - x_follower[:n] - L)
            desired.append(d0 + h * v_follower[:n])
            labels.append(f'Ego→P{i+1}')
        else:
            x_f = np.asarray(data['platoon_x'][i - 1])
            x_b = np.asarray(data['platoon_x'][i])
            v_b = np.asarray(data['platoon_vx'][i])
            gaps.append(x_f - x_b - L)
            desired.append(d0 + h * v_b)
            labels.append(str(i))

    # If ego was not inserted between any pair (back/front scenario), append at end
    if ego_inserted_before < 0 and len(g_leader):
        gaps.append(g_leader)
        desired.append(d_leader)
        labels.append(str(n_pv))

    return gaps, desired, labels


# =============================================================================
# Convenience: plot all
# =============================================================================

def plot_all(data: Dict, title_suffix: str = '', animate: bool = True):
    """Produce all IEEE-style plots from a simulation data dictionary."""
    plot_longitudinal(data, title_suffix=title_suffix)
    plot_lateral(data, title_suffix=title_suffix)
    plot_platoon_analysis(data, title_suffix=title_suffix)
    if animate:
        animate_birdseye(data, title_suffix=title_suffix)


__all__ = ['plot_longitudinal', 'plot_lateral', 'plot_platoon_analysis',
           'animate_birdseye', 'plot_all']
