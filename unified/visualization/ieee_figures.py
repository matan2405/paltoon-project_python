"""
unified/visualization/ieee_figures.py
IEEE-ready analytical figures for the Nash platoon merging paper.

Four stand-alone figure generators, each returning a matplotlib Figure
and saving a 300-dpi PDF to RESULTS_DIR.

Functions
---------
plot_authority_sigmoid   — λ(e) curves (Swain & Rath 2023, Fig. 7)
plot_nash_control_inputs — u1/u2/u_shared time series (Li et al. 2019, Fig. 8d-e)
plot_safety_field_heatmap — stacked DSF heatmaps (lat. Wang field + long. danger zones) at two instants
plot_gne_pareto          — J1 vs J2 Pareto frontier (Pustilnik & Borrelli 2025, Fig. 2)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse as MplEllipse
from matplotlib.colors import LogNorm
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from unified.visualization.ieee_style import (
    apply_ieee_style, label_subplots, shade_phases, save_fig,
    IEEE_SINGLE_COL, IEEE_DOUBLE_COL, IEEE_TALL_SINGLE, IEEE_TALL_DOUBLE,
    IEEE_COLORS,
)
from unified.config import (
    RESULTS_DIR,
    LONG_AUTHORITY_LAMBDA_MIN, LONG_AUTHORITY_LAMBDA_MAX,
    LONG_AUTHORITY_FORCE_MIDPOINT, LONG_AUTHORITY_K_STEEPNESS,
    LAT_AUTHORITY_LAMBDA_MIN, LAT_AUTHORITY_LAMBDA_MAX,
    LAT_AUTHORITY_SIGMOID_M1, LAT_AUTHORITY_SIGMOID_M2,
    LAT_AUTHORITY_ALPHA_FAST_THR,
    # Longitudinal safety field
    LONG_SAFETY_MIN_SAFE_DISTANCE,
    LONG_SAFETY_EMERGENCY_BRAKE_DIST, LONG_SAFETY_MAX_REPULSIVE_FORCE,
    PLATOON_TIME_GAP, PLATOON_STANDSTILL_DISTANCE,
    # Lateral DSF (Wang 2015/2016, Li 2019)
    LAT_DSF_TS, LAT_DSF_TAU, LAT_DSF_A_MIN,
    LAT_DSF_G, DSF_SPEED_COEFF, DSF_SPEED_EXPONENT, DSF_SPEED_OFFSET,
    DSF_VEHICLE_MASS, DSF_EPSILON,
    # Geometry
    VEHICLE_LENGTH, VEHICLE_WIDTH, PLATOON_LANE_Y,
)

apply_ieee_style()


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# 1.  Authority Sigmoid  λ(e)
# ============================================================

def plot_authority_sigmoid(
    save=True, show=True, results_dir=None,
):
    """Authority allocation sigmoid curves using actual simulation parameters.

    Left panel : longitudinal — λ vs DSF safety force F [N]
                 (force-based sigmoid: λ = λ_min + (λ_max-λ_min)·σ(k·(F-F_mid)))
    Right panel: lateral      — λ vs lane offset |e_y| [m]
                 (Swain & Rath 2023 Eq. 15 error-based sigmoid)

    Parameters are loaded directly from unified.config.
    """
    results_dir = _ensure(results_dir or RESULTS_DIR)
    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)

    def _sigma(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _human_pct(lam):
        """γ₂ (human weight) as percentage: γ₂ = 1/(1+λ)."""
        return 100.0 / (1.0 + lam)

    # ── Left: longitudinal (force-based) ────────────────────────────────────
    ax = axes[0]
    F_max = 3.0 * LONG_AUTHORITY_FORCE_MIDPOINT       # show up to 3× midpoint
    F = np.linspace(0, F_max, 600)
    lam_long = LONG_AUTHORITY_LAMBDA_MIN + (
        LONG_AUTHORITY_LAMBDA_MAX - LONG_AUTHORITY_LAMBDA_MIN
    ) * _sigma(LONG_AUTHORITY_K_STEEPNESS * (F - LONG_AUTHORITY_FORCE_MIDPOINT))
    gamma_hum = _human_pct(lam_long)

    ax.semilogy(F, lam_long, color=IEEE_COLORS['blue'], lw=1.2,
                label=r'$\lambda$ (left axis)')

    # Equal-authority line (λ=1 ↔ γ₁=γ₂=50%)
    ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.7, label=r'Equal authority')

    # Shade: human-dominant (λ<1) vs system-dominant (λ>1)
    ax.axhspan(LONG_AUTHORITY_LAMBDA_MIN * 0.8, 1.0,
               alpha=0.06, color=IEEE_COLORS['green'])
    ax.axhspan(1.0, LONG_AUTHORITY_LAMBDA_MAX * 1.2,
               alpha=0.06, color=IEEE_COLORS['red'])

    # Midpoint marker
    ax.axvline(LONG_AUTHORITY_FORCE_MIDPOINT, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.text(LONG_AUTHORITY_FORCE_MIDPOINT, LONG_AUTHORITY_LAMBDA_MAX * 0.5,
            r'$F_\mathrm{mid}$', ha='center', va='bottom', fontsize=7, color='k')

    ax2 = ax.twinx()
    ax2.plot(F, gamma_hum, color=IEEE_COLORS['orange'], lw=1.0, ls='--',
             label=r'$\gamma_2$ human [%]')
    ax2.set_ylabel(r'Human weight $\gamma_2$ [%]', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, 105)

    ax.set_xlim(0, F_max)
    ax.set_ylim(LONG_AUTHORITY_LAMBDA_MIN * 0.8, LONG_AUTHORITY_LAMBDA_MAX * 1.2)
    ax.set_xlabel(r'DSF safety force $F$ [N]')
    ax.set_ylabel(r'Authority ratio $\lambda$')

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=6, loc='upper left')

    # ── Right: lateral (Swain & Rath Eq. 15) ────────────────────────────────
    ax = axes[1]
    e_max = 2.0 * LAT_AUTHORITY_ALPHA_FAST_THR          # show up to 2× fast threshold
    ey = np.linspace(0, e_max, 400)
    e_norm = ey / LAT_AUTHORITY_ALPHA_FAST_THR           # normalise by fast threshold
    lam_lat = LAT_AUTHORITY_LAMBDA_MIN + (
        LAT_AUTHORITY_LAMBDA_MAX - LAT_AUTHORITY_LAMBDA_MIN
    ) * _sigma(LAT_AUTHORITY_SIGMOID_M1 * (e_norm - LAT_AUTHORITY_SIGMOID_M2))
    gamma_hum_lat = _human_pct(lam_lat)

    ax.semilogy(ey, lam_lat, color=IEEE_COLORS['blue'], lw=1.2,
                label=r'$\lambda$ (left axis)')
    ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.7, label=r'Equal authority')

    # Shade human / system dominant
    ax.axhspan(LAT_AUTHORITY_LAMBDA_MIN * 0.8, 1.0,
               alpha=0.06, color=IEEE_COLORS['green'])
    ax.axhspan(1.0, LAT_AUTHORITY_LAMBDA_MAX * 1.2,
               alpha=0.06, color=IEEE_COLORS['red'])

    # Crossover marker
    e_cross = LAT_AUTHORITY_ALPHA_FAST_THR * LAT_AUTHORITY_SIGMOID_M2
    ax.axvline(e_cross, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.axvline(LAT_AUTHORITY_ALPHA_FAST_THR, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.text(LAT_AUTHORITY_ALPHA_FAST_THR, LAT_AUTHORITY_LAMBDA_MAX * 0.5,
            r'$e_\mathrm{fast}$', ha='left', va='bottom', fontsize=7, color='gray')

    ax3 = ax.twinx()
    ax3.plot(ey, gamma_hum_lat, color=IEEE_COLORS['orange'], lw=1.0, ls='--',
             label=r'$\gamma_2$ human [%]')
    ax3.set_ylabel(r'Human weight $\gamma_2$ [%]', fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.spines['top'].set_visible(False)
    ax3.set_ylim(0, 105)

    ax.set_xlim(0, e_max)
    ax.set_ylim(LAT_AUTHORITY_LAMBDA_MIN * 0.8, LAT_AUTHORITY_LAMBDA_MAX * 1.2)
    ax.set_xlabel(r'Lateral offset $|e_y|$ [m]')
    ax.set_ylabel(r'Authority ratio $\lambda$')

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=6, loc='upper left')

    label_subplots(axes)
    fig.tight_layout(pad=0.6)

    if save:
        path = os.path.join(results_dir, f"ieee_authority_sigmoid_{_ts()}.pdf")
        save_fig(fig, path)

    if show:
        plt.show()

    return fig


# ============================================================
# 2.  Nash Control Inputs  u1 / u2 / u_shared  over time
# ============================================================

def plot_nash_control_inputs(
    sim_data: dict,
    scenario_name: str = '',
    save: bool = True,
    show: bool = True,
    results_dir=None,
):
    """Four-panel figure of Nash control strategies — longitudinal (top) + lateral (bottom).

    (a) Longitudinal control: u_shared, u₁ (system), u₂ (human) [m/s²]
    (b) Longitudinal authority ratio λ_long(t) — log scale
    (c) Lateral control: δ_shared, u₁_lat, u₂_lat [deg]
    (d) Lateral authority ratio λ_lat(t) — log scale

    Reproduces Li et al. 2019 Fig. 8d–e style.
    """
    results_dir = _ensure(results_dir or RESULTS_DIR)

    time  = np.asarray(sim_data['time'])
    _z    = np.zeros(len(time))
    _one  = np.ones(len(time))

    # Longitudinal
    u1_long = np.asarray(sim_data.get('u1',      sim_data.get('u_system',  sim_data.get('u1_long',  _z))))
    u2_long = np.asarray(sim_data.get('u2',      sim_data.get('u_human',   sim_data.get('u2_long',  _z))))
    u_long  = np.asarray(sim_data.get('u_shared', sim_data.get('u_long',   u1_long + u2_long)))
    lam_long = np.asarray(sim_data.get('lambda',  sim_data.get('authority_ratio', sim_data.get('long_lambda', _one))))

    # Lateral
    u1_lat   = np.asarray(sim_data.get('u1_lat',   _z))
    u2_lat   = np.asarray(sim_data.get('u2_lat',   _z))
    delta_sh = np.asarray(sim_data.get('delta_lat', sim_data.get('ego_delta', _z)))
    lam_lat  = np.asarray(sim_data.get('lat_lambda', _one))

    has_lateral = np.any(delta_sh != 0) or np.any(u1_lat != 0)

    phase = list(sim_data.get('phase', []))
    mt    = sim_data.get('merge_time', sim_data.get('ego_join_time', None))

    _PHASE_CLR = {
        'APPROACH':    '#d0e8ff',
        'MERGE':       '#fff3cd',
        'FOLLOWING':   '#d4edda',
        'LANE_CHANGE': '#ffe0b2',
    }

    fig, axes = plt.subplots(2, 2, figsize=IEEE_TALL_DOUBLE, sharex=True)

    def _shade(ax):
        if len(phase) == len(time):
            shade_phases(ax, time, phase, _PHASE_CLR, alpha=0.10)

    def _merge_vline(ax):
        if mt is not None:
            ax.axvline(mt, color=IEEE_COLORS['green'], lw=0.8, ls=':', label='Merge')

    # ── (a) Longitudinal control inputs ──────────────────────────────────────
    ax = axes[0, 0]
    _shade(ax)
    ax.plot(time, u_long,  color=IEEE_COLORS['red'],    lw=1.4, label=r'$u_\mathrm{shared}$', zorder=3)
    ax.plot(time, u1_long, color=IEEE_COLORS['blue'],   lw=0.9, ls='--', label=r'$u_1$ sys', alpha=0.85)
    ax.plot(time, u2_long, color=IEEE_COLORS['orange'], lw=0.9, ls='--', label=r'$u_2$ hum', alpha=0.85)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    _merge_vline(ax)
    ax.set_ylabel(r'$a_x$ [m/s$^2$]')
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (b) Longitudinal authority ratio ─────────────────────────────────────
    ax = axes[0, 1]
    _shade(ax)
    lp = np.maximum(lam_long, 1e-3)
    ax.semilogy(time, lp, color=IEEE_COLORS['purple'], lw=1.2, label=r'$\lambda_\mathrm{long}$')
    ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.7, label='Equal')
    ax.fill_between(time, lp, 1.0, where=(lp >= 1.0),
                    color=IEEE_COLORS['blue'],   alpha=0.10, label='Sys dominant')
    ax.fill_between(time, lp, 1.0, where=(lp <  1.0),
                    color=IEEE_COLORS['orange'], alpha=0.10, label='Hum dominant')
    _merge_vline(ax)
    ax.set_ylabel(r'$\lambda_\mathrm{long}$')
    ax.legend(loc='upper right', ncol=2, fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (c) Lateral control inputs ───────────────────────────────────────────
    ax = axes[1, 0]
    _shade(ax)
    if has_lateral:
        ax.plot(time, np.degrees(delta_sh), color=IEEE_COLORS['red'],    lw=1.4,
                label=r'$\delta_\mathrm{shared}$', zorder=3)
        ax.plot(time, np.degrees(u1_lat),   color=IEEE_COLORS['blue'],   lw=0.9, ls='--',
                label=r'$u_1$ sys', alpha=0.85)
        ax.plot(time, np.degrees(u2_lat),   color=IEEE_COLORS['orange'], lw=0.9, ls='--',
                label=r'$u_2$ hum', alpha=0.85)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    _merge_vline(ax)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\delta$ [deg]')
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True, alpha=0.2)

    # ── (d) Lateral authority ratio ───────────────────────────────────────────
    ax = axes[1, 1]
    _shade(ax)
    if has_lateral:
        lp_lat = np.maximum(lam_lat, 1e-3)
        ax.semilogy(time, lp_lat, color=IEEE_COLORS['purple'], lw=1.2,
                    label=r'$\lambda_\mathrm{lat}$')
        ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.7, label='Equal')
        ax.fill_between(time, lp_lat, 1.0, where=(lp_lat >= 1.0),
                        color=IEEE_COLORS['blue'],   alpha=0.10, label='Sys dominant')
        ax.fill_between(time, lp_lat, 1.0, where=(lp_lat <  1.0),
                        color=IEEE_COLORS['orange'], alpha=0.10, label='Hum dominant')
    _merge_vline(ax)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\lambda_\mathrm{lat}$')
    ax.legend(loc='upper right', ncol=2, fontsize=6)
    ax.grid(True, alpha=0.2)

    label_subplots(axes.flat)
    fig.tight_layout(pad=0.5)
    if scenario_name:
        fig.suptitle(scenario_name, fontsize=9, y=1.01)

    if save:
        tag = scenario_name.replace(' ', '_')
        path = os.path.join(results_dir, f"ieee_nash_inputs_{tag}_{_ts()}.pdf")
        save_fig(fig, path)

    if show:
        plt.show()

    return fig


# ============================================================
# 3.  Driving Safety Field — shared preamble helper
# ============================================================

def _dsf_preamble(sim_data, timesteps_idx, x_window, y_range):
    """Parse sim_data arrays shared by both DSF figure functions."""
    time   = np.asarray(sim_data['time'])
    N      = len(time)
    ego_x  = np.asarray(sim_data.get('ego_x',  sim_data.get('human_x',  np.zeros(N))))
    ego_y  = np.asarray(sim_data.get('ego_y',  sim_data.get('human_y',  np.zeros(N))))
    ego_vx = np.asarray(sim_data.get('ego_vx', np.full(N, 25.0)))

    if timesteps_idx is None:
        timesteps_idx = [N // 4, 3 * N // 4]
    timesteps_idx = [min(int(i), N - 1) for i in timesteps_idx[:2]]

    platoon_x_list  = sim_data.get('platoon_x',  [])
    platoon_vx_list = sim_data.get('platoon_vx', [])
    platoon_y_list  = sim_data.get('platoon_y',  [])

    platoon_veh = []
    for j, px in enumerate(platoon_x_list):
        py  = (np.asarray(platoon_y_list[j])
               if j < len(platoon_y_list) else np.full(N, PLATOON_LANE_Y))
        pvx = (np.asarray(platoon_vx_list[j])
               if j < len(platoon_vx_list) else np.full(N, 25.0))
        platoon_veh.append((f'P{j+1}', np.asarray(px), py, pvx))

    leader_idx_arr   = np.asarray(
        sim_data.get('ego_leader_idx',   np.full(N, -1, dtype=int)), dtype=int)
    follower_idx_arr = np.asarray(
        sim_data.get('ego_follower_idx', np.full(N, -1, dtype=int)), dtype=int)
    long_force_arr     = np.asarray(sim_data.get('long_force',          np.zeros(N)))
    lat_force_arr      = np.asarray(sim_data.get('lat_force',           np.zeros(N)))
    gap_arr            = np.asarray(sim_data.get('ego_gap_to_leader',   np.full(N, np.nan)))
    des_gap_arr        = np.asarray(sim_data.get('desired_gap',         np.full(N, np.nan)))
    dsf_leader_a_arr   = np.asarray(sim_data.get('dsf_leader_a',        np.full(N, np.nan)))
    dsf_follower_a_arr = np.asarray(sim_data.get('dsf_follower_a',      np.full(N, np.nan)))

    return (time, N, ego_x, ego_y, ego_vx, timesteps_idx, platoon_veh,
            leader_idx_arr, follower_idx_arr,
            long_force_arr, lat_force_arr, gap_arr, des_gap_arr,
            dsf_leader_a_arr, dsf_follower_a_arr)


def _draw_platoon_vehicles(ax, platoon_veh, dsf_vehicles, t_idx, x_lo, x_hi,
                           v_len, v_wid, dark_bg=False):
    """Draw platoon vehicle boxes; skip any vehicle outside the x window."""
    for vid, px_arr, py_arr, _ in platoon_veh:
        px, py = float(px_arr[t_idx]), float(py_arr[t_idx])
        if px + v_len / 2 < x_lo or px - v_len / 2 > x_hi:
            continue          # outside view — do not draw
        is_dsf = any(pv[0] == vid for pv in dsf_vehicles)
        fc = IEEE_COLORS['sky'] if is_dsf else '#aaccee'
        ec = 'navy'             if is_dsf else '#667788'
        lw = 1.2               if is_dsf else 0.6
        rect = mpatches.FancyBboxPatch(
            (px - v_len / 2, py - v_wid / 2), v_len, v_wid,
            boxstyle='round,pad=0.1', lw=lw,
            edgecolor=ec, facecolor=fc, alpha=0.90, zorder=5)
        ax.add_patch(rect)
        txt_color = 'white' if dark_bg else 'navy'
        ax.text(px, py, str(vid),
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=txt_color, zorder=6)


# ============================================================
# 3a. Lateral DSF Heatmap
# ============================================================

def plot_lateral_dsf_heatmap(
    sim_data: dict,
    timesteps_idx: list = None,
    x_window: float = 80.0,
    y_range: tuple = (-5.0, 5.0),
    save: bool = True,
    show: bool = True,
    results_dir=None,
    scenario_name: str = '',
):
    """Two-panel lateral DSF heatmap (Wang et al. 2015/2016) at two time instants.

    Background colour shows E = G·M_obs/r*, r* = sqrt((dx/a)^2 + (dy/b)^2).
    Cyan dashed ellipse marks r*=1 isocontour of the immediate leader/follower.
    """
    results_dir = _ensure(results_dir or RESULTS_DIR)
    (time, N, ego_x, ego_y, ego_vx, timesteps_idx, platoon_veh,
     leader_idx_arr, follower_idx_arr,
     _, lat_force_arr, _, _,
     dsf_leader_a_arr, dsf_follower_a_arr) = _dsf_preamble(
         sim_data, timesteps_idx, x_window, y_range)

    v_len = VEHICLE_LENGTH
    v_wid = VEHICLE_WIDTH

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)

    for panel, t_idx in enumerate(timesteps_idx):
        ax = axes[panel]

        ex_now = float(ego_x[t_idx])
        ey_now = float(ego_y[t_idx])
        ev_now = float(ego_vx[t_idx])

        li = int(leader_idx_arr[t_idx])
        fi = int(follower_idx_arr[t_idx])
        leader_pv   = platoon_veh[li] if 0 <= li < len(platoon_veh) else None
        follower_pv = platoon_veh[fi] if 0 <= fi < len(platoon_veh) else None
        dsf_vehicles = [pv for pv in (leader_pv, follower_pv) if pv is not None]

        x_lo = ex_now - x_window / 2
        x_hi = ex_now + x_window / 2
        if leader_pv is not None:
            x_hi = max(x_hi, float(leader_pv[1][t_idx]) + v_len / 2 + 5.0)
        if follower_pv is not None:
            x_lo = min(x_lo, float(follower_pv[1][t_idx]) - v_len / 2 - 5.0)

        xg = np.linspace(x_lo, x_hi, 400)
        yg = np.linspace(y_range[0], y_range[1], 100)
        XX, YY = np.meshgrid(xg, yg)

        # Sum over leader + follower only
        ZZ = np.zeros_like(XX)
        for _, px_arr, py_arr, pvx_arr in dsf_vehicles:
            px  = float(px_arr[t_idx])
            py  = float(py_arr[t_idx])
            pv  = float(pvx_arr[t_idx])
            dv  = abs(ev_now - pv)
            a   = max(dv * LAT_DSF_TS, LAT_DSF_A_MIN)
            b   = LAT_DSF_TAU
            M_obs = DSF_VEHICLE_MASS * (
                DSF_SPEED_COEFF * abs(pv) ** DSF_SPEED_EXPONENT + DSF_SPEED_OFFSET)
            r_star = np.sqrt(((XX - px) / a) ** 2 + ((YY - py) / b) ** 2)
            ZZ += LAT_DSF_G * M_obs / np.maximum(r_star, DSF_EPSILON)

        vmax_dsf = float(np.percentile(ZZ, 99)) if ZZ.max() > 0 else 1.0
        vmin_dsf = max(float(ZZ.min()) + 1e-9, 1e-6)
        pcm = ax.pcolormesh(XX, YY, np.clip(ZZ, vmin_dsf, vmax_dsf),
                            shading='gouraud', cmap='YlOrRd',
                            norm=LogNorm(vmin=vmin_dsf, vmax=vmax_dsf))
        cb = fig.colorbar(pcm, ax=ax, pad=0.01, format='%.1e')
        cb.set_label(r'Lat. DSF  $E\!=\!G M_\mathrm{obs}/r^*$  [J/kg]', fontsize=7)

        if vmax_dsf > vmin_dsf:
            ax.contour(XX, YY, ZZ,
                       levels=[0.20 * vmax_dsf, 0.50 * vmax_dsf, 0.80 * vmax_dsf],
                       colors='white', linewidths=0.7, alpha=0.75,
                       linestyles=[':', '--', '-'])

        # Draw r*=1 ellipses for leader + follower only
        _ell_labeled = False
        for vid, px_arr, py_arr, pvx_arr in dsf_vehicles:
            px = float(px_arr[t_idx])
            py = float(py_arr[t_idx])
            pv = float(pvx_arr[t_idx])
            if px + v_len / 2 < x_lo or px - v_len / 2 > x_hi:
                continue   # vehicle outside x-window — skip
            dv = abs(ev_now - pv)
            a  = max(dv * LAT_DSF_TS, LAT_DSF_A_MIN)
            b  = LAT_DSF_TAU
            r_ego = np.sqrt(((ex_now - px) / a) ** 2 + ((ey_now - py) / b) ** 2)
            lbl = (fr'$r^*\!=\!1$: $a\!=\!{a:.1f}$ m, $b\!=\!{b:.1f}$ m'
                   if not _ell_labeled else '_nolegend_')
            ax.add_patch(MplEllipse(
                (px, py), width=2 * a, height=2 * b, angle=0,
                fill=False, edgecolor='cyan', lw=1.4, ls='--', zorder=7, label=lbl))
            # Annotate r*_ego so the viewer can see whether ego is inside/outside
            ax.text(px, py + b + 0.25, fr'$r^*={r_ego:.2f}$',
                    ha='center', va='bottom', fontsize=6, color='cyan', zorder=8)
            _ell_labeled = True

        lat_f_val = float(lat_force_arr[t_idx])
        ax.text(0.01, 0.04,
                fr'Lat. DSF: $F_{{lat}}={lat_f_val:.1f}$ N',
                transform=ax.transAxes, fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222',
                          alpha=0.75, edgecolor='none'))

        for ly in [y_range[0] + 0.15, PLATOON_LANE_Y, y_range[1] - 0.15]:
            ax.axhline(ly, color='white', lw=0.8, ls='--', alpha=0.5)

        _draw_platoon_vehicles(ax, platoon_veh, dsf_vehicles, t_idx,
                               x_lo, x_hi, v_len, v_wid, dark_bg=True)

        rect_ego = mpatches.FancyBboxPatch(
            (ex_now - v_len / 2, ey_now - v_wid / 2), v_len, v_wid,
            boxstyle='round,pad=0.1', lw=1.2,
            edgecolor='darkred', facecolor='#cc0000', alpha=0.95, zorder=5)
        ax.add_patch(rect_ego)
        ax.text(ex_now, ey_now, 'Ego',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white', zorder=6)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(*y_range)
        ax.set_aspect('equal')   # ensures ellipse a/b axes appear geometrically correct
        ax.set_ylabel(r'$y$ [m]', fontsize=9)
        ax.set_title(fr'$t = {time[t_idx]:.1f}$ s', fontsize=9)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85, ncol=2)
        ax.grid(False)
        if panel == 1:
            ax.set_xlabel(r'$x$ [m]', fontsize=9)

    label_subplots(axes, x=-0.04, y=1.06)
    if scenario_name:
        fig.suptitle(scenario_name, fontsize=10)

    if save:
        tag  = scenario_name.replace(' ', '_')
        path = os.path.join(results_dir, f"ieee_lat_dsf_{tag}_{_ts()}.pdf")
        save_fig(fig, path)

    if show:
        plt.show()

    return fig


# ============================================================
# 3b. Longitudinal DSF Road Diagram
# ============================================================

def plot_longitudinal_dsf(
    sim_data: dict,
    timesteps_idx: list = None,
    x_window: float = 80.0,
    y_range: tuple = (-5.0, 5.0),
    save: bool = True,
    show: bool = True,
    results_dir=None,
    scenario_name: str = '',
):
    """Two-panel longitudinal DSF 1-D heatmap at two time instants.

    Road bird's-eye view (same style as lateral DSF) with 1-D colour field:
    background colour = F_long that would act on the ego at each x position.
    Red = high repulsive force (ego too close to leader), green = safe zone.
    Force model mirrors coordinator._long_safety_force() piecewise formula.
    """
    results_dir = _ensure(results_dir or RESULTS_DIR)
    (time, N, ego_x, ego_y, ego_vx, timesteps_idx, platoon_veh,
     leader_idx_arr, follower_idx_arr,
     long_force_arr, _, gap_arr, des_gap_arr,
     _, _) = _dsf_preamble(sim_data, timesteps_idx, x_window, y_range)

    d_brake = LONG_SAFETY_EMERGENCY_BRAKE_DIST
    F_max   = LONG_SAFETY_MAX_REPULSIVE_FORCE
    v_len   = VEHICLE_LENGTH
    v_wid   = VEHICLE_WIDTH

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)

    for panel, t_idx in enumerate(timesteps_idx):
        ax = axes[panel]

        ex_now = float(ego_x[t_idx])
        ey_now = float(ego_y[t_idx])
        ev_now = float(ego_vx[t_idx])

        li = int(leader_idx_arr[t_idx])
        fi = int(follower_idx_arr[t_idx])
        leader_pv   = platoon_veh[li] if 0 <= li < len(platoon_veh) else None
        follower_pv = platoon_veh[fi] if 0 <= fi < len(platoon_veh) else None
        dsf_vehicles = [pv for pv in (leader_pv, follower_pv) if pv is not None]

        x_lo = ex_now - x_window / 2
        x_hi = ex_now + x_window / 2
        if leader_pv is not None:
            x_hi = max(x_hi, float(leader_pv[1][t_idx]) + v_len / 2 + 5.0)
        if follower_pv is not None:
            x_lo = min(x_lo, float(follower_pv[1][t_idx]) - v_len / 2 - 5.0)

        # Desired gap (actual saved value or estimated from ego speed)
        des_gap_val = float(des_gap_arr[t_idx])
        if np.isnan(des_gap_val):
            des_gap_val = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ev_now

        # ── 1-D heatmap: F_long(x) — force if ego centre were at x ───────────
        xg = np.linspace(x_lo, x_hi, 400)
        yg = np.linspace(y_range[0], y_range[1], 80)

        # ── Leader contribution: mirrors coordinator._long_safety_force() ───
        F_x = np.zeros(len(xg))
        if leader_pv is not None:
            lx_rear = float(leader_pv[1][t_idx]) - v_len / 2
            gaps_to_leader = lx_rear - (xg + v_len / 2)
            F_leader_x = np.where(
                gaps_to_leader < d_brake,
                F_max,
                np.clip(
                    (des_gap_val - gaps_to_leader) / max(des_gap_val, 1.0) * F_max,
                    -0.25 * F_max, F_max,
                )
            )
            F_leader_x = np.where(gaps_to_leader <= 0, F_max, F_leader_x)
            F_x += F_leader_x

        # ── Follower contribution: mirrors coordinator._long_follower_force_raw() ──
        # F_follower >= 0: follower too close pushes ego forward (risk)
        if follower_pv is not None:
            fx_front = float(follower_pv[1][t_idx]) + v_len / 2
            gaps_rear    = (xg - v_len / 2) - fx_front
            des_gap_rear = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ev_now
            d_brake_rear = LONG_SAFETY_EMERGENCY_BRAKE_DIST
            d_safe_rear  = LONG_SAFETY_MIN_SAFE_DISTANCE
            near_t = (1.0 - (gaps_rear - d_brake_rear)
                      / np.maximum(d_safe_rear - d_brake_rear, 1e-3))
            prop   = np.maximum(
                0.0,
                (des_gap_rear - gaps_rear) / np.maximum(des_gap_rear, 1.0) * F_max,
            )
            F_follower_x = np.where(
                gaps_rear < d_brake_rear,
                F_max,
                np.where(gaps_rear < d_safe_rear,
                         np.clip(near_t, 0.0, 1.0) * F_max,
                         prop),
            )
            F_follower_x = np.where(gaps_rear <= 0, F_max, F_follower_x)
            F_x -= F_follower_x  # negate for heatmap: follower danger shown in blue

        # Tile in y direction: field is constant across lane width (1-D)
        ZZ = np.tile(F_x, (len(yg), 1))
        XX, YY = np.meshgrid(xg, yg)

        # Diverging colormap: blue=follower danger (<0), white=safe (0), red=leader danger (>0)
        vmin_f = -F_max
        vmax_f =  F_max
        pcm = ax.pcolormesh(XX, YY, ZZ, shading='gouraud', cmap='seismic',
                            vmin=vmin_f, vmax=vmax_f, alpha=0.55, zorder=1)
        cb = fig.colorbar(pcm, ax=ax, pad=0.01)
        cb.set_label(r'Long. DSF  $F_\mathrm{long}$ [N]  (red=leader danger, blue=follower danger)', fontsize=7)

        # ── Desired-gap boundary (leader) and min-safe boundary (follower) ───
        if leader_pv is not None and not np.isnan(des_gap_val):
            x_des_line = float(leader_pv[1][t_idx]) - v_len / 2 - des_gap_val - v_len / 2
            ax.axvline(x_des_line, color='#ff8800', lw=1.2, ls='--',
                       alpha=0.9, zorder=4,
                       label=fr'$\Delta x_{{des}}={des_gap_val:.1f}$ m')
        if follower_pv is not None:
            fx_front = float(follower_pv[1][t_idx]) + v_len / 2
            x_min_safe = fx_front + LONG_SAFETY_MIN_SAFE_DISTANCE + v_len / 2
            ax.axvline(x_min_safe, color='#5555ff', lw=1.0, ls=':',
                       alpha=0.85, zorder=4,
                       label=fr'Follower min-safe $d={LONG_SAFETY_MIN_SAFE_DISTANCE:.0f}$ m')

        # ── Lane guides ───────────────────────────────────────────────────────
        for ly in [y_range[0] + 0.15, PLATOON_LANE_Y, y_range[1] - 0.15]:
            ax.axhline(ly, color='white', lw=0.8, ls='--', alpha=0.5)

        # ── Platoon vehicles ──────────────────────────────────────────────────
        _draw_platoon_vehicles(ax, platoon_veh, dsf_vehicles, t_idx,
                               x_lo, x_hi, v_len, v_wid, dark_bg=True)

        rect_ego = mpatches.FancyBboxPatch(
            (ex_now - v_len / 2, ey_now - v_wid / 2), v_len, v_wid,
            boxstyle='round,pad=0.1', lw=1.2,
            edgecolor='darkred', facecolor='#cc0000', alpha=0.95, zorder=5)
        ax.add_patch(rect_ego)
        ax.text(ex_now, ey_now, 'Ego',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white', zorder=6)

        # ── Force annotation ─────────────────────────────────────────────────
        N_sim = len(time)
        long_f_leader_arr   = np.asarray(
            sim_data.get('long_force_leader',   np.zeros(N_sim)))
        long_f_follower_arr = np.asarray(
            sim_data.get('long_force_follower', np.zeros(N_sim)))
        long_f_total = float(long_force_arr[t_idx])
        long_f_ldr   = float(long_f_leader_arr[t_idx])
        long_f_flw   = float(long_f_follower_arr[t_idx])
        actual_gap = float(gap_arr[t_idx])
        gap_str = f'{actual_gap:.1f}' if not np.isnan(actual_gap) else '\u2014'
        des_str = f'{des_gap_val:.1f}' if not np.isnan(des_gap_val) else '\u2014'
        ax.text(0.01, 0.04,
                fr'$F_{{ldr}}={long_f_ldr:.0f}$ N  '
                fr'$F_{{flw}}={long_f_flw:.0f}$ N  '
                fr'$F_{{tot}}={long_f_total:.0f}$ N   '
                fr'$\Delta x={gap_str}$ m,  $\Delta x_{{des}}={des_str}$ m',
                transform=ax.transAxes, fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222',
                          alpha=0.75, edgecolor='none'))

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(*y_range)
        ax.set_ylabel(r'$y$ [m]', fontsize=9)
        ax.set_title(fr'$t = {time[t_idx]:.1f}$ s', fontsize=9)
        _h, _l = ax.get_legend_handles_labels()
        if _h:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
        ax.grid(False)
        if panel == 1:
            ax.set_xlabel(r'$x$ [m]', fontsize=9)

    label_subplots(axes, x=-0.04, y=1.06)
    if scenario_name:
        fig.suptitle(scenario_name, fontsize=10)

    if save:
        tag  = scenario_name.replace(' ', '_')
        path = os.path.join(results_dir, f"ieee_long_dsf_{tag}_{_ts()}.pdf")
        save_fig(fig, path)

    if show:
        plt.show()

    return fig


# ============================================================
# 3.  Backward-compatible wrapper (calls both DSF figures)
# ============================================================

def plot_safety_field_heatmap(sim_data, timesteps_idx=None, x_window=80.0,
                               y_range=(-5.0, 5.0), save=True, show=True,
                               results_dir=None, scenario_name=''):
    """Convenience wrapper: produces both lateral and longitudinal DSF figures."""
    plot_lateral_dsf_heatmap(sim_data, timesteps_idx=timesteps_idx,
                              x_window=x_window, y_range=y_range,
                              save=save, show=show, results_dir=results_dir,
                              scenario_name=scenario_name)
    plot_longitudinal_dsf(sim_data, timesteps_idx=timesteps_idx,
                          x_window=x_window, y_range=y_range,
                          save=save, show=show, results_dir=results_dir,
                          scenario_name=scenario_name)


# ============================================================
# 4.  GNE Cost Tradeoff (Pareto frontier)
# ============================================================

def plot_gne_pareto(
    sim_data_list: list = None,
    alpha_arr: np.ndarray = None,
    J1_arr: np.ndarray = None,
    J2_arr: np.ndarray = None,
    save: bool = True,
    show: bool = True,
    results_dir=None,
    scenario_name: str = '',
):
    """J₁ vs J₂ Pareto frontier as function of system authority α.

    Reproduces Pustilnik & Borrelli (2025) Fig. 2 style.

    Three calling modes
    -------------------
    1. ``sim_data_list`` : list of sim_data dicts, one per authority level.
       J_i are computed as ΣΔt · u_i².
    2. ``alpha_arr, J1_arr, J2_arr`` : pre-computed arrays (one entry per run).
    3. No arguments → illustrative synthetic data.

    Left  (a): Pareto scatter  J₁ vs J₂, coloured by α.
    Right (b): J₁(α) and J₂(α) as line plots with cost gap shading.
    """
    results_dir = _ensure(results_dir or RESULTS_DIR)

    # ── Resolve inputs ──────────────────────────────────────────────────────
    if J1_arr is None or J2_arr is None:
        if sim_data_list is not None:
            alpha_list, j1_list, j2_list = [], [], []
            for sd in sim_data_list:
                time = np.asarray(sd.get('time', [0.0, 1.0]))
                dt   = float(time[1] - time[0]) if len(time) > 1 else 0.01
                lam  = np.asarray(sd.get('lambda', sd.get('authority_ratio', [1.0])))
                u1   = np.asarray(sd.get('u1',     sd.get('u_system',  np.zeros_like(lam))))
                u2   = np.asarray(sd.get('u2',     sd.get('u_human',   np.zeros_like(lam))))
                alpha_list.append(float(np.mean(lam) / (1.0 + np.mean(lam))))
                j1_list.append(float(np.sum(u1 ** 2) * dt))
                j2_list.append(float(np.sum(u2 ** 2) * dt))
            alpha_arr = np.array(alpha_list)
            J1_arr    = np.array(j1_list)
            J2_arr    = np.array(j2_list)
        else:
            # Synthetic illustrative data
            alpha_arr = np.linspace(0.02, 0.98, 60)
            J1_arr = 1.0 / (1.0 + 4.0 * alpha_arr) + 0.03 * np.random.default_rng(0).random(60)
            J2_arr = 1.0 / (1.0 + 4.0 * (1.0 - alpha_arr)) + 0.03 * np.random.default_rng(1).random(60)

    alpha_arr = np.asarray(alpha_arr)
    J1_arr    = np.asarray(J1_arr)
    J2_arr    = np.asarray(J2_arr)

    # Normalise for readability
    J1_n = J1_arr / (J1_arr.max() or 1.0)
    J2_n = J2_arr / (J2_arr.max() or 1.0)

    # Sort by alpha for line plot
    sort_idx = np.argsort(alpha_arr)
    alpha_s, J1_s, J2_s = alpha_arr[sort_idx], J1_n[sort_idx], J2_n[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)

    # ── (a) Pareto scatter ──────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(J1_n, J2_n, c=alpha_arr, cmap='coolwarm',
                    s=18, zorder=3, vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=ax,
                      label=r'System authority $\alpha = \lambda/(1+\lambda)$',
                      shrink=0.88, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    # Mark the equal-authority point (α ≈ 0.5)
    mid = np.argmin(np.abs(alpha_arr - 0.5))
    ax.scatter(J1_n[mid], J2_n[mid], marker='*',
               s=70, color='k', zorder=5, label=r'$\alpha = 0.5$')

    ax.set_xlabel(r'Normalised $J_1$ (system)')
    ax.set_ylabel(r'Normalised $J_2$ (human)')
    ax.legend(fontsize=7)

    # ── (b) J1, J2 vs alpha ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(alpha_s, J1_s, color=IEEE_COLORS['blue'],
            lw=1.2, label=r'$J_1$ (system cost)')
    ax.plot(alpha_s, J2_s, color=IEEE_COLORS['orange'],
            lw=1.2, label=r'$J_2$ (human cost)')
    ax.fill_between(alpha_s, J1_s, J2_s,
                    alpha=0.08, color=IEEE_COLORS['purple'], label='Cost gap')
    ax.axvline(0.5, color='k', lw=0.6, ls=':', alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 1.0,
            r'$\alpha=0.5$', ha='center', va='top', fontsize=7, color='k')

    ax.set_xlabel(r'System authority $\alpha$')
    ax.set_ylabel('Normalised cost')
    ax.legend(fontsize=7)

    label_subplots(axes)
    fig.tight_layout(pad=0.6)
    if scenario_name:
        fig.suptitle(scenario_name, fontsize=9, y=1.02)

    if save:
        tag = scenario_name.replace(' ', '_')
        path = os.path.join(results_dir, f"ieee_gne_pareto_{tag}_{_ts()}.pdf")
        save_fig(fig, path)

    if show:
        plt.show()

    return fig


__all__ = [
    'plot_authority_sigmoid',
    'plot_nash_control_inputs',
    'plot_lateral_dsf_heatmap',
    'plot_longitudinal_dsf',
    'plot_safety_field_heatmap',
    'plot_gne_pareto',
]
