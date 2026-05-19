"""
unified/visualization/experiment_plots.py — Experiment result visualisations.

Four plot groups matching the experiment classes in unified/simulation/experiments.py:

  plot_monte_carlo      — velocity/gap envelopes + gap-error histogram (2×3 grid)
  plot_q_weight_sweep   — Pareto curves: tracking error vs control effort (2×2)
  plot_r_weight_sweep   — u1/u2 decomposition per R-config (2×N panels)
  plot_lambda_sweep     — gap/y error + α as function of λ (2×3 grid)

Layout convention: top row = longitudinal, bottom row = lateral.

References:
  Zhang & Sun 2024  — MA-IDM noise modes
  Pustilnik 2025    — α = 1/(1+λ)
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

from unified.config import RESULTS_DIR
from unified.visualization.ieee_style import apply_ieee_style

# ── Colour palette (mode → colour) ───────────────────────────────────────────
_MODE_COLOUR = {
    'deterministic': '#1f77b4',   # blue
    'iid':           '#ff7f0e',   # orange
    'gp':            '#2ca02c',   # green
}
_MODE_LABEL = {
    'deterministic': 'Deterministic (IDM)',
    'iid':           'B-IDM (i.i.d. noise)',
    'gp':            'MA-IDM (SE-GP)',
}


# =============================================================================
# Helpers
# =============================================================================

def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _save_fig(fig: plt.Figure, filename: str, save: bool):
    if save:
        path = os.path.join(_ensure_results_dir(), filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[experiment_plots] Saved: {path}")


def _time_array(data: dict) -> np.ndarray:
    return np.asarray(data.get('time', []), dtype=float)


def _phase_merge_start(data: dict) -> Optional[float]:
    """First time MERGE phase is active, or None."""
    t   = _time_array(data)
    ph  = data.get('phase', [])
    for i, p in enumerate(ph):
        if p == 'MERGE' and i < len(t):
            return float(t[i])
    return None


def _envelope(trials_signal: List[np.ndarray]) -> tuple:
    """Return (mean, std) over a list of arrays, aligned to the shortest length."""
    if not trials_signal:
        empty = np.array([])
        return empty, np.array([]), np.array([])
    min_len = min(len(s) for s in trials_signal)
    if min_len == 0:
        empty = np.array([])
        return empty, np.array([]), np.array([])
    mat = np.vstack([np.asarray(s, dtype=float)[:min_len] for s in trials_signal])
    with np.errstate(all='ignore'):
        mu  = np.nanmean(mat, axis=0)
        sig = np.nanstd(mat,  axis=0)
    return mu, mu - sig, mu + sig


def _cross_trial_std(trials_signal: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time_indices, std_across_trials).  std=0 ↔ deterministic."""
    if not trials_signal:
        return np.array([]), np.array([])
    min_len = min(len(s) for s in trials_signal)
    if min_len == 0:
        return np.array([]), np.array([])
    mat = np.vstack([np.asarray(s, dtype=float)[:min_len] for s in trials_signal])
    with np.errstate(all='ignore'):
        std = np.nanstd(mat, axis=0)
    return np.arange(min_len), std


# =============================================================================
# A. Monte Carlo plots
# =============================================================================

def plot_monte_carlo(mc_results: Dict[str, List[dict]],
                     mc_summary: Dict[str, dict],
                     title: str = '',
                     show: bool = True,
                     save: bool = True):
    """
    2-row × 3-col figure designed to show what actually differs between modes:

    Row 0 — Longitudinal:
      [0,0] Cross-trial std of ego_vx(t)  — width=0 for deterministic, >0 for stochastic
      [0,1] u2_long(t) zoom (FOLLOWING)   — spiky for i.i.d., smooth bumps for GP
      [0,2] Autocorrelation of u2_long    — δ(0) for i.i.d., exp decay for GP

    Row 1 — Lateral:
      [1,0] Cross-trial std of ego_y(t)   — same logic
      [1,1] u2_lat(t) zoom (MERGE phase)  — noise character in steering
      [1,2] Bar: gap_rms ± std  and  y_rms ± std  (performance per mode)

    Why NOT mean trajectories:
      The Nash controller corrects noise → mean gap_error ≈ same for all modes.
      The *spread* across trials (std) and the *character* of u2 reveal the difference.
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'Monte Carlo Driver Comparison{" — " + title if title else ""}',
                 fontsize=11, fontweight='bold')

    modes = list(mc_results.keys())

    # ── Deterministic baseline for residual computation ─────────────────────
    det_trials = mc_results.get('deterministic', [])
    _det_ref   = det_trials[0] if det_trials else {}
    _det_u_long   = np.asarray(_det_ref.get('u_long',    []), dtype=float)
    _det_delta    = np.asarray(_det_ref.get('delta_lat', []), dtype=float)

    # ── Gather phase reference from the first deterministic trial ───────────
    _ref_data = _det_ref if _det_ref else next((mc_results[m][0]
                                                for m in modes if mc_results[m]), {})

    for mode in modes:
        trials = mc_results[mode]
        if not trials:
            continue
        clr = _MODE_COLOUR.get(mode, 'gray')
        lbl = _MODE_LABEL.get(mode, mode)

        min_len = min(len(d.get('time', [])) for d in trials)
        if min_len < 2:
            continue
        t_ref  = _time_array(trials[0])[:min_len]
        phase0 = _ref_data.get('phase', [])

        # ── [0,0] Cross-trial std of vx ───────────────────────────────────
        vx_trials = [np.asarray(d.get('ego_vx', []), dtype=float) for d in trials]
        _, std_vx = _cross_trial_std(vx_trials)
        axes[0, 0].plot(t_ref, std_vx, color=clr, label=lbl, linewidth=1.3)

        # ── [0,1] NOISE RESIDUAL u_long: trial − det_baseline (FOLLOWING) ─
        # Shows the actual noise character: flat=0 (det), spiky (i.i.d.), smooth (GP)
        follow_idx = next((i for i, p in enumerate(phase0) if p == 'FOLLOWING'), None)
        if follow_idx is not None and follow_idx + 1 < min_len:
            window_s   = min(15.0, float(t_ref[-1]) - float(t_ref[follow_idx]))
            t_zoom_end = float(t_ref[follow_idx]) + window_s
            zoom_mask  = (t_ref >= float(t_ref[follow_idx])) & (t_ref <= t_zoom_end)
        else:
            zoom_mask = np.ones(min_len, dtype=bool)

        n_show   = min(3, len(trials))
        det_base = _det_u_long[:min_len]          # deterministic reference
        for k in range(n_show):
            u_noisy  = np.asarray(trials[k].get('u_long', []), dtype=float)[:min_len]
            residual = u_noisy - det_base          # pure noise
            alpha_k  = 0.85 if k == 0 else 0.4
            axes[0, 1].plot(t_ref[zoom_mask], residual[zoom_mask],
                            color=clr, alpha=alpha_k, linewidth=0.9,
                            label=lbl if k == 0 else None)

        # ── [0,2] Autocorrelation of NOISE RESIDUAL ───────────────────────
        # Uses pre-computed noise_autocorr_all from experiments._compute_summary()
        autocorr_list = mc_summary[mode].get('noise_autocorr_all', [])
        if autocorr_list:
            min_ac_len = min(len(a) for a in autocorr_list)
            if min_ac_len > 1:
                ac_mat  = np.vstack([a[:min_ac_len] for a in autocorr_list])
                with np.errstate(all='ignore'):
                    ac_mean = np.nanmean(ac_mat, axis=0)
                    ac_std  = np.nanstd(ac_mat,  axis=0)
                dt_sim  = float(np.median(np.diff(t_ref))) if len(t_ref) > 1 else 0.01
                lags_s  = np.arange(min_ac_len) * dt_sim
                axes[0, 2].plot(lags_s, ac_mean, color=clr, label=lbl, linewidth=1.4)
                axes[0, 2].fill_between(lags_s, ac_mean - ac_std, ac_mean + ac_std,
                                        color=clr, alpha=0.15)

        # ── [1,0] Cross-trial std of ego_y ────────────────────────────────
        y_trials = [np.asarray(d.get('ego_y', []), dtype=float) for d in trials]
        _, std_y = _cross_trial_std(y_trials)
        axes[1, 0].plot(t_ref, std_y, color=clr, label=lbl, linewidth=1.3)

        # ── [1,1] NOISE RESIDUAL delta_lat: trial − det_baseline (MERGE) ──
        merge_idx = next((i for i, p in enumerate(phase0) if p == 'MERGE'), None)
        if merge_idx is not None and merge_idx + 1 < min_len:
            window_s2  = min(20.0, float(t_ref[-1]) - float(t_ref[merge_idx]))
            t_zoom2    = float(t_ref[merge_idx]) + window_s2
            zoom_mask2 = (t_ref >= float(t_ref[merge_idx])) & (t_ref <= t_zoom2)
        else:
            zoom_mask2 = np.ones(min_len, dtype=bool)

        det_base_lat = _det_delta[:min_len]
        for k in range(n_show):
            d_noisy  = np.asarray(trials[k].get('delta_lat', []), dtype=float)[:min_len]
            residual = d_noisy - det_base_lat
            alpha_k  = 0.85 if k == 0 else 0.4
            axes[1, 1].plot(t_ref[zoom_mask2], residual[zoom_mask2],
                            color=clr, alpha=alpha_k, linewidth=0.9,
                            label=lbl if k == 0 else None)

    # ── [1,2] Dual-axis bar chart: gap_rms (left, [m]) | y_rms (right, [mm]) ──
    mode_labels = [_MODE_LABEL.get(m, m) for m in modes]
    x           = np.arange(len(modes))
    w           = 0.35

    gap_means = [mc_summary[m].get('gap_rms_mean', float('nan')) for m in modes]
    gap_stds  = [mc_summary[m].get('gap_rms_std',  0.0)          for m in modes]
    y_means   = [mc_summary[m].get('y_rms_mean',   float('nan')) for m in modes]
    y_stds    = [mc_summary[m].get('y_rms_std',    0.0)          for m in modes]
    # Convert y_rms to mm so both axes read at comparable bar heights
    y_means_mm = [v * 1e3 if np.isfinite(v) else float('nan') for v in y_means]
    y_stds_mm  = [v * 1e3 for v in y_stds]

    clrs = [_MODE_COLOUR.get(m, 'gray') for m in modes]
    ax_gap = axes[1, 2]
    ax_y   = ax_gap.twinx()

    ax_gap.bar(x - w / 2, gap_means, w, yerr=gap_stds, capsize=4,
               color=clrs, alpha=0.8, label='gap RMS [m]', edgecolor='k', linewidth=0.5)
    ax_y.bar(x + w / 2, y_means_mm, w, yerr=y_stds_mm, capsize=4,
             color=clrs, alpha=0.4, label='y RMS [mm]', edgecolor='k', linewidth=0.5,
             hatch='//')

    ax_gap.set_xticks(list(x))
    ax_gap.set_xticklabels(['Det.', 'i.i.d.', 'GP'])
    ax_gap.set_ylabel('Gap error RMS [m]')
    ax_y.set_ylabel('Lateral error RMS [mm]', color='dimgray')
    ax_gap.set_title('Tracking performance ± std (FOLLOWING)')

    # Combined legend from both axes
    h1, l1 = ax_gap.get_legend_handles_labels()
    h2, l2 = ax_y.get_legend_handles_labels()
    ax_gap.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')

    # ── Labels and decoration ─────────────────────────────────────────────
    axes[0, 0].set(xlabel='Time [s]', ylabel='σ(vx) across trials [m/s]',
                   title='Long: inter-trial velocity spread')
    axes[0, 0].set_ylim(bottom=0)

    axes[0, 1].set(xlabel='Time [s]', ylabel='noise residual [m/s²]',
                   title='Long: u_long noise (trial − det. baseline)\n3 trials per mode')
    axes[0, 1].axhline(0, color='k', linewidth=0.4, linestyle='--')

    axes[0, 2].set(xlabel='Lag τ [s]', ylabel='R(τ)',
                   title='Autocorrelation of noise residual\ni.i.d.→R≈0, GP→exp(-τ²/2ℓ²)')
    axes[0, 2].axhline(0, color='k', linewidth=0.4, linestyle='--')
    axes[0, 2].set_ylim(-0.3, 1.05)

    axes[1, 0].set(xlabel='Time [s]', ylabel='σ(y) across trials [m]',
                   title='Lat: inter-trial y spread')
    axes[1, 0].set_ylim(bottom=0)

    axes[1, 1].set(xlabel='Time [s]', ylabel='noise residual [rad]',
                   title='Lat: δ noise (trial − det. baseline)\n3 trials per mode')
    axes[1, 1].axhline(0, color='k', linewidth=0.4, linestyle='--')

    # Annotation highlighting the two key findings
    ax_gap.annotate(
        'GP: lower mean = predictive benefit\nGP: larger std = temporal persistence',
        xy=(0.03, 0.97), xycoords='axes fraction',
        va='top', ha='left', fontsize=6.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8),
    )

    for ax in axes.flat:
        if ax is axes[1, 2]:
            # legend already set manually (dual-axis); only add grid
            ax.grid(True, alpha=0.3)
        else:
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, 'experiment_monte_carlo.png', save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# B1. Q-weight sweep plots
# =============================================================================

def plot_q_weight_sweep(q_results: Dict[str, list],
                        q_summary: Dict[str, list],
                        title: str = '',
                        show: bool = True,
                        save: bool = True):
    """
    2 × 3 figure:
      [0,0] Long Pareto: gap_rms vs u_long_rms as Q_pos varies
      [0,1] Long:        gap_rms vs Q_pos bar chart
      [0,2] Long:        u_long_rms vs Q_pos bar chart
      [1,0] Lat  Pareto: y_rms (FOLLOWING) vs u_lat_rms as Q_y varies
      [1,1] Lat:         y_rms (FOLLOWING) vs Q_y bar chart
      [1,2] Lat:         MERGE delta_rms vs Q_y bar chart
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Q-Weight Sweep{" — " + title if title else ""}',
                 fontsize=11, fontweight='bold')

    cmap = plt.cm.viridis

    # ── Longitudinal ─────────────────────────────────────────────────────────
    long_s = q_summary['long']
    if long_s:
        n      = len(long_s)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        q_vals   = [s['Q_pos']     for s in long_s]
        gap_vals = [s['gap_rms']   for s in long_s]
        u_vals   = [s['u_long_rms'] for s in long_s]

        for i, s in enumerate(long_s):
            axes[0, 0].scatter(s['u_long_rms'], s['gap_rms'],
                               color=colors[i], s=80, zorder=3,
                               label=f"Q={s['Q_pos']:.0f}")
        axes[0, 0].plot(u_vals, gap_vals, color='gray', linewidth=0.8, linestyle='--', zorder=2)
        axes[0, 0].set(xlabel='u_long RMS [m/s²]', ylabel='Gap error RMS [m]',
                       title='Long: Pareto (effort vs tracking)')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)

        # Current operating point: Q_pos=5000 → index 3 in default sweep (500,1000,2500,5000,10000)
        _long_cur = next((i for i, s in enumerate(long_s) if s['Q_pos'] == 5000), None)
        _long_ec  = ['red' if i == _long_cur else 'k' for i in range(n)]
        _long_lw  = [2.5  if i == _long_cur else 0.5  for i in range(n)]

        gap_stds = [s.get('gap_rms_std', 0.0) for s in long_s]
        u_stds   = [s.get('u_long_rms_std', 0.0) for s in long_s]
        axes[0, 1].bar(range(n), gap_vals,
                       yerr=gap_stds if any(v > 0 for v in gap_stds) else None,
                       color=colors, edgecolor=_long_ec, linewidth=_long_lw, capsize=4)
        axes[0, 1].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_pos', ylabel='Gap error RMS [m]',
                       title='Long: gap RMS vs Q_pos')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[0, 2].bar(range(n), u_vals,
                       yerr=u_stds if any(v > 0 for v in u_stds) else None,
                       color=colors, edgecolor=_long_ec, linewidth=_long_lw, capsize=4)
        axes[0, 2].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_pos', ylabel='u_long RMS [m/s²]',
                       title='Long: u_long RMS vs Q_pos')
        axes[0, 2].grid(True, alpha=0.3, axis='y')

    # ── Lateral ───────────────────────────────────────────────────────────────
    lat_s = q_summary['lat']
    if lat_s:
        n      = len(lat_s)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        q_vals   = [s['Q_y']       for s in lat_s]
        y_vals   = [s['y_rms']     for s in lat_s]
        u_vals   = [s['u_lat_rms'] for s in lat_s]
        m_vals   = [s.get('merge_delta_rms', float('nan')) for s in lat_s]

        # [1,0] FOLLOWING Pareto — should collapse near y=0 after fix
        for i, s in enumerate(lat_s):
            axes[1, 0].scatter(s['u_lat_rms'], s['y_rms'],
                               color=colors[i], s=80, zorder=3,
                               label=f"Q_y={s['Q_y']:.0f}")
        axes[1, 0].plot(u_vals, y_vals, color='gray', linewidth=0.8, linestyle='--', zorder=2)
        axes[1, 0].set(xlabel='δ RMS [rad]', ylabel='y error RMS [m]',
                       title='Lat FOLLOWING: Pareto (effort vs tracking)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        # Current operating point: Q_y=800 → index 2 in default sweep (200,400,800,1600,3200)
        _lat_cur = next((i for i, s in enumerate(lat_s) if s['Q_y'] == 800), None)
        _lat_ec  = ['red' if i == _lat_cur else 'k' for i in range(n)]
        _lat_lw  = [2.5  if i == _lat_cur else 0.5  for i in range(n)]

        # [1,1] FOLLOWING y_rms bar — flat (≈0) for all Q_y
        y_stds = [s.get('y_rms_std', 0.0) for s in lat_s]
        m_stds = [s.get('merge_delta_rms_std', 0.0) for s in lat_s]
        axes[1, 1].bar(range(n), y_vals,
                       yerr=y_stds if any(v > 0 for v in y_stds) else None,
                       color=colors, edgecolor=_lat_ec, linewidth=_lat_lw, capsize=4)
        axes[1, 1].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_y', ylabel='y error RMS [m]',
                       title='Lat FOLLOWING: y RMS vs Q_y')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # [1,2] MERGE delta_rms bar — NEW: Q_y selection criterion
        m_plot      = [v if np.isfinite(v) else 0.0 for v in m_vals]
        m_stds_plot = [m_stds[i] if np.isfinite(m_vals[i]) else 0.0 for i in range(n)]
        axes[1, 2].bar(range(n), m_plot,
                       yerr=m_stds_plot if any(v > 0 for v in m_stds_plot) else None,
                       color=colors, edgecolor=_lat_ec, linewidth=_lat_lw, capsize=4)
        axes[1, 2].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_y', ylabel='MERGE δ RMS [rad]',
                       title='Lat MERGE: δ_rms vs Q_y\n(lane-change effort)')
        axes[1, 2].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _save_fig(fig, 'experiment_q_sweep.png', save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# B2. R-weight sweep plots
# =============================================================================

def plot_r_weight_sweep(r_results: Dict[str, list],
                        r_summary: Dict[str, list],
                        title: str = '',
                        show: bool = True,
                        save: bool = True):
    """
    2 × 2 figure:
      [0,0] Long: u1/u2 RMS stacked bar per R-config
      [0,1] Long: gap_rms vs R2/R1 ratio
      [1,0] Lat:  u1/u2 RMS stacked bar per R-config
      [1,1] Lat:  y_rms  vs R2/R1 ratio
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'R-Weight (Effort) Sweep{" — " + title if title else ""}',
                 fontsize=11, fontweight='bold')

    def _stacked_bar(ax, summary, u1_key, u2_key, ratio_key, metric_key, metric_lbl):
        labels = [s['label'] for s in summary]
        u1s    = [s[u1_key]  for s in summary]
        u2s    = [s[u2_key]  for s in summary]
        x      = range(len(labels))
        ax.bar(x, u1s, label='u1 (system)', color='#1f77b4', edgecolor='k', linewidth=0.5)
        ax.bar(x, u2s, bottom=u1s, label='u2 (human)', color='#ff7f0e',
               edgecolor='k', linewidth=0.5)
        ax.set(xticks=list(x), xticklabels=labels)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    def _metric_line(ax, summary, ratio_key, metric_key, xlabel, ylabel, base_idx=None):
        ratios  = [s[ratio_key]  for s in summary]
        metrics = [s[metric_key] for s in summary]
        stds    = [s.get(metric_key + '_std', 0.0) for s in summary]
        ax.plot(ratios, metrics, 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
        if any(v > 0 for v in stds):
            lo = [m - s for m, s in zip(metrics, stds)]
            hi = [m + s for m, s in zip(metrics, stds)]
            ax.fill_between(ratios, lo, hi, color='#2ca02c', alpha=0.15)
        if base_idx is not None:
            ax.axvline(ratios[base_idx], color='gray', linewidth=0.8, linestyle='--',
                       label='default')
            ax.legend(fontsize=7)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.3)

    long_s = r_summary['long']
    lat_s  = r_summary['lat']

    if long_s:
        _stacked_bar(axes[0, 0], long_s, 'u1_rms', 'u2_rms', 'R2_R1', 'gap_rms', 'gap RMS')
        axes[0, 0].set(title='Long: control effort decomposition')
        _metric_line(axes[0, 1], long_s, 'R2_R1', 'gap_rms',
                     'R2/R1 ratio', 'Gap error RMS [m]', base_idx=1)
        axes[0, 1].set_title('Long: gap error vs R2/R1')

    if lat_s:
        _stacked_bar(axes[1, 0], lat_s, 'u1_rms', 'u2_rms', 'R2_R1', 'y_rms', 'y RMS')
        axes[1, 0].set(title='Lat: control effort decomposition')
        _metric_line(axes[1, 1], lat_s, 'R2_R1', 'y_rms',
                     'R2/R1 ratio', 'y error RMS [m]', base_idx=1)
        axes[1, 1].set_title('Lat: y error vs R2/R1')

        # Annotate when lateral panel is flat (range < 5% of mean) — the flatness IS the finding
        y_vals_r = [s['y_rms'] for s in lat_s]
        y_mean_r = float(np.nanmean(y_vals_r)) if y_vals_r else 1.0
        y_range_r = max(y_vals_r) - min(y_vals_r) if y_vals_r else 0.0
        if y_mean_r > 1e-9 and (y_range_r / y_mean_r) < 0.05:
            axes[1, 1].annotate('Lateral tracking\ninsensitive to R ratio\nin tested range',
                                 xy=(0.5, 0.5), xycoords='axes fraction',
                                 ha='center', va='center', fontsize=8, color='gray',
                                 bbox=dict(boxstyle='round', fc='lightyellow',
                                           ec='gray', alpha=0.7))

    fig.tight_layout()
    _save_fig(fig, 'experiment_r_sweep.png', save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# B3. Lambda sweep plots
# =============================================================================

def plot_lambda_sweep(lam_results: Dict[str, list],
                      lam_summary: Dict[str, list],
                      title: str = '',
                      show: bool = True,
                      save: bool = True,
                      dynamic_ref: Optional[dict] = None):
    """
    2 × 5 figure:
      Row 0 (long): α vs λ | gap_rms±std vs λ | u1/u2 signed-mean FOLLOWING | GP uncertainty bar | λ histogram (dynamic ref)
      Row 1 (lat):  α vs λ | duration & δ vs λ | y_rms±std vs λ (success)   | y_rms±std (success) | λ histogram (dynamic ref)

    dynamic_ref: data dict from a single simulation run with Safety Field active (no fixed λ).
      If provided, panels [0,4] and [1,4] show the actual λ distribution during FOLLOWING,
      which is what the fixed-λ ablation sweep compares against.

    α = 1/(1+λ) is the HUMAN tracking weight in J2 = α·Q1:
      small λ → large α → human dominant (tracks r2 aggressively) → fast/aggressive merge
      large λ → small α → system dominant (human withdraws)       → slow/gentle merge
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 5, figsize=(21, 7))
    fig.suptitle(f'Fixed-λ Authority Sweep{" — " + title if title else ""}',
                 fontsize=11, fontweight='bold')

    def _alpha(lam): return 1.0 / (1.0 + lam)

    def _plot_alpha_col(ax, lams, alphas):
        lam_fine = np.logspace(np.log10(min(lams)), np.log10(max(lams)), 200)
        ax.semilogx(lam_fine, _alpha(lam_fine), 'k--', linewidth=1.0,
                    label='α=1/(1+λ)')
        ax.semilogx(lams, alphas, 'o', color='#1f77b4', markersize=7)
        ax.set(xlabel='λ (log)', ylabel='α  (human tracking weight)',
               title='Human weight α vs λ\nsmall λ→human active, large λ→system carries',
               ylim=(0, 1))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_effort_col(ax, alphas, u1s, u2s, xlabel='α (human tracking weight)',
                         u1_stds=None, u2_stds=None):
        yerr1 = u1_stds if (u1_stds is not None and any(v > 0 for v in u1_stds)) else None
        yerr2 = u2_stds if (u2_stds is not None and any(v > 0 for v in u2_stds)) else None
        ax.errorbar(alphas, u1s, yerr=yerr1, fmt='o-', color='#1f77b4',
                    linewidth=1.4, markersize=6, capsize=4, label='u1 (system)')
        ax.errorbar(alphas, u2s, yerr=yerr2, fmt='s-', color='#ff7f0e',
                    linewidth=1.4, markersize=6, capsize=4, label='u2 (human)')
        ax.set(xlabel=xlabel, ylabel='RMS', title='Control effort vs α')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 0: Longitudinal ───────────────────────────────────────────────────
    if lam_summary['long']:
        s      = lam_summary['long']
        lams   = [r['lambda'] for r in s]
        alphas = [r['alpha']  for r in s]

        _plot_alpha_col(axes[0, 0], lams, alphas)

        metrics  = [r['gap_rms']              for r in s]
        met_stds = [r.get('gap_rms_std', 0.0) for r in s]
        vel_mets = [r.get('vel_rms', float('nan')) for r in s]
        vel_stds = [r.get('vel_rms_std', 0.0)      for r in s]

        ax_g = axes[0, 1]
        ax_v = ax_g.twinx()

        ax_g.semilogx(lams, metrics, 'o-', color='#2ca02c',
                      linewidth=1.5, markersize=6, label='gap RMS')
        if any(v > 0 for v in met_stds):
            lo = [m - d for m, d in zip(metrics, met_stds)]
            hi = [m + d for m, d in zip(metrics, met_stds)]
            ax_g.fill_between(lams, lo, hi, color='#2ca02c', alpha=0.15)

        if any(np.isfinite(v) for v in vel_mets):
            ax_v.semilogx(lams, vel_mets, 's--', color='#ff7f0e',
                          linewidth=1.4, markersize=6, label='vel RMS')
            if any(v > 0 for v in vel_stds):
                lo_v = [m - d for m, d in zip(vel_mets, vel_stds)]
                hi_v = [m + d for m, d in zip(vel_mets, vel_stds)]
                ax_v.fill_between(lams, lo_v, hi_v, color='#ff7f0e', alpha=0.12)
            ax_v.set_ylabel('Vel error RMS [m/s]', color='#ff7f0e')
            ax_v.tick_params(axis='y', labelcolor='#ff7f0e')

        ax_g.set(xlabel='λ (log)', ylabel='Gap error RMS [m]',
                 title='Long: gap & vel error vs λ\nhigh fixed-λ → worse tracking → adaptive needed')
        ax_g.grid(True, alpha=0.3)
        h1, l1 = ax_g.get_legend_handles_labels()
        h2, l2 = ax_v.get_legend_handles_labels()
        ax_g.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')

        _plot_effort_col(axes[0, 2], alphas,
                         [r['u1_rms'] for r in s],
                         [r['u2_rms'] for r in s],
                         u1_stds=[r.get('u1_rms_std', 0.0) for r in s],
                         u2_stds=[r.get('u2_rms_std', 0.0) for r in s])

        # [0,3] GP uncertainty bar: gap_rms_std per λ
        x_pos = np.arange(len(lams))
        axes[0, 3].bar(x_pos, met_stds, color='#2ca02c', alpha=0.7)
        axes[0, 3].set_xticks(x_pos)
        axes[0, 3].set_xticklabels([f'{l:.1f}' for l in lams],
                                    rotation=45, fontsize=7)
        axes[0, 3].set(xlabel='λ', ylabel='gap_rms std [m]',
                       title='Long: GP uncertainty vs λ\n(std over n_avg trials)')
        axes[0, 3].grid(True, alpha=0.3, axis='y')

    # ── Row 1: Lateral — MERGE metrics ───────────────────────────────────────
    if lam_summary['lat']:
        s      = lam_summary['lat']
        lams   = [r['lambda']  for r in s]
        alphas = [r['alpha']   for r in s]

        _plot_alpha_col(axes[1, 0], lams, alphas)

        # [1,1] merge_duration (left) + merge_delta_rms (right twin)
        durs = [r.get('merge_duration',  float('nan')) for r in s]
        drms = [r.get('merge_delta_rms', float('nan')) for r in s]
        ax_d  = axes[1, 1]
        ax_dr = ax_d.twinx()

        fin_lams_d = [lams[i] for i in range(len(lams)) if np.isfinite(durs[i])]
        fin_durs   = [durs[i] for i in range(len(durs))  if np.isfinite(durs[i])]
        fin_lams_r = [lams[i] for i in range(len(lams)) if np.isfinite(drms[i])]
        fin_drms   = [drms[i] for i in range(len(drms))  if np.isfinite(drms[i])]

        if fin_lams_d:
            ax_d.semilogx(fin_lams_d, fin_durs, 'o-', color='#9467bd',
                          linewidth=1.5, markersize=6, label='Merge duration [s]')
        if fin_lams_r:
            ax_dr.semilogx(fin_lams_r, fin_drms, 's--', color='#d62728',
                           linewidth=1.4, markersize=6, label='δ RMS MERGE [rad]')

        ax_d.set(xlabel='λ (log)', ylabel='Merge duration [s]',
                 title='Lat MERGE: duration & steering effort vs λ\n'
                       'λ<1 (human dominant) → merge fails (94.5s); λ≥1 → success')
        ax_d.annotate(
            'λ<1: merge fails —\nhuman target y=0.75m\n> completion thresh 0.3m',
            xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))
        ax_dr.set_ylabel('MERGE δ RMS [rad]', color='#d62728')
        ax_dr.tick_params(axis='y', labelcolor='#d62728')
        ax_d.grid(True, alpha=0.3)

        h1, l1 = ax_d.get_legend_handles_labels()
        h2, l2 = ax_dr.get_legend_handles_labels()
        ax_d.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')

        # [1,2] signed mean control during FOLLOWING — reveals tug-of-war
        # (RMS is uninformative: |u1|≈|u2| by Nash symmetry even at different λ)
        u1_fol = [r.get('u1_mean_fol', float('nan')) for r in s]
        u2_fol = [r.get('u2_mean_fol', float('nan')) for r in s]
        if any(np.isfinite(v) for v in u1_fol):
            axes[1, 2].plot(alphas, u1_fol, 'o-', color='#1f77b4', linewidth=1.4,
                            markersize=6, label='u1 mean (system)')
            axes[1, 2].plot(alphas, u2_fol, 's-', color='#ff7f0e', linewidth=1.4,
                            markersize=6, label='u2 mean (human)')
            axes[1, 2].axhline(0, color='k', linewidth=0.8, linestyle='--')
            axes[1, 2].set(xlabel='α (human tracking weight)',
                           ylabel='Mean δ FOLLOWING [rad]',
                           title='Lat FOLLOWING: signed Nash inputs vs α\n'
                                 'u1<0: system resists, u2>0: human pushes bias')
            axes[1, 2].legend(fontsize=7)
            axes[1, 2].grid(True, alpha=0.3)
        else:
            _plot_effort_col(axes[1, 2], alphas,
                             [r['u1_rms'] for r in s],
                             [r['u2_rms'] for r in s],
                             u1_stds=[r.get('u1_rms_std', 0.0) for r in s],
                             u2_stds=[r.get('u2_rms_std', 0.0) for r in s])

        # [1,3] y_rms ± std vs λ for successful merges (merge_duration < 50s).
        # n_successful annotated on each point so the selection bias is explicit:
        # at low λ very few merges succeed, so y_rms reflects only the luckiest runs.
        ok_s = [r for r in s if r.get('merge_duration', float('inf')) < 50.0]
        if ok_s:
            ok_lams = [r['lambda']                    for r in ok_s]
            ok_yrms = [r['y_rms']                     for r in ok_s]
            ok_stds = [r.get('y_rms_std', 0.0)        for r in ok_s]
            ok_ns   = [r.get('n_successful', '?')     for r in ok_s]
            axes[1, 3].semilogx(ok_lams, ok_yrms, 'o-', color='#9467bd',
                                 linewidth=1.5, markersize=6, label='y_rms (FOLLOWING)')
            if any(v > 0 for v in ok_stds):
                lo = [m - d for m, d in zip(ok_yrms, ok_stds)]
                hi = [m + d for m, d in zip(ok_yrms, ok_stds)]
                axes[1, 3].fill_between(ok_lams, lo, hi, color='#9467bd', alpha=0.2,
                                         label='±1σ GP')
            for lam, yrms, n_ok in zip(ok_lams, ok_yrms, ok_ns):
                axes[1, 3].annotate(f'n={n_ok}', xy=(lam, yrms), xytext=(0, 7),
                                     textcoords='offset points', ha='center', fontsize=6.5,
                                     color='#555555')
            axes[1, 3].set(xlabel='λ (log)', ylabel='y error RMS [m]',
                           title='Lat: y_rms vs λ  (human_y_bias=0.75 m)\n'
                                 'y_rms→0: system wins tug-of-war  (n annotated)')
            axes[1, 3].legend(fontsize=7)
            axes[1, 3].grid(True, alpha=0.3)
        else:
            axes[1, 3].text(0.5, 0.5, 'No successful merges\n(all dur ≥ 50s)',
                             ha='center', va='center', fontsize=9, color='#d62728',
                             transform=axes[1, 3].transAxes)
            axes[1, 3].set_title('Lat: y_rms vs λ\n(no successful merges)')

    # ── [0,4] & [1,4]: Safety Field λ distribution (dynamic reference) ────────
    # Shows what the Safety Field actually produces — the ground truth that
    # the fixed-λ ablation sweep compares against.
    def _draw_lambda_hist(ax, lam_arr, phase_list, key_label, colour, lam_fixed_vals):
        """Draw histogram of λ values during FOLLOWING + vertical markers for fixed-λ levels."""
        mask = np.array([p == 'FOLLOWING' for p in phase_list], dtype=bool)
        lam_fol = lam_arr[mask & np.isfinite(lam_arr)]
        if len(lam_fol) == 0:
            ax.text(0.5, 0.5, 'No FOLLOWING data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            ax.set_title(f'{key_label}: λ in FOLLOWING\n(dynamic Safety Field)')
            return
        # λ spans orders of magnitude → log-spaced bins reveal the full distribution
        _lmin = max(lam_fol.min(), 1e-3)
        _lmax = lam_fol.max()
        if _lmin < _lmax:
            log_bins = np.logspace(np.log10(_lmin), np.log10(_lmax), 30)
        else:
            log_bins = 30
        ax.hist(lam_fol, bins=log_bins, color=colour, alpha=0.75, density=True)
        ax.set_xscale('log')
        med = float(np.median(lam_fol))
        ax.axvline(med, color='k', linewidth=1.2, linestyle='--',
                   label=f'median = {med:.3f}')
        # vertical tick for each fixed-λ level tested in the sweep
        for lv in lam_fixed_vals:
            ax.axvline(lv, color='#d62728', linewidth=0.7, linestyle=':')
        ax.set(xlabel=f'λ_{key_label}', ylabel='Density',
               title=f'{key_label}: Safety Field λ in FOLLOWING\n'
                     f'(red dotted = fixed-λ ablation levels)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    if dynamic_ref is not None:
        long_lam = np.asarray(dynamic_ref.get('long_lambda', []), dtype=float)
        lat_lam  = np.asarray(dynamic_ref.get('lat_lambda',  []), dtype=float)
        phase    = dynamic_ref.get('phase', [])
        long_levels = [r['lambda'] for r in lam_summary.get('long', [])]
        lat_levels  = [r['lambda'] for r in lam_summary.get('lat',  [])]
        _draw_lambda_hist(axes[0, 4], long_lam, phase, 'long', '#2ca02c', long_levels)
        _draw_lambda_hist(axes[1, 4], lat_lam,  phase, 'lat',  '#9467bd', lat_levels)
    else:
        for ax in (axes[0, 4], axes[1, 4]):
            ax.text(0.5, 0.5, 'No dynamic reference\n(pass dynamic_ref=)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.set_title('Safety Field λ distribution\n(not available)')

    fig.tight_layout()
    _save_fig(fig, 'experiment_lambda_sweep.png', save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# Convenience: run all plots for a given experiment result set
# =============================================================================

def plot_all_experiments(mc=None, q_sweep=None, r_sweep=None, lam_sweep=None,
                         show: bool = True, save: bool = True):
    """Plot whatever experiment results are provided (None = skip that group)."""
    if mc is not None:
        plot_monte_carlo(mc.results, mc.summary, show=show, save=save)
    if q_sweep is not None:
        plot_q_weight_sweep(q_sweep.results, q_sweep.summary, show=show, save=save)
    if r_sweep is not None:
        plot_r_weight_sweep(r_sweep.results, r_sweep.summary, show=show, save=save)
    if lam_sweep is not None:
        plot_lambda_sweep(lam_sweep.results, lam_sweep.summary, show=show, save=save,
                          dynamic_ref=getattr(lam_sweep, 'dynamic_ref', None))
