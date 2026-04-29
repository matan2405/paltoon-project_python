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
    2 × 2 figure:
      [0,0] Long Pareto: gap_rms vs u_long_rms as Q_pos varies
      [0,1] Long:        gap_rms vs Q_pos bar chart
      [1,0] Lat  Pareto: y_rms   vs u_lat_rms  as Q_y varies
      [1,1] Lat:         y_rms   vs Q_y  bar chart
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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

        bars = axes[0, 1].bar(range(n), gap_vals, color=colors, edgecolor='k', linewidth=0.5)
        axes[0, 1].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_pos', ylabel='Gap error RMS [m]',
                       title='Long: gap RMS vs Q_pos')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # ── Lateral ───────────────────────────────────────────────────────────────
    lat_s = q_summary['lat']
    if lat_s:
        n      = len(lat_s)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        q_vals  = [s['Q_y']      for s in lat_s]
        y_vals  = [s['y_rms']    for s in lat_s]
        u_vals  = [s['u_lat_rms'] for s in lat_s]

        for i, s in enumerate(lat_s):
            axes[1, 0].scatter(s['u_lat_rms'], s['y_rms'],
                               color=colors[i], s=80, zorder=3,
                               label=f"Q_y={s['Q_y']:.0f}")
        axes[1, 0].plot(u_vals, y_vals, color='gray', linewidth=0.8, linestyle='--', zorder=2)
        axes[1, 0].set(xlabel='δ RMS [rad]', ylabel='y error RMS [m]',
                       title='Lat: Pareto (effort vs tracking)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(range(n), y_vals, color=colors, edgecolor='k', linewidth=0.5)
        axes[1, 1].set(xticks=range(n), xticklabels=[f"{q:.0f}" for q in q_vals],
                       xlabel='Q_y', ylabel='y error RMS [m]',
                       title='Lat: y RMS vs Q_y')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

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
        ax.plot(ratios, metrics, 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
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
                     'R2/R1 ratio', 'Gap error RMS [m]')
        axes[0, 1].set_title('Long: gap error vs R2/R1')

    if lat_s:
        _stacked_bar(axes[1, 0], lat_s, 'u1_rms', 'u2_rms', 'R2_R1', 'y_rms', 'y RMS')
        axes[1, 0].set(title='Lat: control effort decomposition')
        _metric_line(axes[1, 1], lat_s, 'R2_R1', 'y_rms',
                     'R2/R1 ratio', 'y error RMS [m]')
        axes[1, 1].set_title('Lat: y error vs R2/R1')

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
                      save: bool = True):
    """
    2 × 3 figure:
      Row 0 (long): α vs λ  | gap_rms vs λ  | u1/u2 RMS vs α
      Row 1 (lat):  α vs λ  | y_rms   vs λ  | u1/u2 RMS vs α
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'Fixed-λ Authority Sweep{" — " + title if title else ""}',
                 fontsize=11, fontweight='bold')

    def _alpha(lam): return 1.0 / (1.0 + lam)

    def _plot_row(row, summary, metric_key, metric_label):
        lams    = [s['lambda']  for s in summary]
        alphas  = [s['alpha']   for s in summary]
        metrics = [s[metric_key] for s in summary]
        u1s     = [s['u1_rms']  for s in summary]
        u2s     = [s['u2_rms']  for s in summary]

        # α vs λ (theoretical curve)
        lam_fine  = np.logspace(np.log10(min(lams)), np.log10(max(lams)), 200)
        axes[row, 0].semilogx(lam_fine, _alpha(lam_fine), 'k--',
                              linewidth=1.0, label='α=1/(1+λ)')
        axes[row, 0].semilogx(lams, alphas, 'o', color='#1f77b4', markersize=7)
        axes[row, 0].set(xlabel='λ (log)', ylabel='α (system authority)',
                         title='Authority α vs λ', ylim=(0, 1))
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].grid(True, alpha=0.3)

        # metric vs λ
        axes[row, 1].semilogx(lams, metrics, 'o-', color='#2ca02c', linewidth=1.5,
                              markersize=6)
        axes[row, 1].set(xlabel='λ (log)', ylabel=metric_label,
                         title=f'{metric_label} vs λ')
        axes[row, 1].grid(True, alpha=0.3)

        # u1/u2 vs α
        axes[row, 2].plot(alphas, u1s, 'o-', color='#1f77b4', linewidth=1.4,
                          markersize=6, label='u1 (system)')
        axes[row, 2].plot(alphas, u2s, 's-', color='#ff7f0e', linewidth=1.4,
                          markersize=6, label='u2 (human)')
        axes[row, 2].set(xlabel='α (system authority)', ylabel='RMS',
                         title='Control effort vs α')
        axes[row, 2].legend(fontsize=7)
        axes[row, 2].grid(True, alpha=0.3)

    if lam_summary['long']:
        _plot_row(0, lam_summary['long'], 'gap_rms', 'Gap error RMS [m]')
    if lam_summary['lat']:
        _plot_row(1, lam_summary['lat'],  'y_rms',   'y error RMS [m]')

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
        plot_lambda_sweep(lam_sweep.results, lam_sweep.summary, show=show, save=save)
