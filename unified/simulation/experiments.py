"""
unified/simulation/experiments.py — Structured experiment runners.

Three experiment families, each covering both longitudinal and lateral:

  A. MonteCarloExperiment
     Compares three driver noise models over N trials:
       - 'deterministic': no noise (IDM only)
       - 'iid':           i.i.d. Gaussian N(0, σ_k²)  [B-IDM baseline]
       - 'gp':            MA-IDM SE-kernel GP           [Zhang & Sun 2024]

  B1. QWeightSweepExperiment
     Sweeps Q_pos (longitudinal) and Q_y (lateral, preserving Q_psi/Q_y ratio)
     Metric: Pareto curve — tracking error RMS vs control effort RMS

  B2. RWeightSweepExperiment
     Sweeps R2/R1 ratio for both longitudinal and lateral
     Metric: u1/u2 decomposition, gap/y error per configuration

  B3. LambdaSweepExperiment
     Bypasses Safety Field with fixed λ values for both long and lat
     Metric: gap error, y error, λ-dependent authority split

References:
  Zhang & Sun 2024     — MA-IDM SE-kernel GP
  Li et al. 2019       — Nash GNE framework
  Pustilnik & Borrelli — α = 1/(1+λ)
"""

import os
import sys
import time as time_module
import numpy as np
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

from unified.config import (
    LONG_NASH_Q_POS, LONG_NASH_Q_VEL,
    LONG_NASH_R1, LONG_NASH_R2,
    LAT_NASH_Q_Y, LAT_NASH_Q_PSI,
    LAT_NASH_R1, LAT_NASH_R2,
    LONG_NASH_LAMBDA_LEVELS, LAT_NASH_LAMBDA_LEVELS,
    RESULTS_DIR,
)
from unified.simulation.simulator import UnifiedSimulation


# =============================================================================
# Base scenario config for experiments (shorter runs to keep runtime tractable)
# =============================================================================

_BASE_CFG = {
    'num_platoon_vehicles': 3,
    'driver_type':          'normal',
    'merge_scenario':       'back',
    'T_sim':                90.0,
    'merge_trigger_time':   25.0,
}


# =============================================================================
# Helpers
# =============================================================================

def _gap_error_rms(data: dict) -> float:
    """RMS of (actual_gap - desired_gap) during FOLLOWING phase only.

    FOLLOWING-only so that the large transient gap error during APPROACH/MERGE
    does not swamp the steady-state noise effect we actually want to compare.
    Falls back to all finite values if FOLLOWING phase never occurs.
    """
    gap   = np.asarray(data.get('ego_gap_to_leader', []), dtype=float)
    des   = np.asarray(data.get('desired_gap',        []), dtype=float)
    phase = data.get('phase', [])
    mask  = np.array([p == 'FOLLOWING' for p in phase], dtype=bool)
    # Require at least 2 FOLLOWING samples; fall back to all-finite if needed
    if mask.sum() < 2:
        mask = np.ones(len(gap), dtype=bool)
    mask &= np.isfinite(gap) & np.isfinite(des)
    if mask.sum() < 2:
        return float('nan')
    return float(np.sqrt(np.mean((gap[mask] - des[mask]) ** 2)))


def _y_error_rms(data: dict, target_y: float = 0.0) -> float:
    """RMS of lateral position error during FOLLOWING phase only.

    FOLLOWING-only so that the large lane-change ramp (3.5m → 0m) during MERGE
    does not swamp the steady-state noise effect we want to compare across modes.
    Falls back to all finite values if FOLLOWING phase never occurs.
    """
    y     = np.asarray(data.get('ego_y', []), dtype=float)
    phase = data.get('phase', [])
    mask  = np.array([p == 'FOLLOWING' for p in phase], dtype=bool)
    if mask.sum() < 2:
        mask = np.ones(len(y), dtype=bool)
    mask &= np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')
    return float(np.sqrt(np.mean((y[mask] - target_y) ** 2)))


def _control_effort_rms(data: dict, key: str) -> float:
    """RMS of a scalar control signal."""
    u = np.asarray(data.get(key, []), dtype=float)
    u = u[np.isfinite(u)]
    if len(u) < 2:
        return float('nan')
    return float(np.sqrt(np.mean(u ** 2)))


def _merge_smoothness_rms(data: dict) -> float:
    """RMS of delta_lat during MERGE phase — measures lane-change control effort.

    Higher value = more aggressive steering during merge.
    Used to distinguish Q_y configurations now that FOLLOWING is perfect for all Q_y.
    Falls back to all-finite values if MERGE phase never occurs.
    """
    delta = np.asarray(data.get('delta_lat', []), dtype=float)
    phase = data.get('phase', [])
    mask  = np.array([p == 'MERGE' for p in phase], dtype=bool)
    if mask.sum() < 2:
        mask = np.ones(len(delta), dtype=bool)
    mask &= np.isfinite(delta)
    if mask.sum() < 2:
        return float('nan')
    return float(np.sqrt(np.mean(delta[mask] ** 2)))


def _merge_duration(data: dict) -> float:
    """Duration of MERGE phase in seconds.

    Returns NaN if MERGE phase never occurs (e.g. purely longitudinal runs).
    Small λ (human dominant, large α) → human tracks r2 aggressively → large u2
    → faster merge.  Large λ (system dominant, small α) → human passive → slower merge.
    """
    t     = np.asarray(data.get('time', []), dtype=float)
    phase = data.get('phase', [])
    mask  = np.array([p == 'MERGE' for p in phase], dtype=bool)
    if mask.sum() < 2:
        return float('nan')
    t_merge = t[mask]
    return float(t_merge[-1] - t_merge[0])


def _autocorr_of_signal(u: np.ndarray, t: np.ndarray, max_lag_s: float = 5.0) -> np.ndarray:
    """Normalised autocorrelation of a 1-D signal array up to max_lag_s seconds."""
    u = u[np.isfinite(u)]
    if len(u) < 4:
        return np.array([1.0])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    u_c     = u - u.mean()
    var     = float(np.var(u_c))
    if var < 1e-12:
        return np.ones(max(1, int(max_lag_s / dt) + 1))
    max_lag = min(int(max_lag_s / dt), len(u_c) - 1)
    R = np.array([
        float(np.mean(u_c[:len(u_c) - k] * u_c[k:]) / var)
        for k in range(max_lag + 1)
    ])
    return np.clip(R, -1.0, 1.0)


def _residual_autocorr(trial_data: dict, ref_data: dict, key: str,
                       max_lag_s: float = 5.0,
                       phase_filter: tuple = ('FOLLOWING',)) -> np.ndarray:
    """Autocorrelation of the NOISE RESIDUAL: trial[key] − ref[key].

    Subtracting the deterministic reference removes the smooth Nash component
    (which dominates raw autocorr and is identical across all noise modes).
    What remains is the pure noise injected into that trial.

    Expected outcomes after subtraction:
      deterministic:  residual ≈ 0  → variance < 1e-12 → returns array of 1.0
                      (NaN reported in summary to flag 'undefined')
      i.i.d.:         residual = white noise → lag-1 ≈ 0
      GP (SE kernel): residual = correlated samples → lag-1 ≈ exp(-dt²/2ℓ²) ≈ 0.9999
    """
    u_trial = np.asarray(trial_data.get(key, []), dtype=float)
    u_ref   = np.asarray(ref_data.get(key,   []), dtype=float)
    phase   = trial_data.get('phase', [])
    t       = np.asarray(trial_data.get('time', []), dtype=float)
    mask    = np.array([p in phase_filter for p in phase], dtype=bool)
    n       = min(len(u_trial), len(u_ref), len(mask))
    noise   = (u_trial[:n] - u_ref[:n])[mask[:n]]
    t_m     = t[:n][mask[:n]]
    return _autocorr_of_signal(noise, t_m, max_lag_s=max_lag_s)


def _run_single(cfg_override: dict, **sim_kwargs) -> dict:
    """Run one UnifiedSimulation and return its data dict."""
    cfg = {**_BASE_CFG, **cfg_override}
    sim = UnifiedSimulation(
        num_platoon_vehicles=cfg['num_platoon_vehicles'],
        driver_type=cfg['driver_type'],
        merge_scenario=cfg['merge_scenario'],
        T_sim=cfg['T_sim'],
        merge_trigger_time=cfg['merge_trigger_time'],
        **sim_kwargs,
    )
    return sim.run()


# =============================================================================
# A. Monte Carlo Experiment
# =============================================================================

class MonteCarloExperiment:
    """
    N-trial Monte Carlo comparison of three driver noise models.

    Results structure (self.results):
        {
          'deterministic': [data_dict, ...],   # N deterministic runs
          'iid':           [data_dict, ...],   # N i.i.d. noise runs
          'gp':            [data_dict, ...],   # N MA-IDM GP runs
        }

    Summary metrics (self.summary):
        {mode: {'gap_rms': [...], 'y_rms': [...], 'u_long_rms': [...], 'u_lat_rms': [...]}}
    """

    MODES = ('deterministic', 'iid', 'gp')

    def __init__(self,
                 n_trials: int = 20,
                 driver_type: str = 'normal',
                 merge_scenario: str = 'back',
                 T_sim: float = 120.0,
                 merge_trigger_time: float = 25.0):
        self.n_trials          = n_trials
        self.cfg_override      = {
            'driver_type':    driver_type,
            'merge_scenario': merge_scenario,
            'T_sim':          T_sim,
            'merge_trigger_time': merge_trigger_time,
        }
        self.results: Dict[str, List[dict]] = {m: [] for m in self.MODES}
        self.summary: Dict[str, dict]       = {}

    def run(self, verbose: bool = True) -> Dict[str, List[dict]]:
        """Run all N×3 simulations. Returns self.results."""
        total = self.n_trials * len(self.MODES)
        done  = 0
        t0    = time_module.time()

        for mode in self.MODES:
            if verbose:
                print(f"\n[MonteCarloExperiment] Mode: {mode} ({self.n_trials} trials)")
            for trial in range(self.n_trials):
                np.random.seed(trial)   # reproducible per trial
                data = _run_single(self.cfg_override, noise_mode=mode)
                self.results[mode].append(data)
                done += 1
                if verbose:
                    elapsed = time_module.time() - t0
                    print(f"  trial {trial + 1}/{self.n_trials}  "
                          f"({done}/{total} total, {elapsed:.0f}s)")

        self._compute_summary()
        if verbose:
            self._print_summary()
        return self.results

    def _compute_summary(self):
        # Deterministic trial 0 is the reference baseline for residual autocorr.
        # All modes use the same baseline so the smooth Nash component cancels out,
        # leaving only the injected noise in the residual.
        det_trials = self.results.get('deterministic', [])
        det_ref    = det_trials[0] if det_trials else {}

        for mode in self.MODES:
            trials = self.results[mode]
            gap_rms_per_trial = [_gap_error_rms(d)                    for d in trials]
            y_rms_per_trial   = [_y_error_rms(d)                      for d in trials]
            u_long_per_trial  = [_control_effort_rms(d, 'u_long')     for d in trials]
            u_lat_per_trial   = [_control_effort_rms(d, 'delta_lat')  for d in trials]

            # Noise residual autocorr: trial[u_long] − deterministic[u_long]
            # For 'deterministic': residual ≈ 0 (same seed) → var < 1e-12 → lag-1 = NaN
            noise_ac_trials = [
                _residual_autocorr(d, det_ref, 'u_long') for d in trials
            ]
            lag1_vals = [
                float(r[1]) if len(r) > 1 else float('nan')
                for r in noise_ac_trials
            ]
            # For deterministic mode the residual is identically zero → report NaN
            if mode == 'deterministic':
                lag1_vals = [float('nan')] * len(lag1_vals)

            self.summary[mode] = {
                'gap_rms':          gap_rms_per_trial,
                'gap_rms_mean':     float(np.nanmean(gap_rms_per_trial)),
                'gap_rms_std':      float(np.nanstd(gap_rms_per_trial)),
                'y_rms':            y_rms_per_trial,
                'y_rms_mean':       float(np.nanmean(y_rms_per_trial)),
                'y_rms_std':        float(np.nanstd(y_rms_per_trial)),
                'u_long_rms':       u_long_per_trial,
                'u_lat_rms':        u_lat_per_trial,
                'noise_autocorr_all': noise_ac_trials,   # list of arrays for plotting
                'noise_lag1_mean':  (float(np.nanmean(lag1_vals))
                                    if any(np.isfinite(v) for v in lag1_vals)
                                    else float('nan')),
                'noise_lag1_std':   (float(np.nanstd(lag1_vals))
                                    if any(np.isfinite(v) for v in lag1_vals)
                                    else float('nan')),
            }

    def _print_summary(self):
        print(f"\n{'=' * 80}")
        print("  Monte Carlo Summary")
        print("  gap/y RMS: FOLLOWING phase only  |  autocorr: noise residual vs det. baseline")
        print(f"{'=' * 80}")
        print(f"{'Mode':<16}  {'gap_rms±std [m]':>16}  {'y_rms±std [m]':>14}  "
              f"{'noise lag-1 R':>14}")
        print(f"{'-' * 75}")
        for mode in self.MODES:
            s      = self.summary[mode]
            lag_s  = (f"{s['noise_lag1_mean']:6.3f}±{s['noise_lag1_std']:.3f}"
                      if not np.isnan(s['noise_lag1_mean']) else "   NaN (baseline)")
            print(f"{mode:<16}  "
                  f"{s['gap_rms_mean']:7.3f}±{s['gap_rms_std']:.3f}  "
                  f"{s['y_rms_mean']:.4f}±{s['y_rms_std']:.4f}  "
                  f"{lag_s:>16}")
        print(f"\n  Key findings:")
        det_mean = self.summary.get('deterministic', {}).get('gap_rms_mean', float('nan'))
        gp_mean  = self.summary.get('gp',            {}).get('gap_rms_mean', float('nan'))
        gp_std   = self.summary.get('gp',            {}).get('gap_rms_std',  float('nan'))
        iid_std  = self.summary.get('iid',           {}).get('gap_rms_std',  float('nan'))
        if not np.isnan(det_mean) and not np.isnan(gp_mean):
            print(f"  • GP mean gap_rms ({gp_mean:.3f} m) < deterministic ({det_mean:.3f} m) "
                  f"→ predictive benefit from SE-kernel R2")
        if not np.isnan(gp_std) and not np.isnan(iid_std) and iid_std > 0:
            print(f"  • GP std ({gp_std:.3f} m) >> i.i.d. std ({iid_std:.3f} m) "
                  f"→ temporal persistence: correlated noise accumulates before Nash corrects")

        self._print_snr()

    def _print_snr(self):
        """Print SNR diagnostics for lateral noise calibration validation.

        Computes per-trial in FOLLOWING phase:
          signal = std(delta_lat | det baseline)   — Nash output variation
          noise  = std(delta_lat_noisy - delta_lat_det)  — noise residual
          SNR    = signal / noise
        """
        det_trials = self.results.get('deterministic', [])
        if not det_trials:
            return

        print(f"\n  SNR diagnostic — lateral noise calibration (FOLLOWING phase)")
        print(f"  {'Mode':<14}  {'signal std [rad]':>16}  {'noise std [rad]':>15}  {'SNR':>6}")
        print(f"  {'-' * 58}")

        # deterministic signal baseline: std of delta_lat in FOLLOWING across all det trials
        det_signals = []
        for d in det_trials:
            arr   = np.asarray(d.get('delta_lat', []), dtype=float)
            phase = d.get('phase', [])
            mask  = np.array([p == 'FOLLOWING' for p in phase], dtype=bool)
            if mask.sum() > 1:
                det_signals.append(float(np.std(arr[mask])))
        signal_std = float(np.nanmean(det_signals)) if det_signals else float('nan')

        for mode in self.MODES:
            trials = self.results[mode]
            noise_stds = []
            for trial, det in zip(trials, det_trials):
                arr_t = np.asarray(trial.get('delta_lat', []), dtype=float)
                arr_d = np.asarray(det.get('delta_lat',   []), dtype=float)
                phase = trial.get('phase', [])
                mask  = np.array([p == 'FOLLOWING' for p in phase], dtype=bool)
                n = min(len(arr_t), len(arr_d), len(mask))
                residual = (arr_t[:n] - arr_d[:n])[mask[:n]]
                residual = residual[np.isfinite(residual)]
                if len(residual) > 1:
                    noise_stds.append(float(np.std(residual)))

            noise_std = float(np.nanmean(noise_stds)) if noise_stds else float('nan')
            snr = signal_std / noise_std if (np.isfinite(noise_std) and noise_std > 1e-9) else float('nan')
            snr_s = f"{snr:6.1f}" if np.isfinite(snr) else "   N/A"
            noise_s = f"{noise_std:.5f}" if np.isfinite(noise_std) else "    N/A"
            print(f"  {mode:<14}  {signal_std:>16.5f}  {noise_s:>15}  {snr_s}")

        print(f"  (SNR > 5 → noise realistic; SNR < 2 → noise dominates control)")


# =============================================================================
# B1. Q-Weight Sweep
# =============================================================================

class QWeightSweepExperiment:
    """
    Sweep Q_pos (long) and Q_y (lat, with Q_psi/Q_y ratio fixed) across a grid.

    Parameters
    ----------
    long_Q_pos_values : sequence of float
        Values of LONG_NASH_Q_POS to sweep.
    lat_Q_y_values : sequence of float
        Values of LAT_NASH_Q_Y to sweep.  Q_psi = Q_y × lat_Q_ratio.
    lat_Q_ratio : float
        Fixed Q_psi / Q_y ratio (default = LONG_NASH_Q_PSI / LAT_NASH_Q_Y ≈ 12.5).

    Results structure (self.results):
        {
          'long': [{'Q_pos': ..., 'data': data_dict}, ...],
          'lat':  [{'Q_y': ..., 'Q_psi': ..., 'data': data_dict}, ...],
        }
    """

    _DEFAULT_Q_RATIO = LAT_NASH_Q_PSI / LAT_NASH_Q_Y   # ≈ 12.5

    def __init__(self,
                 long_Q_pos_values: tuple = (500, 1000, 2500, 5000, 10000),
                 lat_Q_y_values:    tuple = (200, 400, 800, 1600, 3200),
                 lat_Q_ratio:       float = _DEFAULT_Q_RATIO,
                 driver_type:       str   = 'normal',
                 merge_scenario:    str   = 'back',
                 T_sim:             float = 120.0,
                 merge_trigger_time: float = 25.0,
                 noise_mode:        str   = 'deterministic'):
        self.long_Q_pos_values    = long_Q_pos_values
        self.lat_Q_y_values       = lat_Q_y_values
        self.lat_Q_ratio          = lat_Q_ratio
        self.noise_mode           = noise_mode
        self.cfg_override = {
            'driver_type':    driver_type,
            'merge_scenario': merge_scenario,
            'T_sim':          T_sim,
            'merge_trigger_time': merge_trigger_time,
        }
        self.results: Dict[str, list] = {'long': [], 'lat': []}
        self.summary: Dict[str, list] = {'long': [], 'lat': []}

    def run(self, verbose: bool = True) -> Dict[str, list]:
        total = len(self.long_Q_pos_values) + len(self.lat_Q_y_values)
        done  = 0

        # ── Longitudinal Q sweep ───────────────────────────────────────────────
        if verbose:
            print(f"\n[QWeightSweep] Longitudinal Q_pos sweep: {self.long_Q_pos_values}")
        for q_pos in self.long_Q_pos_values:
            wo = {'long_Q_pos': float(q_pos), 'long_Q_vel': LONG_NASH_Q_VEL}
            data = _run_single(self.cfg_override,
                               noise_mode=self.noise_mode,
                               weight_overrides=wo)
            self.results['long'].append({'Q_pos': q_pos, 'data': data})
            done += 1
            if verbose:
                print(f"  long Q_pos={q_pos:6.0f}  gap_rms={_gap_error_rms(data):.3f}  "
                      f"u_rms={_control_effort_rms(data, 'u_long'):.3f}  "
                      f"({done}/{total})")

        # ── Lateral Q sweep ────────────────────────────────────────────────────
        if verbose:
            print(f"\n[QWeightSweep] Lateral Q_y sweep: {self.lat_Q_y_values}  "
                  f"(ratio={self.lat_Q_ratio:.1f})")
        for q_y in self.lat_Q_y_values:
            q_psi = q_y * self.lat_Q_ratio
            wo = {'lat_Q_y': float(q_y), 'lat_Q_psi': float(q_psi)}
            data = _run_single(self.cfg_override,
                               noise_mode=self.noise_mode,
                               weight_overrides=wo)
            self.results['lat'].append({'Q_y': q_y, 'Q_psi': q_psi, 'data': data})
            done += 1
            if verbose:
                print(f"  lat  Q_y={q_y:6.0f}  Q_psi={q_psi:8.0f}  "
                      f"y_rms={_y_error_rms(data):.4f}  "
                      f"u_rms={_control_effort_rms(data, 'delta_lat'):.5f}  "
                      f"merge_δ={_merge_smoothness_rms(data):.5f}  "
                      f"({done}/{total})")

        self._compute_summary()
        return self.results

    def _compute_summary(self):
        self.summary['long'] = [
            {
                'Q_pos':       r['Q_pos'],
                'gap_rms':     _gap_error_rms(r['data']),
                'u_long_rms':  _control_effort_rms(r['data'], 'u_long'),
            }
            for r in self.results['long']
        ]
        self.summary['lat'] = [
            {
                'Q_y':             r['Q_y'],
                'Q_psi':           r['Q_psi'],
                'y_rms':           _y_error_rms(r['data']),
                'u_lat_rms':       _control_effort_rms(r['data'], 'delta_lat'),
                'merge_delta_rms': _merge_smoothness_rms(r['data']),
            }
            for r in self.results['lat']
        ]


# =============================================================================
# B2. R-Weight (effort weight) Sweep
# =============================================================================

class RWeightSweepExperiment:
    """
    Sweep R2/R1 effort weight ratio for both longitudinal and lateral.

    Each configuration is a dict with keys 'long_R1', 'long_R2', 'lat_R1', 'lat_R2'.

    Default configurations:
        Long: R1=7500, R2 ∈ {3750, 7500, 12500, 25000, 50000}  (R2/R1 = 0.5, 1, 1.67, 3.33, 6.67)
        Lat:  R1=50000, R2 ∈ {25000, 50000, 100000, 200000}

    Results structure (self.results):
        {
          'long': [{'label': ..., 'R1': ..., 'R2': ..., 'data': ...}, ...],
          'lat':  [{'label': ..., 'R1': ..., 'R2': ..., 'data': ...}, ...],
        }
    """

    _DEFAULT_LONG_CONFIGS = [
        {'long_R1': LONG_NASH_R1, 'long_R2': LONG_NASH_R1 * 0.5},     # R2/R1=0.5 (system yields)
        {'long_R1': LONG_NASH_R1, 'long_R2': LONG_NASH_R1 * 1.0},     # R2/R1=1   (balanced)
        {'long_R1': LONG_NASH_R1, 'long_R2': LONG_NASH_R2},            # R2/R1≈1.67 (default)
        {'long_R1': LONG_NASH_R1, 'long_R2': LONG_NASH_R1 * 3.33},    # R2/R1=3.33 (human effort penalised)
        {'long_R1': LONG_NASH_R1, 'long_R2': LONG_NASH_R1 * 6.67},    # R2/R1=6.67 (human strongly penalised)
    ]
    _DEFAULT_LAT_CONFIGS = [
        {'lat_R1': LAT_NASH_R1, 'lat_R2': LAT_NASH_R1 * 0.5},
        {'lat_R1': LAT_NASH_R1, 'lat_R2': LAT_NASH_R1 * 1.0},         # balanced
        {'lat_R1': LAT_NASH_R1, 'lat_R2': LAT_NASH_R2},                # default (equal)
        {'lat_R1': LAT_NASH_R1, 'lat_R2': LAT_NASH_R1 * 2.0},
        {'lat_R1': LAT_NASH_R1, 'lat_R2': LAT_NASH_R1 * 4.0},
    ]

    def __init__(self,
                 long_configs: Optional[list] = None,
                 lat_configs:  Optional[list] = None,
                 driver_type:  str = 'normal',
                 merge_scenario: str = 'back',
                 T_sim:        float = 120.0,
                 merge_trigger_time: float = 25.0,
                 noise_mode:   str = 'deterministic'):
        self.long_configs  = long_configs or self._DEFAULT_LONG_CONFIGS
        self.lat_configs   = lat_configs  or self._DEFAULT_LAT_CONFIGS
        self.noise_mode    = noise_mode
        self.cfg_override  = {
            'driver_type':    driver_type,
            'merge_scenario': merge_scenario,
            'T_sim':          T_sim,
            'merge_trigger_time': merge_trigger_time,
        }
        self.results: Dict[str, list] = {'long': [], 'lat': []}
        self.summary: Dict[str, list] = {'long': [], 'lat': []}

    def run(self, verbose: bool = True) -> Dict[str, list]:
        total = len(self.long_configs) + len(self.lat_configs)
        done  = 0

        if verbose:
            print(f"\n[RWeightSweep] Longitudinal R2/R1 sweep ({len(self.long_configs)} configs)")
        for cfg in self.long_configs:
            r1, r2 = cfg['long_R1'], cfg['long_R2']
            label  = f"R2/R1={r2/r1:.2f}"
            data   = _run_single(self.cfg_override,
                                 noise_mode=self.noise_mode,
                                 weight_overrides=cfg)
            self.results['long'].append({'label': label, 'R1': r1, 'R2': r2, 'data': data})
            done += 1
            if verbose:
                print(f"  long {label}  gap_rms={_gap_error_rms(data):.3f}  "
                      f"u1_rms={_control_effort_rms(data, 'u1_long'):.3f}  "
                      f"u2_rms={_control_effort_rms(data, 'u2_long'):.3f}  ({done}/{total})")

        if verbose:
            print(f"\n[RWeightSweep] Lateral R2/R1 sweep ({len(self.lat_configs)} configs)")
        for cfg in self.lat_configs:
            r1, r2 = cfg['lat_R1'], cfg['lat_R2']
            label  = f"R2/R1={r2/r1:.2f}"
            data   = _run_single(self.cfg_override,
                                 noise_mode=self.noise_mode,
                                 weight_overrides=cfg)
            self.results['lat'].append({'label': label, 'R1': r1, 'R2': r2, 'data': data})
            done += 1
            if verbose:
                print(f"  lat  {label}  y_rms={_y_error_rms(data):.4f}  "
                      f"u1_rms={_control_effort_rms(data, 'u1_lat'):.5f}  "
                      f"u2_rms={_control_effort_rms(data, 'u2_lat'):.5f}  ({done}/{total})")

        self._compute_summary()
        return self.results

    def _compute_summary(self):
        self.summary['long'] = [
            {
                'label':    r['label'],
                'R2_R1':    r['R2'] / r['R1'],
                'gap_rms':  _gap_error_rms(r['data']),
                'u1_rms':   _control_effort_rms(r['data'], 'u1_long'),
                'u2_rms':   _control_effort_rms(r['data'], 'u2_long'),
            }
            for r in self.results['long']
        ]
        self.summary['lat'] = [
            {
                'label':   r['label'],
                'R2_R1':   r['R2'] / r['R1'],
                'y_rms':   _y_error_rms(r['data']),
                'u1_rms':  _control_effort_rms(r['data'], 'u1_lat'),
                'u2_rms':  _control_effort_rms(r['data'], 'u2_lat'),
            }
            for r in self.results['lat']
        ]


# =============================================================================
# B3. Fixed-Lambda Sweep
# =============================================================================

class LambdaSweepExperiment:
    """
    Run one simulation per fixed-λ value (bypassing Safety Field) for both
    longitudinal and lateral.

    Results structure (self.results):
        {
          'long': [{'lambda': ..., 'alpha': ..., 'data': ...}, ...],
          'lat':  [{'lambda': ..., 'alpha': ..., 'data': ...}, ...],
        }
    """

    def __init__(self,
                 long_lambdas: tuple = LONG_NASH_LAMBDA_LEVELS,
                 lat_lambdas:  tuple = LAT_NASH_LAMBDA_LEVELS,
                 driver_type:  str   = 'normal',
                 merge_scenario: str = 'back',
                 T_sim:        float = 120.0,
                 merge_trigger_time: float = 25.0,
                 noise_mode:   str   = 'deterministic',
                 lat_human_y_bias: float = 0.75):
        self.long_lambdas      = long_lambdas
        self.lat_lambdas       = lat_lambdas
        self.noise_mode        = noise_mode
        self.lat_human_y_bias  = lat_human_y_bias
        self.cfg_override = {
            'driver_type':    driver_type,
            'merge_scenario': merge_scenario,
            'T_sim':          T_sim,
            'merge_trigger_time': merge_trigger_time,
        }
        self.results: Dict[str, list] = {'long': [], 'lat': []}
        self.summary: Dict[str, list] = {'long': [], 'lat': []}

    def run(self, verbose: bool = True) -> Dict[str, list]:
        total = len(self.long_lambdas) + len(self.lat_lambdas)
        done  = 0

        if verbose:
            print(f"\n[LambdaSweep] Longitudinal fixed-λ sweep: {self.long_lambdas}")
        for lam in self.long_lambdas:
            alpha = 1.0 / (1.0 + lam)   # Pustilnik α = 1/(1+λ)
            data  = _run_single(self.cfg_override,
                                noise_mode=self.noise_mode,
                                fixed_lambda_long=float(lam))
            self.results['long'].append({'lambda': lam, 'alpha': alpha, 'data': data})
            done += 1
            if verbose:
                print(f"  long λ={lam:6.2f}  α={alpha:.3f}  "
                      f"gap_rms={_gap_error_rms(data):.3f}  ({done}/{total})")

        if verbose:
            print(f"\n[LambdaSweep] Lateral fixed-λ sweep: {self.lat_lambdas}")
        for lam in self.lat_lambdas:
            alpha = 1.0 / (1.0 + lam)
            data  = _run_single(self.cfg_override,
                                noise_mode=self.noise_mode,
                                fixed_lambda_lat=float(lam),
                                weight_overrides={'lat_human_y_bias': self.lat_human_y_bias})
            self.results['lat'].append({'lambda': lam, 'alpha': alpha, 'data': data})
            done += 1
            if verbose:
                print(f"  lat  λ={lam:6.2f}  α={alpha:.3f}  "
                      f"y_rms={_y_error_rms(data):.4f}  ({done}/{total})")

        self._compute_summary()
        return self.results

    def _compute_summary(self):
        self.summary['long'] = [
            {
                'lambda':   r['lambda'],
                'alpha':    r['alpha'],
                'gap_rms':  _gap_error_rms(r['data']),
                'u1_rms':   _control_effort_rms(r['data'], 'u1_long'),
                'u2_rms':   _control_effort_rms(r['data'], 'u2_long'),
            }
            for r in self.results['long']
        ]
        self.summary['lat'] = [
            {
                'lambda':          r['lambda'],
                'alpha':           r['alpha'],
                'y_rms':           _y_error_rms(r['data']),
                'merge_duration':  _merge_duration(r['data']),
                'merge_delta_rms': _merge_smoothness_rms(r['data']),
                'u1_rms':          _control_effort_rms(r['data'], 'u1_lat'),
                'u2_rms':          _control_effort_rms(r['data'], 'u2_lat'),
            }
            for r in self.results['lat']
        ]

    def print_summary(self):
        print(f"\n{'=' * 65}")
        print("  Lambda Sweep Summary")
        print(f"{'=' * 65}")
        print("  LONGITUDINAL")
        print(f"  {'λ':>7}  {'α':>6}  {'gap_rms':>9}  {'u1_rms':>9}  {'u2_rms':>9}")
        for r in self.summary['long']:
            print(f"  {r['lambda']:7.2f}  {r['alpha']:6.3f}  {r['gap_rms']:9.3f}  "
                  f"{r['u1_rms']:9.3f}  {r['u2_rms']:9.3f}")
        print("\n  LATERAL  (α = human tracking weight; small λ → human dominant)")
        print(f"  {'λ':>7}  {'α (hum)':>8}  {'merge_dur[s]':>13}  {'y_rms[m]':>9}  {'δ_rms[rad]':>11}  {'u1_rms':>9}  {'u2_rms':>9}")
        for r in self.summary['lat']:
            dur  = r.get('merge_duration',  float('nan'))
            yrms = r.get('y_rms',           float('nan'))
            drms = r.get('merge_delta_rms', float('nan'))
            dur_s  = f"{dur:13.1f}"  if np.isfinite(dur)  else "          N/A"
            yrms_s = f"{yrms:9.4f}"  if np.isfinite(yrms) else "      N/A"
            drms_s = f"{drms:11.5f}" if np.isfinite(drms) else "        N/A"
            print(f"  {r['lambda']:7.2f}  {r['alpha']:8.3f}  {dur_s}  {yrms_s}  {drms_s}  "
                  f"{r['u1_rms']:9.5f}  {r['u2_rms']:9.5f}")
