"""
Lane-change comfort evaluation for Nash shared-control lateral simulation.

Metrics (all computed over the 5%–95% LC window):
  1. LC duration         [s]    Toledo & Zohar 2007; Lee et al. 2004
  2. Peak lateral ay     [m/s²] ISO 2631-1 Table 2
  3. RMS  lateral ay     [m/s²] ISO 2631-1 §6.2
  4. Peak lateral jerk   [m/s³] numerical derivative of ay
  5. Peak body lateral v [m/s]  human_y_dot
  6. Authority Disruption Index (ADI) — Nash-specific
     ADI = mean(|d(lambda)/dt|) / AUTHORITY_LAMBDA_MAX

Pure function — no side effects, numpy only.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LANE_WIDTH, AUTHORITY_LAMBDA_MAX,
    COMFORT_LC_DURATION_GOOD, COMFORT_LC_DURATION_ACCEPTABLE,
    COMFORT_AY_PEAK_GOOD, COMFORT_AY_PEAK_ACCEPTABLE,
    COMFORT_AY_RMS_GOOD, COMFORT_AY_RMS_ACCEPTABLE,
    COMFORT_JERK_GOOD, COMFORT_JERK_ACCEPTABLE,
    COMFORT_LATERAL_VEL_GOOD, COMFORT_LATERAL_VEL_ACCEPTABLE,
    COMFORT_ADI_GOOD, COMFORT_ADI_ACCEPTABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(value: float, good: float, acceptable: float):
    """Return (int score 0-2, str verdict) for a lower-is-better metric."""
    if value <= good:
        return 2, 'Good'
    elif value <= acceptable:
        return 1, 'Acceptable'
    return 0, 'Poor'


def _lc_window(y: np.ndarray, lane_width: float):
    """
    Detect the 5%–95% lateral-displacement window of the lane change.

    The vehicle starts at y[0] and ends near y[-1].  We find the first
    sample where lateral travel >= 5% of total travel (start of active
    maneuver) and the first where it >= 95% (end of active maneuver).

    Falls back to (0, len(y)-1) when total travel < 0.1 m (no LC).
    """
    n = len(y)
    y_start = float(y[0])
    y_end   = float(y[-1])
    travel  = abs(y_end - y_start)

    if travel < 0.1:
        return 0, n - 1

    direction = np.sign(y_end - y_start)
    th5  = y_start + direction * 0.05 * travel
    th95 = y_start + direction * 0.95 * travel

    if direction > 0:
        idx5  = np.where(y >= th5)[0]
        idx95 = np.where(y >= th95)[0]
    else:
        idx5  = np.where(y <= th5)[0]
        idx95 = np.where(y <= th95)[0]

    i0 = int(idx5[0])  if len(idx5)  > 0 else 0
    i1 = int(idx95[0]) if len(idx95) > 0 else n - 1
    if i1 <= i0:
        i1 = min(i0 + 1, n - 1)
    return i0, i1


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_lane_change_comfort(data: dict, lane_width: float = LANE_WIDTH) -> dict:
    """
    Multi-dimensional comfort evaluation over the lane-change window.

    Parameters
    ----------
    data : dict
        Simulation output dict with numpy arrays.  Required keys:
          'time', 'human_y', 'human_ay', 'human_y_dot', 'authority_ratio'
    lane_width : float
        Lane width in metres.  Default = LANE_WIDTH from config.py.

    Returns
    -------
    dict
        lc_window     : (t_start, t_end, i_start, i_end)
        metrics       : dict of raw float values
        scores        : dict mapping metric name -> (int 0-2, str verdict)
        overall_score : float in [0, 100]
        overall_label : 'Comfortable' | 'Acceptable' | 'Uncomfortable'
    """
    t   = np.asarray(data['time'])
    y   = np.asarray(data['human_y'])
    ay  = np.asarray(data['human_ay'])
    vy  = np.asarray(data['human_y_dot'])
    lam = np.asarray(data['authority_ratio'], dtype=float)

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.01
    i0, i1 = _lc_window(y, lane_width)

    sl    = slice(i0, i1 + 1)
    ay_w  = ay[sl]
    vy_w  = vy[sl]
    lam_w = lam[sl]

    if len(ay_w) < 2:   # guard: fall back to full signal
        ay_w = ay; vy_w = vy; lam_w = lam

    # ── Metrics ─────────────────────────────────────────────────────────
    lc_dur    = float(t[i1] - t[i0])
    peak_ay   = float(np.max(np.abs(ay_w)))
    rms_ay    = float(np.sqrt(np.mean(ay_w ** 2)))
    peak_jerk = float(np.max(np.abs(np.gradient(ay_w, dt))))
    peak_vy   = float(np.max(np.abs(vy_w)))
    adi       = float(
        np.mean(np.abs(np.gradient(lam_w, dt))) / max(AUTHORITY_LAMBDA_MAX, 1e-9)
    )

    # ── Scores (each metric: 0=Poor, 1=Acceptable, 2=Good) ──────────────
    s_dur,  v_dur  = _score(lc_dur,    COMFORT_LC_DURATION_GOOD,    COMFORT_LC_DURATION_ACCEPTABLE)
    s_pay,  v_pay  = _score(peak_ay,   COMFORT_AY_PEAK_GOOD,        COMFORT_AY_PEAK_ACCEPTABLE)
    s_rms,  v_rms  = _score(rms_ay,    COMFORT_AY_RMS_GOOD,         COMFORT_AY_RMS_ACCEPTABLE)
    s_jerk, v_jerk = _score(peak_jerk, COMFORT_JERK_GOOD,           COMFORT_JERK_ACCEPTABLE)
    s_vy,   v_vy   = _score(peak_vy,   COMFORT_LATERAL_VEL_GOOD,    COMFORT_LATERAL_VEL_ACCEPTABLE)
    s_adi,  v_adi  = _score(adi,       COMFORT_ADI_GOOD,            COMFORT_ADI_ACCEPTABLE)

    total = s_dur + s_pay + s_rms + s_jerk + s_vy + s_adi
    pct   = round(total / 12.0 * 100.0, 1)   # 6 metrics × 2 pts max = 12

    if pct >= 75.0:
        label = 'Comfortable'
    elif pct >= 50.0:
        label = 'Acceptable'
    else:
        label = 'Uncomfortable'

    return {
        'lc_window': (float(t[i0]), float(t[i1]), i0, i1),
        'metrics': {
            'lc_duration_s': lc_dur,
            'peak_ay_ms2':   peak_ay,
            'rms_ay_ms2':    rms_ay,
            'peak_jerk_ms3': peak_jerk,
            'peak_vy_ms':    peak_vy,
            'adi':           adi,
        },
        'scores': {
            'lc_duration': (s_dur,  v_dur),
            'peak_ay':     (s_pay,  v_pay),
            'rms_ay':      (s_rms,  v_rms),
            'peak_jerk':   (s_jerk, v_jerk),
            'peak_vy':     (s_vy,   v_vy),
            'adi':         (s_adi,  v_adi),
        },
        'overall_score': pct,
        'overall_label': label,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text, replacing any character that can't be encoded by the terminal."""
    import sys
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(
            sys.stdout.encoding or 'utf-8'))


# Verdict symbols (emoji with ASCII fallback handled by _safe_print)
def _sym(verdict: str) -> str:
    return {
        'Good':       '\u2705',   # ✅
        'Acceptable': '\U0001f7e1',  # 🟡
        'Poor':       '\u274c',   # ❌
    }.get(verdict, '?')


def print_comfort_report(report: dict) -> None:
    """Pretty-print the comfort evaluation report to stdout."""
    t0, t1, _, _ = report['lc_window']
    m = report['metrics']
    s = report['scores']

    _safe_print(f"   LC window: {t0:.1f}s \u2192 {t1:.1f}s  (duration {m['lc_duration_s']:.1f}s)")
    _safe_print(f"   {'Metric':<30} {'Value':>12}   Verdict")
    _safe_print(f"   {'-'*54}")
    rows = [
        ("LC Duration",             f"{m['lc_duration_s']:.2f} s",      s['lc_duration']),
        ("Peak Lateral Accel",      f"{m['peak_ay_ms2']:.3f} m/s\u00b2", s['peak_ay']),
        ("RMS  Lateral Accel",      f"{m['rms_ay_ms2']:.3f} m/s\u00b2",  s['rms_ay']),
        ("Peak Lateral Jerk",       f"{m['peak_jerk_ms3']:.3f} m/s\u00b3", s['peak_jerk']),
        ("Peak Body Lateral Vel",   f"{m['peak_vy_ms']:.3f} m/s",        s['peak_vy']),
        ("Authority Disrupt Index", f"{m['adi']:.4f}",                    s['adi']),
    ]
    for name, val, (_, verdict) in rows:
        _safe_print(f"   [{_sym(verdict)}] {name:<30} {val:>13}   {verdict}")
    _safe_print(f"   {'-'*54}")
    _safe_print(f"   Overall: {report['overall_score']:.1f}%  \u2014  {report['overall_label']}")
