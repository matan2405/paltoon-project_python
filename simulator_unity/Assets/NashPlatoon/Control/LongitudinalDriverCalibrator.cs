// LongitudinalDriverCalibrator — Online IDM time-headway (T) estimation.
//
// Two calibration modes feeding the same circular buffer:
//
//   GAP_SEARCH / MERGE — IDM inversion:
//     T = [gap×√(freeTerm - a/aMax) - s0 - vx×dv/(2√(aMax×β))] / vx
//     Valid only when |dv| < 2 m/s. Breaks down at vx ≈ v0 (freeTerm → 0).
//
//   FOLLOWING — direct headway:
//     T = (gap - s0) / vx
//     Uses the actual gap the driver is maintaining right now as the IDM attractor.
//     This is valid because HumanRef is computed independently of Nash — R2 sees
//     only driver inputs and real sensor data (gap, vx), so there is no circular
//     dependency with the Nash output. T here means: "the driver is holding this
//     gap, predict that they will continue to hold it."
//     Filters: |dv| < 1 m/s (speed-matched), vx > 5 m/s.
//     No ax filter — the driver may press pedals; what matters is the gap they
//     choose to maintain, not whether they are momentarily accelerating.
//
// The circular buffer runs continuously. In Following, steady-state samples
// quickly evict the noisier Merge inversion samples (window=20, ~2s to refill).
//
// Fallback: until IsReady, T = fallbackT (Zhang 2024 population mean = 1.5 s).
//
// aMax fixed at 0.553 m/s² (Zhang & Sun 2024). Not calibrated from HITL.

using UnityEngine;

public class LongitudinalDriverCalibrator
{
    // ── Fixed IDM parameters (not calibrated) ────────────────────────────────
    readonly float _s0;        // jam spacing [m]
    readonly float _aMax;      // fixed comfortable acceleration [m/s²] (Zhang 2024 pop. mean)
    readonly float _beta;      // comfortable deceleration [m/s²]
    readonly float _delta;     // free-road exponent (default 4)
    readonly float _fallbackT; // T used if calibration insufficient
    readonly float _v0;        // target velocity [m/s] (from constructor)

    // ── Calibration bounds ────────────────────────────────────────────────────
    const float T_MIN = 0.5f;
    const float T_MAX = 3.0f;

    // ── T running-mean window (GapSearch + Merge) ────────────────────────────
    // Window of 20 samples. At 10 Hz Nash rate with |dv|<2 + rhs>0 filters,
    // empirically ~50 valid samples arrive across GapSearch+Merge (~30s),
    // but only ~22 pass all filters before Following — 20 is achievable and
    // still averages out transients (std≈0.4s → SE≈0.09s at n=20).
    // IsReady triggers only when the window is full → 20 valid samples.
    const int WINDOW_T = 20;

    // ── State ─────────────────────────────────────────────────────────────────
    float   _tEst;

    // Circular buffer for T running mean
    float[] _tWindow;
    int     _tWindowHead;   // next write index
    int     _tWindowCount;  // valid entries so far (≤ WINDOW_T)
    float   _tWindowSum;    // sum of entries in buffer

    // ── Public outputs ────────────────────────────────────────────────────────
    public float T       => _tEst;
    public float V0      => _v0;
    /// True when the T window is full — estimate is reliable.
    public bool  IsReady => _tWindowCount >= WINDOW_T;
    public int   GapSearchSamples => _tWindowCount;

    // ── Constructor ───────────────────────────────────────────────────────────
    public LongitudinalDriverCalibrator(float s0, float aMax, float beta, float delta,
                                        float fallbackT, float v0)
    {
        _s0        = s0;
        _aMax      = aMax;
        _beta      = beta;
        _delta     = delta;
        _fallbackT = fallbackT;
        _v0        = v0;

        _tEst    = fallbackT;
        _tWindow = new float[WINDOW_T];
    }

    // ── Reset (call at start of each merge attempt) ───────────────────────────
    public void Reset(float tInit)
    {
        _tEst = tInit;

        System.Array.Clear(_tWindow, 0, WINDOW_T);
        _tWindowHead  = 0;
        _tWindowCount = 0;
        _tWindowSum   = 0f;
    }

    // ── GAP_SEARCH update — estimate T via running mean ───────────────────────
    //
    // Called every Nash tick (10 Hz) during GapSearch/Merge when leader is visible.
    // Filters:
    //   gap ∈ [2, 200] m, vx > 5 m/s, |dv| < 2 m/s (near-equilibrium),
    //   rhs > 0 (IDM physically consistent), tSample ∈ [T_MIN, T_MAX].
    public void UpdateGapSearch(float vx, float aActual, float gap, float dv)
    {
        if (gap < 2f || gap > 200f)  return;
        if (vx < 5f)                 return;
        if (Mathf.Abs(dv) > 2f)     return;   // near-equilibrium only

        float v0 = Mathf.Max(_v0, 1f);

        float freeTerm = 1f - Mathf.Pow(vx / v0, _delta);
        float rhs      = freeTerm - aActual / _aMax;
        if (rhs < 0f) return;

        float sStar   = gap * Mathf.Sqrt(rhs);
        float dvTerm  = vx * dv / (2f * Mathf.Sqrt(Mathf.Max(_aMax * _beta, 1e-6f)));
        float tSample = (sStar - _s0 - dvTerm) / Mathf.Max(vx, 1f);

        if (tSample < T_MIN || tSample > T_MAX) return;

        // Circular buffer running mean
        if (_tWindowCount == WINDOW_T)
            _tWindowSum -= _tWindow[_tWindowHead];   // evict oldest
        else
            _tWindowCount++;

        AddSample(tSample);
        DebugLog($"[Calibrator/IDM] T={_tEst:F3}  sample={tSample:F3}  n={_tWindowCount}/{WINDOW_T}  gap={gap:F1}  dv={dv:F2}");
    }

    // ── Following update — direct headway T = (gap - s0) / vx ────────────────
    //
    // Uses the gap the driver is actually maintaining as the IDM attractor.
    // No ax filter: the driver may press pedals; the chosen gap is what matters.
    // Filter |dv| < 1 m/s ensures speed is matched so gap reflects steady intent.
    public void UpdateFollowing(float vx, float gap, float dv)
    {
        if (gap < 2f || gap > 200f) return;
        if (vx < 5f)                return;
        if (Mathf.Abs(dv) > 1f)    return;

        float tSample = (gap - _s0) / Mathf.Max(vx, 1f);
        if (tSample < T_MIN || tSample > T_MAX) return;

        AddSample(tSample);
        DebugLog($"[Calibrator/Following] T={_tEst:F3}  sample={tSample:F3}  n={_tWindowCount}/{WINDOW_T}  gap={gap:F1}");
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────
    public string GetStatusString()
    {
        return $"T={_tEst:F2}s (n={_tWindowCount}/{WINDOW_T}, ready={IsReady}) | " +
               $"aMax={_aMax:F3} [fixed, Zhang2024]";
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    void AddSample(float tSample)
    {
        if (_tWindowCount == WINDOW_T)
            _tWindowSum -= _tWindow[_tWindowHead];
        else
            _tWindowCount++;

        _tWindow[_tWindowHead] = tSample;
        _tWindowSum           += tSample;
        _tWindowHead           = (_tWindowHead + 1) % WINDOW_T;
        _tEst                  = _tWindowSum / _tWindowCount;
    }

    [System.Diagnostics.Conditional("UNITY_EDITOR")]
    static void DebugLog(string msg) => Debug.Log(msg);
}
