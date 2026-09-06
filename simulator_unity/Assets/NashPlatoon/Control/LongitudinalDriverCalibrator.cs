// LongitudinalDriverCalibrator — Online IDM time-headway (T) estimation.
//
// All phases use the same direct headway formula:
//   T = (gap - s0) / vx
//
// Rationale: IDM inversion (used previously in GapSearch/Merge) assumes the driver
// is near IDM equilibrium, which is never true during catch-up or lane change.
// Direct headway makes no equilibrium assumption — it reads the gap the driver
// is currently accepting and asks: "if they held this gap at this speed, what T?"
//
// Phase-specific filters:
//   GapSearch : no |dv| filter — driver is intentionally closing, gap is informative.
//               Measures the natural approach headway (typically 1.0–2.0 s).
//   Merge     : no |dv| filter — driver is executing a lane change.
//   Following : |dv| < 1 m/s — speed-matched steady-state only.
//
// The circular buffer (window=20) runs continuously. Following samples at platoon
// headway (~0.4 s) quickly evict the larger GapSearch/Merge values, so T
// converges to the platoon-synchronised headway within ~2 s of Following entry.
//
// Fallback: T = fallbackT (1.5 s population mean, Zhang 2024) until first sample.
//
// aMax fixed at 0.553 m/s² (Zhang & Sun 2024). Not calibrated from HITL.

using UnityEngine;

public class LongitudinalDriverCalibrator
{
    // ── Fixed IDM parameters (not calibrated) ────────────────────────────────
    readonly float _s0;        // jam spacing [m]
    readonly float _aMax;      // fixed comfortable acceleration [m/s²] (Zhang 2024 pop. mean) — for diagnostics
    readonly float _fallbackT; // T used if calibration insufficient
    readonly float _v0;        // target velocity [m/s] (from constructor)

    // ── Calibration bounds ────────────────────────────────────────────────────
    // T_MIN = 0.3 s: CACC/SAE J3016 minimum platooning THW — allows Following samples
    // at platoon headway (0.4–0.5 s) to be accepted and evict the pre-join 1.5 s estimate.
    const float T_MIN = 0.3f;
    const float T_MAX = 3.0f;

    // ── T running-mean window ─────────────────────────────────────────────────
    // Window of 20 samples at 10 Hz Nash rate → ~2 s to fill.
    // IsReady triggers when full; _tEst updates from the first sample onward.
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
    public LongitudinalDriverCalibrator(float s0, float aMax, float fallbackT, float v0)
    {
        _s0        = s0;
        _aMax      = aMax;
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

    // ── GAP_SEARCH / MERGE update — direct headway T = (gap - s0) / vx ─────────
    //
    // No |dv| filter: driver is intentionally closing gap (GapSearch) or executing
    // a lane change (Merge). The gap they accept at this moment reflects their
    // natural approach headway — informative even out of IDM equilibrium.
    // IDM inversion is not used: it requires near-equilibrium and breaks down
    // when aActual >> aMax (catch-up) or when freeTerm → 0 (vx ≈ v0).
    public void UpdateGapSearch(float vx, float gap, float dv)
    {
        if (gap < 2f || gap > 200f) return;
        if (vx < 5f)                return;
        if (dv <= 0f)               return;   // ego not closing → not informative

        float tSample = (gap - _s0) / Mathf.Max(vx, 1f);
        if (tSample < T_MIN || tSample > T_MAX) return;

        AddSample(tSample);
        DebugLog($"[Calibrator/GapSearch] T={_tEst:F3}  sample={tSample:F3}  n={_tWindowCount}/{WINDOW_T}  gap={gap:F1}  dv={dv:F2}");
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
