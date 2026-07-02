using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/Timing")]
public class SimTimingConfig : ScriptableObject {
    public float SimDt         => Time.fixedDeltaTime;
    public float NashDtLong    = 0.1f;
    public float NashDtLat     = 0.05f;
    public int   NashNp        = 20;
    public int   NashNu        = 10;
    public float PhaseLockTime = 5.0f;
    public float MaxSteerRate  = 0.087f;
    // ISO 11270 §5.4: jerk_lat ≤ LatJerkMax m/s³
    // Approximation a_lat ≈ vx·δ̇ gives: |Δ²δ| ≤ (LatJerkMax/vx)·dt²
    // Applied in LateralNashSolver (dt=NashDtLat) and RateLimitDelta (dt=SimDt).
    public float LatJerkMax    = 5.0f;   // m/s³ — ISO 11270 §5.4
}
