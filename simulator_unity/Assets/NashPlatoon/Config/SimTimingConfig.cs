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

    [Header("Phase transition thresholds")]
    // GapSearch → Merge
    public float MergeVelThreshold       = 1.5f;   // max |velErr| before lateral merge starts [m/s]
    public float MergeGapThreshold       = 0.40f;  // max |gap-desGap|/desGap before merge [–]
    // Merge → Following
    public float MergeGapTolerance       = 0.10f;  // max |gap-desGap|/desGap to enter Following [–]
    public float MergeVelTolerance       = 0.5f;   // max |velErr| to enter Following [m/s]
    public float MergeGapConvergenceRate = 0.3f;   // max |d(gap_err)/dt| to enter Following [m/s]
    // Following → Merge (hysteresis — intentionally looser than entry conditions)
    public float FollowingHysteresisGapFrac = 0.35f;  // gap fraction threshold to re-enter Merge [–]
    public float FollowingHysteresisDivRate = 0.5f;   // min gap divergence rate to trigger [m/s]
    public float FollowingHysteresisHoldTime = 3.0f;  // must hold diverging for this long [s]
}
