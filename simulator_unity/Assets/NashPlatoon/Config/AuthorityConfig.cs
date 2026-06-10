using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/Authority")]
public class AuthorityConfig : ScriptableObject {

    // ── Longitudinal authority (coordinator.py _long_authority) ──────────────────
    // Three-stage:
    //   1. Safety sigmoid:  λ_s  = f(|risk_force|, ForceMidpoint, Steepness)
    //   2. Gap+vel Swain & Rath Eq.15:  λ_g = g(gap_error, vel_error)  [uses SigmoidM1/M2]
    //   3. λ_raw = max(λ_s, λ_g);  then adaptive EMA with AlphaBase/Fast
    [Header("Long — sigmoid on risk force")]
    public float LongLambdaMin     = 0.1f;    // LONG_AUTHORITY_LAMBDA_MIN
    public float LongLambdaMax     = 100.0f;  // LONG_AUTHORITY_LAMBDA_MAX
    public float LongForceMidpoint = 400.0f;  // LONG_AUTHORITY_FORCE_MIDPOINT [N]
    public float LongSteepness     = 0.015f;  // LONG_AUTHORITY_K_STEEPNESS
    public float LongAlphaBase     = 0.05f;   // LONG_AUTHORITY_ALPHA_BASE  (slow EMA)
    public float LongAlphaFast     = 0.12f;   // LONG_AUTHORITY_ALPHA_FAST  (fast EMA)
    // EMA switches: alpha_fast when combined_error > 5.0 m;
    //               alpha_base when combined_error < 2.0 m; linear blend in between.

    // ── Swain & Rath Eq. 15 sigmoid (shared by long and lat) ─────────────────────
    // l_n = min(1 - |gap_err|/GapErrorMax,  1 - |vel_err|/VelErrorMax)  clipped [0,1]
    // γ₁  = 1/(1+exp(M1·(-l_n + M2)));  λ_gap = (1-γ₁)/γ₁
    [Header("Swain & Rath Eq.15 sigmoid (shared)")]
    public float AuthGapErrorMax   = 5.0f;    // AUTHORITY_GAP_ERROR_MAX   [m]
    public float AuthVelErrorMax   = 1.5f;    // AUTHORITY_VEL_ERROR_MAX   [m/s]
    public float SigmoidM1         = 2.0f;    // LAT_AUTHORITY_SIGMOID_M1  (slope)
    public float SigmoidM2         = 0.5f;    // LAT_AUTHORITY_SIGMOID_M2  (centre shift)

    // ── Lateral authority (coordinator.py _lat_authority) ────────────────────────
    // Same three-stage as longitudinal but:
    //   Gap component uses lane-width normalisation: l_n = (LaneWidth/2 - |y_err|) / (LaneWidth/2)
    //   Adaptive EMA threshold: AlphaFastThr / AlphaBaseThr on |y_error|
    [Header("Lat — sigmoid on DSF force")]
    public float LatLambdaMin      = 0.1f;    // LAT_AUTHORITY_LAMBDA_MIN
    public float LatLambdaMax      = 10.0f;   // LAT_AUTHORITY_LAMBDA_MAX
    public float LatForceMidpoint  = 200.0f;  // LAT_AUTHORITY_FORCE_MIDPOINT [N]
    public float LatSteepness      = 0.025f;  // LAT_AUTHORITY_K_STEEPNESS
    public float LatAlphaBase      = 0.02f;   // LAT_AUTHORITY_ALPHA_BASE
    public float LatAlphaFast      = 0.08f;   // LAT_AUTHORITY_ALPHA_FAST
    public float LatAlphaFastThr   = 1.5f;    // LAT_AUTHORITY_ALPHA_FAST_THR [m]
    public float LatAlphaBaseThr   = 0.5f;    // LAT_AUTHORITY_ALPHA_BASE_THR [m]

    // ── Long FOLLOWING phase caps ─────────────────────────────────────────────
    [Header("Long — FOLLOWING phase caps")]
    public float LongLambdaMaxFollowing = 5.0f;  // LONG_AUTHORITY_LAMBDA_MAX_FOLLOWING
    public float LongAlphaFollowing     = 0.05f; // LONG_AUTHORITY_ALPHA_FOLLOWING (faster EMA ~3s)
}