using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/NashWeights")]
public class NashWeightsConfig : ScriptableObject {
    [Header("Long tracking")]
    public float Q_pos = 5000f, Q_vel = 1500f;
    public float Q_pos_terminal = 50000f, Q_vel_terminal = 6000f;

    [Header("Long effort")]
    // Q_vel/R1 ≈ 0.02 (= Python current B1 Pareto: Q_vel=1500, R1=75000).
    public float R1_long = 75000f, R2_long = 120000f;

    [Header("Long — Adaptive R2 EMA (R2LongStart → R2LongFollow per l_n)")]
    // R2LongStart: system leads when joining. R2LongFollow: human leads in steady-state.
    // R2LongFollow < R1_long so driver has higher authority in steady-state (Na et al. 2022).
    public float R2LongStart           = 120000f;  // R2 when far from platoon (system leads)
    public float R2LongFollow          =  40000f;  // R2 in steady-state following (human leads; < R1_long=75000)
    public float R2LongEmaAlpha        =    0.05f; // slow EMA smoothing constant
    public float R2LongUpdateThreshold =     50f;  // min |ΔR2| to trigger solver rebuild

    [Header("Long — Activity-based R2 scaling (Lazcano et al. 2021)")]
    // When driver actively presses pedals, R2 scales down to R2LongFollow × R2ActivityMinScale,
    // amplifying u2 and giving the driver stronger authority during intentional inputs.
    // activity in [0, 0.3] → scale ramps from 1 → R2ActivityMinScale.
    public float R2ActivityMinScale = 0.25f;

    [Header("Long constraints")]
    public float U1Min = -7f, U1Max = 4f;
    public float U2Min = -4f, U2Max = 3f;  // LONG_NASH_U2_MIN=-4, U2_MAX=3
    public float Du1Max = 1.5f, Du2Max = 2.0f;
    public float VMin = 0f, VMax = 50f;
    public float GapMin = 7f;

    [Header("Lat tracking")]
    // Q_psi reduced relative to Q_y: at vx≈30 m/s, ψ=0.1 rad → ẏ=3 m/s (large).
    // Previous Q_psi/Q_y=12.5 caused the solver to over-correct heading aggressively
    // → large δ → ψ overshoots → oscillation. Target ratio Q_psi/Q_y ≈ 1 so that
    // a 0.1 rad heading error costs the same as a ~0.3 m lateral error.
    public float Q_y = 5000f,   Q_psi = 2000f;
    public float Q_y_terminal = 20000f, Q_psi_terminal = 8000f;

    [Header("Lat effort")]
    // R increased 5× to damp IBR coupling: convergence requires ||HQ1H||/R < 1.
    // With vx≈30 m/s the H matrix has large entries from the -vx·ψ coupling in Ac.
    public float R1_lat = 250000f, R2_lat = 250000f;

    [Header("Lat — Adaptive R2 EMA (R2LatStart → R2LatFollow per l_n)")]
    // R2LatFollow < R1_lat so driver has higher authority once centred (Na et al. 2022).
    public float R2LatStart           = 250000f;  // R2 when far from target lane
    public float R2LatFollow          =  75000f;  // R2 once lane-centred (< R1_lat=250000)
    public float R2LatEmaAlpha        =    0.05f;
    public float R2LatUpdateThreshold =   200f;

    [Header("Lat — Activity-based R2 scaling (Lazcano et al. 2021)")]
    // When driver actively steers, R2 scales down to R2LatFollow × R2LatActivityMinScale.
    // steering in [0, 0.3] → scale ramps from 1 → R2LatActivityMinScale.
    public float R2LatActivityMinScale = 0.25f;

    [Header("Lat constraints")]
    public float DeltaMin = -0.436f, DeltaMax = 0.436f;
    public float DDeltaMaxHuman  = 0.020f;  // human driver: 0.40 rad/s at 20 Hz (~Audi TT realistic steering rate)
    public float DDeltaMaxSystem = 0.035f;  // ADAS system:  0.70 rad/s at 20 Hz (faster correction authority)

    [Header("IBR")]
    public int   IbrMaxIter = 15;
    public float IbrTol = 1e-4f;

    [Header("GP/MA-IDM — Zhang & Sun 2024")]
    public float SigmaK = 1f, Ell = 2f, GammaRisk = 0f;  // 0 = off (matches Python default sigma_k=0)
}
