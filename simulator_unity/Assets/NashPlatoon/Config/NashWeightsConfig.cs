using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/NashWeights")]
public class NashWeightsConfig : ScriptableObject {
    [Header("Long tracking")]
    public float Q_pos = 5000f, Q_vel = 1500f;
    public float Q_pos_terminal = 50000f, Q_vel_terminal = 6000f;

    [Header("Long effort")]
    // R1_long: GapSearch — CATCHUP vel_err is large (≥10 m/s), QP clips to aMax regardless of R1.
    //          R1=75000 safe here (same u1 output as R1=10000 after aMax clip).
    // R1_long_Merge: vel_err≈0 when speeds matched but gap_err=14m. R1=75000 → u1=0.32 m/s² (too weak).
    //          R1=10000 → u1=1.36 m/s² (closes 14m gap). Mirrors R1_lat_GapSearch pattern.
    public float R1_long = 75000f, R2_long = 120000f;
    public float R1_long_Merge     = 10000f;  // replaces R1_long on GapSearch→Merge transition
    public float R1_long_Following = 20000f;  // replaces R1_long in Following (stronger than 75000 to close residual gap)

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
    // Phase-dependent: system-led during Merge, driver-led during Following.
    //   Merge:     1.0 — no R2 reduction; system retains authority for platoon join.
    //   Following: 0.5 — driver has meaningful authority in steady-state.
    public float R2ActivityMinScale          = 0.5f;   // Following default
    public float R2ActivityMinScale_Merge    = 1.0f;   // Merge — no driver amplification

    // Level 1-2 safety ceilings — wide enough to never constrain GetPhysicalAccelBounds().
    // DO NOT tighten these for comfort or style (use cost weights S1/S2 for that).
    // U2Min/U2Max are intentionally unused: UpdatePhysicalBounds() mirrors u1 bounds onto
    // u2 because both players act through the same physical vehicle.
    // Du2Max intentionally tighter than Du1Max: human jerk is smoothed via cost (R2),
    // but a hard rate cap of 0.2 m/s²/step prevents inter-step oscillation from u2.
    [Header("Long constraints — Level 1/2 ceilings (wide, not tuning knobs)")]
    public float U1Min = -20f, U1Max = 10f;
    public float U2Min = -20f, U2Max = 10f;   // kept for serialisation; mirrored from U1 in solver
    public float Du1Max = 15f, Du2Max = 2.0f; // Du2Max=2→0.2 m/s²/step cap prevents u2 inter-step oscillation
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

    [Header("Lat — GapSearch effort override")]
    // During GapSearch the goal is ψ→0 only (hold y at _gsLockedY, no lane change).
    // R1/R2 must be reduced so the solver can apply enough δ to damp a large entry ψ
    // (e.g. ψ=25° from driver steering before J-press). At R1=250000 the solver
    // produces δ≈−2° which is insufficient; R1_lat_GapSearch=25000 gives δ≈−20°.
    // Restored to R1_lat/R2_lat on GapSearch→Merge transition.
    public float R1_lat_GapSearch = 25000f;
    public float R2_lat_GapSearch = 25000f;

    [Header("Lat — Adaptive R2 EMA (R2LatStart → R2LatFollow per l_n)")]
    // R2LatFollow < R1_lat so driver has higher authority once centred (Na et al. 2022).
    public float R2LatStart           = 250000f;  // R2 when far from target lane
    public float R2LatFollow          =  30000f;  // R2 once lane-centred (< R1_lat=250000)
    public float R2LatEmaAlpha        =    0.05f;
    public float R2LatUpdateThreshold =   200f;

    [Header("Lat — Activity-based R2 scaling (Lazcano et al. 2021)")]
    // When driver actively steers, R2 scales down to R2LatFollow × R2LatActivityMinScale.
    // steering in [0, 0.3] → scale ramps from 1 → R2LatActivityMinScale.
    public float R2LatActivityMinScale = 0.25f;

    [Header("Lat — Following boundary weights")]
    // In Following: beyondFrac starts rising above FollowingYDeadband (not quarterLane).
    // Small deadband (3 cm) ensures even minor lane offsets trigger Q_y correction,
    // preventing permanent u1/u2 cancellation when y_err < quarterLane.
    // In Merge the wider quarterLane deadband is used for a softer transition.
    public float FollowingYDeadband = 0.03f;  // y_err threshold for beyondFrac in Following [m]
    // Inside deadband: R2=R2LatFree (large → driver feels free), Q_y=Q_y (low).
    // Outside deadband: R2 lerps toward R2LatStart, Q_y lerps toward Q_y_BeyondBound.
    public float R2LatFree         = 250000f; // R2 inside deadband (driver feels free)
    public float Q_y_BeyondBound   = 150000f; // Q_y outside deadband (strong correction pull)
    public float Q_psi_BeyondBound =   8000f; // Q_psi outside quarter-lane
    public float Q_yTerm_BeyondBound   = 80000f; // Q_y_terminal outside quarter-lane
    public float Q_psiTerm_BeyondBound = 20000f; // Q_psi_terminal outside quarter-lane
    public float Q_LatUpdateThreshold  =   500f;  // min |ΔQ_y| before triggering UpdateQ1

    [Header("Lat constraints")]
    public float DeltaMin = -0.436f, DeltaMax = 0.436f;
    // DDeltaMax computed dynamically in solver: DDelta = LatJerkMax * NashDtLat / vx
    public float LatJerkMaxHuman  = 5.0f;  // UN ECE-R79 §5.6.2.1.3(c) — B1 steady-state jerk limit [m/s³]
    public float LatJerkMaxSystem = 7.0f;  // UN ECE-R79 §5.6.4.4 Category C transient (×1.4 for ≤2s), else 5 m/s³
    public float LatJerkTransientSec = 2.0f;  // max duration of transient jerk allowance [s]

    [Header("IBR")]
    public int   IbrMaxIter = 15;
    public float IbrTol = 1e-4f;

    [Header("GP/MA-IDM — Zhang & Sun 2024")]
    public float SigmaK = 1f, Ell = 2f, GammaRisk = 0f;  // 0 = off (matches Python default sigma_k=0)
}
