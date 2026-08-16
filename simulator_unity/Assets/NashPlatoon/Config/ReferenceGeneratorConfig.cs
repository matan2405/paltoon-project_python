using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/ReferenceGenerator")]
public class ReferenceGeneratorConfig : ScriptableObject {

    // ── Rajamani Ch. 6.7 Transitional (System long reference) ───────────────
    [Header("Rajamani Transitional — System longitudinal reference")]
    public float RajamaniH         = 1.5f;    // RAJAMANI_H  desired time headway [s]
    public float AComfort          = 1.5f;    // comfortable deceleration [m/s²]
    public float KV                = 0.4f;    // velocity tracking gain [1/s]
    public float K1Ctg             = 0.05f;   // CTG position feedback gain
    public float K2Ctg             = 0.4f;    // CTG velocity feedback gain
    public float DetectionRange    = 300.0f;  // free-road cruise threshold [m]
    public float TtcThreshold      = 3.0f;    // Lee 2004: Urgency 3 "forced" threshold → collision avoidance [s]
    public float MinCritGapOffset  = 2.0f;    // min critical gap = offset + L/2 [m]
    public float MaxEmergDecel     = -5.0f;   // emergency deceleration cap [m/s²]
    public float CatchupMultiplier    = 1.05f;  // cruise target = V_platoon × this
    public float FreeDelta            = 4.0f;   // IDM exponent for free-road acceleration
    public float CatchupVelThreshold  = 5.0f;   // vel diff above this always triggers CATCHUP [m/s]
    // Gap deadband: gap errors SMALLER than (desGap × Frac) are treated as settled →
    // only KV velocity tracking, no parabolic correction. Relative to desGap so the
    // deadband scales correctly with speed (desGap = s0 + H*vx).
    // Following: 3% (≈0.55 m at 120 km/h) — suppresses parabolic limit cycle in steady-state.
    // Merge:     0% (always active parabola) — full pursuit of gap error until Following.
    //            Non-zero Merge deadband can leave a static offset if the system does not
    //            transition to Following.
    public float FollowingGapDeadbandFrac = 0.03f;
    public float MergeGapDeadbandFrac     = 0.00f;

    // ── IDM — Human longitudinal reference ───────────────────────────────────
    // HighD-calibrated values from Zhang & Sun 2024 (passenger cars, ~68 km/h steady-state).
    // T and b are driver-dependent (not speed-dependent) → applicable to MERGE/FOLLOWING phases.
    [Header("IDM — Human longitudinal reference (HighD, Zhang & Sun 2024)")]
    public float IdmAMax           = 0.553f;  // Zhang & Sun 2024 hierarchical MA-IDM population mean α [m/s²]
    public float IdmS0             = 3.416f;  // minimum gap [m]
    public float IdmDelta          = 4.0f;    // free-road exponent
    public float IdmPlanT          = 1.5f;    // time headway [s] — Zhang & Sun 2024 population mean
    public float IdmPlanB          = 1.649f;  // comfortable deceleration [m/s²]

    // ── Driver profile (normal) — Lateral reference ───────────────────────────
    // For HITL use, these are measured from the real driver.
    // These defaults correspond to the 'normal' profile in unified/config.py.
    [Header("Driver profile — Lateral reference")]
    public float VelocityOffset    = 0.0f;    // desired speed relative to platoon [m/s]
    public float HumanTlc          = 4.5f;    // human lane-change duration [s]
    public float MaxHeadingDeg     = 4.0f;    // max heading angle for sys trajectory shaping [deg]
    public float SysTlcMultiplier  = 1.5f;    // sys T_lc = human TLC × this

    // ── R2 Human Lateral — Lee & Olsen 2004 (~97 km/h, 8,667 lane changes) ───
    [Header("R2 Human Lateral — Lee & Olsen 2004 (~97 km/h)")]
    public float R2MaxHeadingDeg   = 2.6f;    // empirical max heading at lane change [deg]

    // ── R2 Neuromuscular IIR — Swain & Rath 2023 Eq. 7 ──────────────────────
    [Header("R2 Neuromuscular IIR — Swain & Rath 2023 Eq.7")]
    public float NeuroJ            = 0.0037f; // inertia coefficient
    public float NeuroB            = 0.1363f; // damping coefficient
    public float NeuroK            = 1.1742f; // stiffness coefficient
}