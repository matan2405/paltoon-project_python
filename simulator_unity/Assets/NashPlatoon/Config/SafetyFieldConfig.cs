using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/SafetyField")]
public class SafetyFieldConfig : ScriptableObject {

    // ── Longitudinal safety force (coordinator.py _long_leader/follower_force_raw) ──
    // Piecewise-linear gap formula:
    //   gap < EmergencyDist          → F = MaxRepulsiveForce  (saturate)
    //   EmergencyDist ≤ gap < MinSafe → F = lerp(0 → MaxRepulsiveForce)
    //   gap ≥ MinSafe                → F = clip(gap_error/des_gap * Max, -0.25Max, Max)
    //   des_gap = StandstillDist + TimeGap * vx
    [Header("Long — piecewise gap force")]
    public float LongMinSafeDist   = 7.0f;    // LONG_SAFETY_MIN_SAFE_DISTANCE  [m]
    public float LongEmergencyDist = 8.0f;    // LONG_SAFETY_EMERGENCY_BRAKE_DIST [m]
    public float LongMaxForce      = 800.0f;  // LONG_SAFETY_MAX_REPULSIVE_FORCE [N]
    public float LongEmaAlpha      = 0.2f;    // LONG_SAFETY_FILTER_ALPHA

    // ── Platoon geometry (shared by safety force and reference generators) ────────
    [Header("Platoon geometry")]
    public float PlatoonTimeGap    = 1.5f;    // PLATOON_TIME_GAP        [s]
    public float StandstillDist    = 2.0f;    // PLATOON_STANDSTILL_DISTANCE [m]
    public float VehicleLength     = 4.177f;  // VEHICLE_LENGTH          [m]
    public float LaneWidth         = 3.5f;    // LANE_WIDTH              [m]

    // ── Lateral DSF (Wang et al. 2015/2016, Li et al. 2019) ───────────────────────
    // Full formula (coordinator.py _lat_safety_force):
    //   Virtual mass:    M = DsfVehicleMass * (DsfSpeedCoeff * |v|^DsfSpeedExp + DsfSpeedOffset)
    //   Elliptic dist:   r* = sqrt((dx/a)² + (dy/b)²);  a = max(|Δv|·Ts, AMin),  b = Tau
    //   Field strength:  E  = G * M_obs / r*
    //   Kinetic corr:    E *= exp(K1 * v_obs * cos(θ_v))
    //   Field force:     Fr = E * M_ego * exp(-K2 * v_ego * cos(θ_i)) * (1+DR)
    //   Direction:       Fr *= tanh(dy / sigma)
    [Header("Lat — Wang et al. DSF")]
    public float LatDsfTs          = 2.0f;        // LAT_DSF_TS    TTC margin      [s]
    public float LatDsfTau         = 2.0f;        // LAT_DSF_TAU   lateral radius  [m]
    public float LatDsfAMin        = 2.0f;        // LAT_DSF_A_MIN min long semi-axis [m]
    public float LatDsfG           = 6e-3f;       // LAT_DSF_G     field scaling constant
    public float LatDsfK1          = 0.005f;      // LAT_DSF_K1    kinetic correction (obs)
    public float LatDsfK2          = 0.005f;      // LAT_DSF_K2    kinetic correction (ego)
    public float LatDsfDr          = 0.5f;        // LAT_DSF_DR    driver risk factor
    public float DsfVehicleMass    = 1304.0f;     // DSF_VEHICLE_MASS  [kg]
    public float DsfSpeedCoeff     = 1.566e-14f;  // DSF_SPEED_COEFF
    public float DsfSpeedExponent  = 6.687f;      // DSF_SPEED_EXPONENT
    public float DsfSpeedOffset    = 0.3345f;     // DSF_SPEED_OFFSET
    public float DsfEpsilon        = 1e-3f;       // DSF_EPSILON   min elliptic dist [m]
    public float LatMaxForce       = 2000.0f;     // LAT_SAFETY_MAX_FORCE  [N]
    public float LatEmaAlpha       = 0.3f;        // LAT_SAFETY_FILTER_ALPHA
    public float LatDsfSigma       = 0.5f;        // DSF_SIGMA  tanh direction-smoothing width [m]

    // ── Road boundary repulsion ───────────────────────────────────────────────
    [Header("Lat — Road boundaries")]
    public float RoadHalfWidth     = 7.0f;        // Half road width [m]
    public float BoundaryGain      = 150.0f;      // Boundary force gain [N]
    public float BoundaryScale     = 2.0f;        // tanh normalisation scale [m]
    public float BoundaryProximity = 3.0f;        // Activation distance from edge [m]
    public float BoundaryEpsilon   = 0.1f;        // Min dist to avoid 1/0 [m]

    // ── Li et al. (2019) elliptic DSF core params ─────────────────────────────
    [Header("Long — Li et al. elliptic DSF")]
    public float LongBaseRadius         = 10.0f;  // LONG_SAFETY_BASE_RADIUS [m]
    public float LongObstacleMass       = 400.0f; // LONG_SAFETY_OBSTACLE_MASS [kg]
    public float LongInfluenceFactor    = 1.0f;   // LONG_SAFETY_INFLUENCE_FACTOR
    public float LongDriverRisk         = 0.5f;   // LONG_SAFETY_DRIVER_RISK
    public float LongEpsilon            = 0.1f;   // LONG_SAFETY_EPSILON [m]
    public float LongDistanceDecay      = 15.0f;  // LONG_SAFETY_DISTANCE_DECAY [m]
    public float LongFollowingSoftFactor = 0.35f; // LONG_SAFETY_FOLLOWING_SOFT_FACTOR

    // ── Dynamic DSF position and velocity multipliers ─────────────────────────
    [Header("Long — Dynamic DSF multipliers")]
    public float LongLeaderPosMult      = 0.8f;   // LONG_SAFETY_LEADER_POSITION_MULT
    public float LongMiddlePosMult      = 1.2f;   // LONG_SAFETY_MIDDLE_POSITION_MULT
    public float LongFollowerPosMult    = 1.0f;   // LONG_SAFETY_FOLLOWER_POSITION_MULT
    public float LongVelocityReference  = 120.0f; // LONG_SAFETY_VELOCITY_REFERENCE [m/s, same as split module]
    public float LongVelocityScaling    = 0.8f;   // LONG_SAFETY_VELOCITY_SCALING
    public float LongPlatoonCoherence   = 1.5f;   // LONG_SAFETY_PLATOON_COHERENCE (Ri multiplier in middle)
    public float LongFollowerWeight     = 0.5f;   // LONG_SAFETY_FOLLOWER_WEIGHT
    public float FollowingGapErrorFactor = 0.15f; // FOLLOWING_GAP_ERROR_FACTOR (soft-transition threshold)
}
