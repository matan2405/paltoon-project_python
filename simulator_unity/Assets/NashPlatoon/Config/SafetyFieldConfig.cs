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
    public float StandstillDist    = 4.177f;  // PLATOON_STANDSTILL_DISTANCE [m]  (= vehicle length)
    public float VehicleLength     = 4.177f;  // VEHICLE_LENGTH          [m]
    public float LaneWidth         = 3.5f;    // LANE_WIDTH              [m]

    // ── Longitudinal DSF dynamic scaling (coordinator.py _long_dsf_params) ──────
    // potential = (Mo × Ri) / (gap_error/s + ε)²;  w_dist, w_vel, w_risk applied on top.
    // pos_mult: no_leader=0.8, middle=1.2, tail=1.0  (hardcoded in coordinator).
    // vel_factor = clamp(1 + DsfVelScaleCoeff*(vx/DsfVelScaleRef - 1),  0.5, 2.0)
    // s  = DsfBaseRadius * pos_mult * vel_factor  (EMA with LongEmaAlpha)
    // Mo = DsfBaseMass   * pos_mult               (EMA)
    // Ri = DsfBaseRi * (DsfMiddleRiMult if middle else 1.0)  (EMA)
    [Header("Long — Dynamic DSF ellipse scaling")]
    public float DsfBaseRadius    = 1.5f;    // BASE_RADIUS   safe-distance ellipse half-axis [m]
    public float DsfBaseMass      = 1.0f;    // BASE_MASS     normalized virtual mass
    public float DsfBaseRi        = 1.0f;    // BASE_Ri       influence factor
    public float DsfBaseDr        = 0.5f;    // BASE_DR       driver risk multiplier
    public float DsfVelScaleCoeff = 0.8f;    // vel_factor slope: 1+coeff*(vx/ref - 1)
    public float DsfVelScaleRef   = 33.33f;  // reference speed for vel_factor = 120 km/h [m/s]
    public float DsfMiddleRiMult  = 1.5f;    // Ri boost when ego is middle vehicle in platoon

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
}
