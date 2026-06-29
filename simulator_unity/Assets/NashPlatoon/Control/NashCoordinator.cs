// NashCoordinator — MonoBehaviour (Script Execution Order -50).
//
// Orchestrates longitudinal + lateral Nash DMPC for a single ego vehicle.
// Mirrors unified/control/coordinator.py: UnifiedCoordinator.
//
// Scheduling:
//   Long Nash: every NashDtLong seconds (10 Hz)
//   Lat Nash:  every NashDtLat  seconds (20 Hz), only during MERGE / FOLLOWING
//
// Phase machine: APPROACH → GAP_SEARCH → MERGE → FOLLOWING
//
// Output (per Long Nash step):  ego.NashThrottle / NashBrake (via AccelerationToInputs)
// Output (per Lat Nash step):   ego.NashSteerNorm = delta / maxWheelRad
// Both:                         ego.NashModeActive = true
//
// Note: LateralNashSolver is set from outside (via SetLateralSolver) once Plan 6
//       creates it. If null, lateral Nash is skipped and steering remains manual.
using UnityEngine;
using NashPlatoon;

public class NashCoordinator : MonoBehaviour
{
    // ── Inspector wiring ──────────────────────────────────────────────────────
    [Header("Links")]
    [SerializeField] AdvancedBicycleModel  ego;
    [SerializeField] PlatoonManager        platoonManager;
    [SerializeField] NashPlatoonSettings   settings;

    [Header("Platoon geometry")]
    [SerializeField] float platoonLaneY      = 0f;    // PLATOON_LANE_Y (solver convention: pos.x - ego._x0)
    [SerializeField] bool  autoDetectLane    = true;  // if true, overrides platoonLaneY on J-press with snapped lane centre
    [SerializeField] float laneWidth         = 2.0f;  // road lane width [m] — used for lane-centre snapping
    [SerializeField] float approachCheckDist = 300f;  // APPROACH_MOBIL_CHECK_DISTANCE [m]

    // ── Public read-only state ────────────────────────────────────────────────
    public MergePhase       Phase      { get; private set; } = MergePhase.Approach;
    public bool             NashActive { get; private set; } = false;
    public float            LongLambda { get; private set; } = 0.1f;
    public float            LatLambda  { get; private set; } = 0.1f;
    public CoordinatorData  Data       { get; }               = new CoordinatorData();

    // ── Nash sub-systems ──────────────────────────────────────────────────────
    LongitudinalNashSolver   _longNash;
    LateralNashSolver        _latNash;
    LongitudinalSafetyField  _longSafety;
    LateralSafetyField       _latSafety;
    AuthorityAllocator       _authority;
    LongitudinalReferenceGen _longRef;
    LateralReferenceGen      _latRef;
    MOBILLaneChange          _mobil;
    bool                     _mobilApproved;

    // ── EMA force state ───────────────────────────────────────────────────────
    float _longForceEma;
    float _latForceEma;

    // ── Adaptive R2 EMA state (coordinator.py lines 1026-1034 / 1063-1071) ────
    float _r2LongCurrent, _r2LongPrev;
    float _r2LatCurrent,  _r2LatPrev;

    // ── Lateral conflict detection (Lazcano et al., IEEE T-MECH 2021) ─────────
    // Moving-average filter over last 5 Nash ticks: smoother than instantaneous
    // boost + EMA decay. boost = lerp(0, Mag, (mav−0.2)/0.6) with 20% deadband.
    const int   ConflictBufLen   = 5;
    const float ConflictBoostMag = 0.5f;   // R2_lat *= (1 + 0.5) at full conflict rate
    float[]     _conflictBuf     = new float[ConflictBufLen];
    int         _conflictIdx     = 0;

    // ── Multi-rate timers ─────────────────────────────────────────────────────
    float _longTimer;
    float _latTimer;

    // ── Phase hold state ──────────────────────────────────────────────────────
    float _gapSearchStart = -1f;
    float _phaseHoldTimer;

    // ── Locked merge partners (set at GAP_SEARCH→MERGE) ──────────────────────
    AdvancedBicycleModel _lockedLeader;
    AdvancedBicycleModel _lockedFollower;
    bool                 _mergeLocked;

    // =========================================================================
    // Unity lifecycle
    // =========================================================================

    void Awake()
    {
        SimCfg.I    = settings;
        _longSafety = new LongitudinalSafetyField(settings.SafetyField);
        _latSafety  = new LateralSafetyField(settings.SafetyField);
        _authority  = new AuthorityAllocator(settings.Authority, settings.SafetyField);
        _longRef    = new LongitudinalReferenceGen(settings.RefGen, settings.SafetyField);
        _latRef     = new LateralReferenceGen(settings.RefGen);
        _mobil      = new MOBILLaneChange();
        _mobil.SetPoliteness("normal");

        _r2LongCurrent = _r2LongPrev = settings.Nash.R2LongStart;
        _r2LatCurrent  = _r2LatPrev  = settings.Nash.R2LatStart;
    }

    void Start()
    {
        // LongitudinalNashSolver.Create reads ego.vehicleParams (initialised in ego's Awake).
        // Start runs after all Awakes — guaranteed safe to access ego here.
        _longNash = LongitudinalNashSolver.Create(settings.Nash, settings.Timing, ego);

        if (settings.Nash.GammaRisk > 0f)
            _longNash.SetGpModifier(new SEKernelGP(settings.Nash));
    }

    LateralNashSolver EnsureLatNash()
    {
        if (_latNash != null) return _latNash;
        _latNash = LateralNashSolver.Create(settings.Nash, settings.Timing, ego);

        // Attach SE-kernel GP risk modifier when γ > 0
        if (settings.Nash.GammaRisk > 0f)
            _latNash.SetGpModifier(new SEKernelGP(settings.Nash));
        return _latNash;
    }

    public void EnableNash()
    {
        var platoon = platoonManager.GetPlatoonVehicles();
        float sumX = 0f; float sumVx = 0f; int n = 0;
        foreach (var v in platoon)
        {
            if (v == null || v == ego) continue;
            sumX  += v.GetPosition().x;
            sumVx += v.GetVx();
            n++;
        }
        if (n > 0)
        {
            float platoonX  = sumX  / n;
            float platoonVx = sumVx / n;

            if (autoDetectLane)
            {
                // Snap the platoon's average lateral position to the nearest lane centre,
                // then express it in solver convention (pos.x - _x0).
                // _x0 = ego.GetPosition().x + ego.GetY()  (since GetY() = _x0 - pos.x)
                float x0          = ego.GetPosition().x + ego.GetY();
                float lanecentre   = Mathf.Round(platoonX / laneWidth) * laneWidth;
                platoonLaneY      = lanecentre - x0;
                Debug.Log($"[NashCoordinator] platoonLaneY auto = {platoonLaneY:F2} " +
                          $"(platoonX={platoonX:F2} → lanecentre={lanecentre:F2}, x0={x0:F2})");
            }
            else
            {
                Debug.Log($"[NashCoordinator] platoonLaneY manual = {platoonLaneY:F2} (autoDetect off)");
            }

            // Warm-start: if platoon is significantly faster, seed _u1Prev/_u2Prev at
            // 2.0 m/s² so the first Nash step isn't jerk-capped at Du1Max*dt ≈ 0.15 m/s².
            _longNash.WarmStartPrev(platoonVx > ego.GetVx() + 5f ? 2.0f : ego.GetAx());
        }
        else
        {
            _longNash.WarmStartPrev(ego.GetAx());
        }

        NashActive = true;
    }

    public void DisableNash()
    {
        NashActive = false;
        _longNash.Reset();
        Phase           = MergePhase.Approach;
        _mergeLocked    = false;
        _mobilApproved  = false;
        _longTimer      = 0f;
        _latTimer       = 0f;
        _phaseHoldTimer = 0f;
        Debug.Log("[NashCoordinator] Nash disabled — returning to manual control.");
    }

    void FixedUpdate()
    {
        if (!NashActive)  return;
        if (ego == null || platoonManager == null) return;

        float dt      = Time.fixedDeltaTime;
        var   platoon = platoonManager.GetPlatoonVehicles();
        float t       = Time.fixedTime;

        // ── Safety override (100 Hz) ───────────────────────────────────────────
        // Runs every physics step regardless of Nash timer so the 10 Hz Nash lag
        // cannot allow the ego to pass through a platoon vehicle.
        // Triggers when bumper-to-bumper gap < LongMinSafeDist (hard-brake zone).
        var overrideLeader = FindLeader(platoon);
        if (overrideLeader != null)
        {
            float overrideGap = overrideLeader.GetPosition().z
                              - ego.GetPosition().z
                              - overrideLeader.GetLength();
            if (overrideGap < settings.SafetyField.LongMinSafeDist)
            {
                ego.NashBrake      = 1f;
                ego.NashThrottle   = 0f;
                ego.NashModeActive = true;
                Data.SafetyOverride = true;
            }
            else
            {
                Data.SafetyOverride = false;
            }
        }

        UpdatePhase(platoon, t, dt);

        // Long Nash @ NashDtLong
        _longTimer += dt;
        if (_longTimer >= settings.Timing.NashDtLong)
        {
            RunLongNashStep(platoon);
            _longTimer = 0f;
        }

        // Lat Nash @ NashDtLat — GAP_SEARCH, MERGE, FOLLOWING.
        // In GAP_SEARCH the lateral target is the ego's current lane (straight-ahead),
        // not the platoon lane — this prevents the tug-of-war from yErr≈-2m while still
        // guarding against dangerous heading excursions or opposite-lane drift.
        if (Phase == MergePhase.GapSearch ||
            Phase == MergePhase.Merge     ||
            Phase == MergePhase.Following)
        {
            _latTimer += dt;
            if (_latTimer >= settings.Timing.NashDtLat)
            {
                RunLatNashStep(platoon);
                _latTimer = 0f;
            }
        }
    }

    // =========================================================================
    // Longitudinal Nash step
    // =========================================================================

    void RunLongNashStep(AdvancedBicycleModel[] platoon)
    {
        var leader   = _mergeLocked ? _lockedLeader   : FindLeader(platoon);
        var follower = _mergeLocked ? _lockedFollower : null;

        // Re-linearise with vehicle physical Jacobian when vx drifts > 0.5 m/s
        _longNash.UpdateLinearization(ego, settings.Timing.NashDtLong);

        // Safety force (Li et al. DSF with dynamic params)
        float force = _longSafety.Compute(ego, leader, follower, Phase);
        _longForceEma = force;   // _longSafety already applies EMA internally

        // Gap and velocity errors for authority
        float gapErr = leader != null ? LongGapError(leader) : 0f;
        float velErr = leader != null ? ego.GetVx() - leader.GetVx() : 0f;
        LongLambda = _authority.ComputeLong(force, gapErr, velErr, Phase);

        // Adaptive R2: EMA from R2LongStart (far) → R2LongFollow (integrated)
        // l_n ∈ [0,1]: 0 = far from platoon, 1 = fully synchronised
        float l_n_gap  = Mathf.Clamp01(1f - Mathf.Abs(gapErr) / settings.Authority.AuthGapErrorMax);
        float l_n_vel  = Mathf.Clamp01(1f - Mathf.Abs(velErr) / settings.Authority.AuthVelErrorMax);
        float l_n      = Mathf.Min(l_n_gap, l_n_vel);
        float r2LTarget = settings.Nash.R2LongStart
                        + (settings.Nash.R2LongFollow - settings.Nash.R2LongStart) * l_n;
        _r2LongCurrent  = settings.Nash.R2LongEmaAlpha * r2LTarget
                        + (1f - settings.Nash.R2LongEmaAlpha) * _r2LongCurrent;
        // Activity-based R2 scaling (Lazcano et al. 2021): when driver presses pedals,
        // reduce R2 to amplify u2 — gives perceptible authority during intentional inputs.
        float pedalThr = VehicleInputs.Instance != null ? VehicleInputs.Instance.ThrottleInput : 0f;
        float pedalBrk = VehicleInputs.Instance != null ? VehicleInputs.Instance.BrakeInput    : 0f;
        float longActivity = Mathf.Max(pedalThr, pedalBrk);
        float longActScale = 1f - Mathf.Clamp01(longActivity / 0.3f)
                                * (1f - settings.Nash.R2ActivityMinScale);
        float r2LongEffective = _r2LongCurrent * longActScale;
        if (Mathf.Abs(r2LongEffective - _r2LongPrev) > settings.Nash.R2LongUpdateThreshold)
        {
            _longNash.UpdateR2(r2LongEffective);
            _r2LongPrev = r2LongEffective;
        }

        // Reference trajectories
        float[] r1 = _longRef.SystemRef(
            ego, leader,
            platoonManager.GetTargetVelocity(),
            settings.Timing.NashNp, settings.Timing.NashDtLong);

        float[] r2 = _longRef.HumanRef(
            ego, _lockedLeader, Phase, platoonManager.GetTargetVelocity(),
            settings.Timing.NashNp, settings.Timing.NashDtLong);

        // Nash solve
        float[] x0 = ego.GetLongState();
        var (u1, u2) = _longNash.SolveNashEquilibrium(x0, r1, r2, LongLambda);
        float uShared = u1 + u2;

        // Inject: acceleration → (throttle, brake) via vehicle's inverse engine map
        var (thr, brk) = ego.AccelerationToInputs(uShared);
        ego.NashThrottle   = thr;
        ego.NashBrake      = brk;
        ego.NashModeActive = true;

        // During APPROACH / GAP_SEARCH the lateral Nash is not yet running.
        // Pass driver steering through so NashModeActive=true doesn't zero-out the wheel.
        if (Phase != MergePhase.Merge && Phase != MergePhase.Following)
            ego.NashSteerNorm = VehicleInputs.Instance != null
                              ? VehicleInputs.Instance.SteeringInput : 0f;

        float diagGap = _longSafety.LastGap;
        float diagTTC = _longSafety.LastTTC;
        float diagRaw = _longSafety.LastRaw;
        float r1VTarget = r1.Length > 1 ? r1[1] : 0f;
        float r2VTarget = r2.Length > 1 ? r2[1] : 0f;

        Debug.Log($"[Nash] vx={ego.GetVx():F2} u1={u1:F3} u2={u2:F3} u={uShared:F3} " +
                  $"r1v={r1VTarget:F2} r2v={r2VTarget:F2} " +
                  $"gap={diagGap:F1} ttc={diagTTC:F1} F={diagRaw:F0}/{force:F0} " +
                  $"λ={LongLambda:F2} thr={thr:F3} brk={brk:F3} phase={Phase}");

        Data.ULong      = uShared;  Data.U1Long    = u1;       Data.U2Long  = u2;
        Data.LongLambda = LongLambda; Data.LongForce = force;  Data.Phase   = Phase;
        Data.GapToLeader  = diagGap;
        Data.TTC          = diagTTC;
        Data.LongForceRaw = diagRaw;
        Data.R1VTarget    = r1VTarget;
        Data.R2VTarget    = r2VTarget;
        Data.VxEgo        = ego.GetVx();
        Data.GapErr       = gapErr;
        Data.VelErr       = velErr;
        Data.LnSync       = l_n;
        Data.R2LongEff    = r2LongEffective;
        Data.LongActivity = longActivity;
    }

    // =========================================================================
    // Lateral Nash step (active only when LateralNashSolver is set from Plan 6)
    // =========================================================================

    void RunLatNashStep(AdvancedBicycleModel[] platoon)
    {
        var lat = EnsureLatNash();

        float force = _latSafety.Compute(ego, platoon);
        _latForceEma = force;

        // During GAP_SEARCH use the ego's current lateral position as target so Nash
        // keeps the vehicle driving straight (psi→0) without fighting a 2m lane error.
        // During MERGE/FOLLOWING use the actual platoon lane target.
        float latTarget = (Phase == MergePhase.GapSearch)
                        ? -ego.GetY()   // solver convention: current posX − _x0
                        : platoonLaneY;

        // yErr in solver convention: positive = ego right of target (needs to go left)
        float yErr = -ego.GetY() - latTarget;
        LatLambda = _authority.ComputeLat(force, yErr);

        // Adaptive R2 lateral: l_n based on lane-width normalised y-error
        float halfLane  = settings.SafetyField.LaneWidth * 0.5f;
        float l_n_lat   = Mathf.Clamp01((halfLane - Mathf.Abs(yErr)) / Mathf.Max(halfLane, 1e-3f));
        float r2LaTarget = settings.Nash.R2LatStart
                         + (settings.Nash.R2LatFollow - settings.Nash.R2LatStart) * l_n_lat;
        _r2LatCurrent    = settings.Nash.R2LatEmaAlpha * r2LaTarget
                         + (1f - settings.Nash.R2LatEmaAlpha) * _r2LatCurrent;

        // Conflict detection (moving average over last 5 Nash ticks)
        float conflictMav = 0f;
        for (int ci = 0; ci < ConflictBufLen; ci++) conflictMav += _conflictBuf[ci];
        conflictMav /= ConflictBufLen;

        // Activity-based R2 scaling (Lazcano et al. 2021): when driver actively steers,
        // reduce R2_lat to amplify u2 — gives perceptible lateral authority.
        // During MERGE with active conflict, skip activity reduction: a resisting driver
        // should not receive amplified authority — conflict boost handles suppression instead.
        float latSteering = VehicleInputs.Instance != null
                          ? Mathf.Abs(VehicleInputs.Instance.SteeringInput) : 0f;
        float latActScale;
        if (Phase == MergePhase.Merge && conflictMav > 0.2f)
            latActScale = 1f;   // no R2 reduction during contested merge
        else
            latActScale = 1f - Mathf.Clamp01(latSteering / 0.3f)
                              * (1f - settings.Nash.R2LatActivityMinScale);
        float r2LatEffective = _r2LatCurrent * latActScale;

        // Conflict boost: during MERGE if the driver resists (u1 and u2 opposing),
        // boost R2 so P2 penalises u2 more, suppressing the opposing human input.
        // In FOLLOWING u1≈0 → buffer stays 0 → no boost.
        float conflictBoost = conflictMav <= 0.2f ? 0f
                            : conflictMav >= 0.8f ? ConflictBoostMag
                            : ConflictBoostMag * (conflictMav - 0.2f) / 0.6f;
        r2LatEffective *= (1f + conflictBoost);

        if (Mathf.Abs(r2LatEffective - _r2LatPrev) > settings.Nash.R2LatUpdateThreshold)
        {
            lat.UpdateR2(r2LatEffective);
            _r2LatPrev = r2LatEffective;
        }

        // Re-linearise bicycle model if vx drifted (Coriolis sensitivity)
        lat.UpdateLinearization(ego);

        float[] r1 = _latRef.SystemRef(
            ego, latTarget, Phase,
            Time.fixedTime,
            settings.Timing.NashNp, settings.Timing.NashDtLat);

        float[] r2 = _latRef.HumanRef(
            ego,
            latTarget, Phase, Time.fixedTime,
            settings.Timing.NashNp, settings.Timing.NashDtLat);

        float u1, u2;
        try
        {
            float[] x0 = ego.GetLatStateVector();
            (u1, u2) = lat.SolveNashEquilibrium(x0, r1, r2, LatLambda);
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[NashCoordinator] LatNash solve failed: {ex.Message}");
            u1 = -0.1f * yErr;   // P-controller fallback
            u2 = 0f;
        }

        float deltaShared = u1 + u2;
        deltaShared = Mathf.Clamp(deltaShared,
            settings.Nash.DeltaMin, settings.Nash.DeltaMax);

        // Record conflict to MAV buffer (1 = opposing directions, 0 = aligned).
        _conflictBuf[_conflictIdx] = (Mathf.Abs(u1) > 1e-4f && Mathf.Abs(u2) > 1e-4f &&
                                      Mathf.Sign(u1) != Mathf.Sign(u2)) ? 1f : 0f;
        _conflictIdx = (_conflictIdx + 1) % ConflictBufLen;

        // Inject: delta [rad] → normalised steer [-1,1]
        float maxWheelRad        = ego.GetVehicleParameters().maxWheelAngle * Mathf.Deg2Rad;
        ego.NashSteerNorm        = Mathf.Clamp(deltaShared / maxWheelRad, -1f, 1f);
        ego.NashModeActive       = true;

        Debug.Log($"[Nash-Lat] y={ego.GetY():F3} yErr={yErr:F3} r1ψ0={r1[1]*Mathf.Rad2Deg:F2}° r2ψ0={r2[1]*Mathf.Rad2Deg:F2}° u1={u1:F4} u2={u2:F4} delta={deltaShared:F4} steer={ego.NashSteerNorm:F4} λ={LatLambda:F3} phase={Phase}");

        Data.DeltaLat  = deltaShared; Data.U1Lat   = u1;   Data.U2Lat   = u2;
        Data.LatLambda = LatLambda;   Data.LatForce = force;
        Data.YErr        = yErr;
        Data.ConflictMav = conflictMav;
        Data.R2LatEff    = r2LatEffective;
        Data.LatActivity = latSteering;
        Data.LnLatSync   = l_n_lat;
    }

    // =========================================================================
    // Phase machine (mirrors coordinator._update_phase)
    // =========================================================================

    void UpdatePhase(AdvancedBicycleModel[] platoon, float t, float dt)
    {
        switch (Phase)
        {
            case MergePhase.Approach:
                // Run MOBIL only when ego is close enough to the platoon
                if (NearestPlatoonDist(platoon) < approachCheckDist)
                {
                    if (!_mobilApproved)
                        _mobilApproved = _mobil.CheckPlatoonMerge(ego, platoon);

                    if (_mobilApproved)
                    {
                        LockMergePosition(platoon);
                        Phase           = MergePhase.GapSearch;
                        _gapSearchStart = t;
                        Debug.Log($"[NashCoordinator] APPROACH → GAP_SEARCH at t={t:F1}s — {_mobil.LastStatus}");
                    }
                }
                break;

            case MergePhase.GapSearch:
                if (t - _gapSearchStart >= 2.0f)   // GAP_SEARCH_DURATION = 2.0 s (was 0.5 — gives Nash time to damp psi before Merge entry)
                {
                    // Notify lat ref generator of MERGE entry for locked polynomial
                    _latRef.OnMergeEntry(-ego.GetY(), -ego.GetYDot(), ego.GetPsi(), ego.GetVx(), platoonLaneY, t);
                    Phase = MergePhase.Merge;
                    Debug.Log($"[NashCoordinator] GAP_SEARCH → MERGE at t={t:F1}s");
                }
                break;

            case MergePhase.Merge:
                if (IsMergeComplete(platoon))
                {
                    _phaseHoldTimer += dt;
                    if (_phaseHoldTimer >= settings.Timing.PhaseLockTime)
                    {
                        _latRef.OnFollowingEntry(t);
                        Phase           = MergePhase.Following;
                        _phaseHoldTimer = 0f;
                        _latNash?.Reset();   // clear warm-start from Merge so Following starts fresh
                        Debug.Log($"[NashCoordinator] MERGE → FOLLOWING at t={t:F1}s");
                    }
                }
                else
                {
                    _phaseHoldTimer = 0f;
                }
                break;

            case MergePhase.Following:
                break;
        }
    }

    bool IsMergeComplete(AdvancedBicycleModel[] platoon)
    {
        bool yOk  = Mathf.Abs(-ego.GetY() - platoonLaneY) < 0.3f;
        bool psiOk = Mathf.Abs(ego.GetPsi())               < 0.05f;

        var leader = _mergeLocked ? _lockedLeader : FindLeader(platoon);
        bool gapOk = true;
        if (leader != null)
        {
            float gap    = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength();
            float desGap = settings.SafetyField.StandstillDist
                         + settings.SafetyField.PlatoonTimeGap * ego.GetVx();
            gapOk = Mathf.Abs(gap - desGap) / Mathf.Max(desGap, 1e-3f) < 0.20f;
        }
        return yOk && psiOk && gapOk;
    }

    // =========================================================================
    // Utility
    // =========================================================================

    float LongGapError(AdvancedBicycleModel leader)
    {
        float gap    = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength();
        float desGap = settings.SafetyField.StandstillDist
                     + settings.SafetyField.PlatoonTimeGap * ego.GetVx();
        return desGap - gap;   // positive = too close
    }

    float NearestPlatoonDist(AdvancedBicycleModel[] platoon)
    {
        float minD = float.MaxValue;
        foreach (var v in platoon)
        {
            if (v == null) continue;
            float d = Mathf.Abs(v.GetPosition().z - ego.GetPosition().z);
            if (d < minD) minD = d;
        }
        return minD;
    }

    void LockMergePosition(AdvancedBicycleModel[] platoon)
    {
        _lockedLeader   = FindLeader(platoon);
        _lockedFollower = FindFollower(platoon);
        _mergeLocked    = true;
        string lName = _lockedLeader   != null ? _lockedLeader.name   : "None";
        string fName = _lockedFollower != null ? _lockedFollower.name : "None";
        Debug.Log($"[NashCoordinator] Merge locked — leader={lName}, follower={fName}");
    }

    AdvancedBicycleModel FindLeader(AdvancedBicycleModel[] platoon)
    {
        AdvancedBicycleModel best = null;
        float minD = float.MaxValue;
        float egoZ = ego.GetPosition().z;
        foreach (var v in platoon)
        {
            if (v == null) continue;
            float d = v.GetPosition().z - egoZ;
            if (d > 0f && d < minD) { minD = d; best = v; }
        }
        return best;
    }

    AdvancedBicycleModel FindFollower(AdvancedBicycleModel[] platoon)
    {
        AdvancedBicycleModel best = null;
        float maxD = float.MinValue;
        float egoZ = ego.GetPosition().z;
        foreach (var v in platoon)
        {
            if (v == null) continue;
            float d = v.GetPosition().z - egoZ;
            if (d < 0f && d > maxD) { maxD = d; best = v; }
        }
        return best;
    }
}
