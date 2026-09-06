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
    public NashPlatoonSettings Settings => settings;

    [Header("Platoon geometry")]
    [SerializeField] float platoonLaneY      = 3.5f;    // PLATOON_LANE_Y (solver convention: pos.x - ego._x0)
    [SerializeField] bool  autoDetectLane    = true;  // if true, overrides platoonLaneY on J-press with snapped lane centre
    [SerializeField] float laneWidth         = 3.5f;  // road lane width [m] — used for lane-centre snapping (overridden from RoadArchitect if roadRef set)
    [SerializeField] RoadArchitect.Road roadRef;      // optional — when set, laneWidth is read from Road.laneWidth at Awake
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
    float _qyPrev; // last Q_y sent to UpdateQ1 — for threshold gating

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

    // ── Lateral warm-start: seed solver from driver state on first tick ──────
    bool _latWarmStartDone;
    bool _latGsRApplied;   // true once GapSearch R1/R2 override has been pushed to solver

    // ── Phase hold state ──────────────────────────────────────────────────────
    float _gapSearchStart = -1f;
    float _phaseHoldTimer;

    // ── Merge→Following convergence tracking ─────────────────────────────────
    float _prevGapErr;
    float _prevGapErrTime;

    // ── Following→Merge hysteresis ────────────────────────────────────────────
    float _followingDivTimer;
    float _prevGap;
    float _prevGapTime;

    // ── Locked merge partners (set at GAP_SEARCH→MERGE) ──────────────────────
    AdvancedBicycleModel _lockedLeader;
    AdvancedBicycleModel _lockedFollower;
    bool                 _mergeLocked;

    // ── Virtual leader (Kennedy 2023) — active in Following when no real leader ──
    readonly VirtualLeaderVehicle _virtualLeader = new VirtualLeaderVehicle();

    // ── Online IDM driver calibrator (Zhang & Sun 2024) ──────────────────────
    LongitudinalDriverCalibrator _driverCalib;

    // ── GapSearch lateral hold: y position locked at J-press ─────────────────
    float _gsLockedY;

    // =========================================================================
    // Unity lifecycle
    // =========================================================================

    void Awake()
    {
        SimCfg.I    = settings;

        // Pull lane width from RoadArchitect so scene geometry is the single source of truth.
        // Falls back to explicit reference → scene lookup → serialized default.
        if (roadRef == null)
            roadRef = FindObjectOfType<RoadArchitect.Road>();
        if (roadRef != null)
        {
            laneWidth = roadRef.laneWidth;
            Debug.Log($"[NashCoordinator] laneWidth = {laneWidth:F2} m (from RoadArchitect.Road '{roadRef.name}', laneAmount={roadRef.laneAmount})");
        }
        else
        {
            Debug.LogWarning($"[NashCoordinator] No RoadArchitect.Road found — using serialized laneWidth={laneWidth:F2} m");
        }

        settings.SafetyField.LaneWidth = laneWidth;
        _longSafety = new LongitudinalSafetyField(settings.SafetyField);
        _latSafety  = new LateralSafetyField(settings.SafetyField);
        _authority  = new AuthorityAllocator(settings.Authority, settings.SafetyField);
        _longRef    = new LongitudinalReferenceGen(settings.RefGen, settings.SafetyField);
        _latRef     = new LateralReferenceGen(settings.RefGen);
        _mobil      = new MOBILLaneChange();
        _mobil.SetPoliteness("normal");

        _r2LongCurrent = _r2LongPrev = settings.Nash.R2LongStart;
        _r2LatCurrent  = _r2LatPrev  = settings.Nash.R2LatStart;

        var rc = settings.RefGen;
        var sf = settings.SafetyField;
        _driverCalib = new LongitudinalDriverCalibrator(
            s0:        rc.IdmS0,
            aMax:      rc.IdmAMax,
            fallbackT: sf.PlatoonTimeGap,
            v0:        sf.LongVelocityReference / 3.6f);
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
        // Reset calibrator at start of each merge attempt (per-run calibration)
        _driverCalib.Reset(tInit: settings.SafetyField.PlatoonTimeGap);

        var platoon = platoonManager.GetPlatoonVehicles();

        // Detect platoon lane from current platoon vehicles (excluding ego)
        float sumX = 0f; int n = 0;
        foreach (var v in platoon)
        {
            if (v == null || v == ego) continue;
            sumX += v.GetPosition().x;
            n++;
        }
        if (n > 0)
        {
            float platoonX = sumX / n;
            if (autoDetectLane)
            {
                float x0         = ego.GetPosition().x + ego.GetY();
                float lanecentre = Mathf.Round(platoonX / laneWidth) * laneWidth;
                platoonLaneY     = lanecentre - x0;
                Debug.Log($"[NashCoordinator] platoonLaneY auto = {platoonLaneY:F2} " +
                          $"(platoonX={platoonX:F2} → lanecentre={lanecentre:F2}, x0={x0:F2})");
            }
            else
            {
                Debug.Log($"[NashCoordinator] platoonLaneY manual = {platoonLaneY:F2} (autoDetect off)");
            }
        }

        // Lock merge position at J-press time so insertion index is correct
        // before the platoon moves further ahead.
        LockMergePosition(platoon);

        // Join platoon between locked leader and follower
        platoonManager.JoinPlatoon(_lockedLeader, _lockedFollower);

        _longNash.WarmStartPrev(ego.GetAx());

        // Lock lateral position at J-press — GapSearch holds the vehicle straight at this y.
        _gsLockedY = -ego.GetY();  // solver convention
        _latRef.OnGapSearchEntry(_gsLockedY, -ego.GetYDot(), ego.GetPsi(), ego.GetPsiDot(), ego.GetVx(), Time.fixedTime);

        // Activate virtual leader immediately if ego has no real leader (joins at front).
        // This gives GapSearch and Merge a valid longitudinal reference from the start.
        if (_lockedLeader == null)
        {
            var platVehicles = platoonManager.GetPlatoonVehicles();
            float initVx = (platVehicles != null && platVehicles.Length > 0)
                         ? platVehicles[0].GetVx()
                         : platoonManager.GetTargetVelocity();
            ego.GetPhysicalAccelBounds(out float vAMin, out float vAMax);
            _virtualLeader.Activate(
                ego.GetPosition().z, ego.GetLength(),
                initVx,
                platoonManager.GetTargetVelocity(),
                settings.SafetyField.StandstillDist,
                settings.SafetyField.PlatoonTimeGap,
                vAMax, -vAMin);
        }

        NashActive       = true;
        Phase            = MergePhase.GapSearch;
        _gapSearchStart  = Time.fixedTime;
        _latGsRApplied   = false;
        Debug.Log($"[NashCoordinator] APPROACH → GAP_SEARCH at t={Time.fixedTime:F1}s (J pressed) | " +
                  $"Calibrator end-of-Approach: {_driverCalib.GetStatusString()}");
    }

    public void DisableNash()
    {
        NashActive = false;
        _longNash.Reset();
        _virtualLeader.Deactivate();
        Phase              = MergePhase.Approach;
        _mergeLocked       = false;
        _mobilApproved     = false;
        _longTimer         = 0f;
        _latTimer          = 0f;
        _phaseHoldTimer    = 0f;
        _latWarmStartDone  = false;
        _latGsRApplied     = false;
        _prevGapErr        = 0f;
        _prevGapErrTime    = 0f;
        _followingDivTimer = 0f;
        _prevGap           = 0f;
        _prevGapTime       = 0f;
        Debug.Log("[NashCoordinator] Nash disabled — returning to manual control.");
    }

    void FixedUpdate()
    {
        if (!NashActive)  return;
        if (ego == null || platoonManager == null) return;

        float dt      = Time.fixedDeltaTime;
        var   platoon = platoonManager.GetPlatoonVehicles();
        float t       = Time.fixedTime;

        // Advance virtual leader every physics frame using FreeRoadAcceleration
        if (_virtualLeader.IsActive)
            _virtualLeader.Step(dt);

        // ── Safety override (100 Hz) — DISABLED FOR TESTING ─────────────────────
        // Commented out to evaluate whether Nash alone handles collision avoidance.
        // Known issue: FindLeader returns any vehicle ahead (including overtaking platoon
        // vehicles during GapSearch), triggering full brakes on non-threatening vehicles.
        // Re-enable and fix with: _mergeLocked ? _lockedLeader : FindLeader(platoon)
        // before returning to production.
        /*
        var overrideLeader = FindLeader(platoon);
        if (overrideLeader != null)
        {
            float overrideGap = overrideLeader.GetPosition().z
                              - ego.GetPosition().z
                              - overrideLeader.GetLength() / 2f - ego.GetLength() / 2f;
            if (overrideGap < settings.SafetyField.ComputeMinSafeDist(ego.GetVx()))
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
        */
        Data.SafetyOverride = false;

        UpdatePhase(platoon, t, dt);

        // Seed longitudinal solver state from physical vehicle acceleration during Approach.
        // GapSearch is excluded: Nash is already running there and manages _u1Prev itself.
        // Must run after UpdatePhase so the GapSearch→Merge transition also gets a correct seed.
        if (Phase == MergePhase.Approach)
            _longNash.WarmStartPrev(ego.GetAx());

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
        // During Approach: keep warm-start updated so solver starts from exact driver state.
        if (Phase == MergePhase.Approach)
        {
            EnsureLatNash().WarmStartFromDriverState(ego.GetLambda(), ego.GetALat());
        }
        else if (Phase == MergePhase.GapSearch ||
                 Phase == MergePhase.Merge     ||
                 Phase == MergePhase.Following)
        {
            _latTimer += dt;
            if (_latTimer >= settings.Timing.NashDtLat)
            {
                if (!_latWarmStartDone)
                {
                    EnsureLatNash().WarmStartFromDriverState(ego.GetLambda(), ego.GetALat());
                    _latWarmStartDone = true;
                }
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

        // When ego has no real leader (joined at front), use the virtual leader across
        // all phases (GapSearch, Merge, Following). Activated in EnableNash() when
        // _lockedLeader == null, so this flag is true from J-press onward.
        bool usingVirtual = (leader == null && _virtualLeader.IsActive);

        // Re-linearise with vehicle physical Jacobian when vx drifts > 0.5 m/s
        _longNash.UpdateLinearization(ego, settings.Timing.NashDtLong);

        // Safety force (Li et al. DSF with dynamic params).
        // When using the virtual leader, pass its position and velocity via the
        // dedicated overload so LeaderForceRaw receives a valid gap and TTC.
        float force;
        if (usingVirtual)
            force = _longSafety.ComputeWithVirtual(ego, _virtualLeader, follower, Phase);
        else
            force = _longSafety.Compute(ego, leader, follower, Phase);
        _longForceEma = force;

        // Gap and velocity errors for authority.
        float targetVel = platoonManager.GetTargetVelocity();
        float gapErr, velErr;
        if (usingVirtual)
        {
            // Virtual leader errors — converge to zero when vx == targetVelocity
            // and gap == desGap(v*), as required by Kennedy Sec. V-A.
            gapErr = _virtualLeader.GapError(
                settings.SafetyField.StandstillDist,
                settings.SafetyField.PlatoonTimeGap,
                ego.GetPosition().z, ego.GetLength(), ego.GetVx());
            velErr = _virtualLeader.VelocityError(ego.GetVx());
        }
        else if (leader != null)
        {
            gapErr = LongGapError(leader);
            velErr = ego.GetVx() - leader.GetVx();
        }
        else
        {
            gapErr = 0f;
            velErr = ego.GetVx() - targetVel;
        }
        LongLambda = _authority.ComputeLong(force, gapErr, velErr, Phase);

        // Cap λ when there is truly no reference at all (not even a virtual leader).
        if (leader == null && !usingVirtual)
            LongLambda = Mathf.Min(LongLambda, settings.Authority.LongLambdaMaxNoLeader);

        // Adaptive R2: EMA from R2LongStart (far) → R2LongFollow (integrated)
        // l_n ∈ [0,1]: 0 = far from platoon, 1 = fully synchronised
        // When no leader, use a wider vel-error normalisation so a large speed
        // overshoot (e.g. 12 m/s) produces a gradual l_n decay, not an instant drop to 0.
        // Virtual leader counts as a real reference for velErrMax normalisation.
        float velErrMax = (leader != null || usingVirtual)
                        ? settings.Authority.AuthVelErrorMax
                        : settings.Authority.AuthVelErrorMaxNoLeader;
        float l_n_gap  = Mathf.Clamp01(1f - Mathf.Abs(gapErr) / settings.Authority.AuthGapErrorMax);
        float l_n_vel  = Mathf.Clamp01(1f - Mathf.Abs(velErr) / velErrMax);
        float l_n      = Mathf.Min(l_n_gap, l_n_vel);
        float r2LTarget = settings.Nash.R2LongStart
                        + (settings.Nash.R2LongFollow - settings.Nash.R2LongStart) * l_n;
        _r2LongCurrent  = settings.Nash.R2LongEmaAlpha * r2LTarget
                        + (1f - settings.Nash.R2LongEmaAlpha) * _r2LongCurrent;
        // Activity-based R2 scaling (Lazcano et al. 2021): when driver presses pedals,
        // reduce R2 to amplify u2 — gives perceptible authority during intentional inputs.
        // Phase-dependent min scale: Merge=1.0 (system-led join), Following=0.5 (driver-led).
        float pedalThr = VehicleInputs.Instance != null ? VehicleInputs.Instance.ThrottleInput : 0f;
        float pedalBrk = VehicleInputs.Instance != null ? VehicleInputs.Instance.BrakeInput    : 0f;
        float longActivity = Mathf.Max(pedalThr, pedalBrk);
        float actMinScale = (Phase == MergePhase.Merge || Phase == MergePhase.GapSearch)
                          ? settings.Nash.R2ActivityMinScale_Merge
                          : settings.Nash.R2ActivityMinScale;
        float longActScale = 1f - Mathf.Clamp01(longActivity / 0.3f)
                                * (1f - actMinScale);
        float r2LongEffective = _r2LongCurrent * longActScale;
        if (Mathf.Abs(r2LongEffective - _r2LongPrev) > settings.Nash.R2LongUpdateThreshold)
        {
            _longNash.UpdateR2(r2LongEffective);
            _r2LongPrev = r2LongEffective;
        }

        // Physical bounds needed both for reference clamping and solver constraints
        ego.GetPhysicalAccelBounds(out float aMin, out float aMax);

        // ── Online driver calibration (Zhang & Sun 2024) ──────────────────────
        // GapSearch/Merge: IDM inversion (valid when |dv|<2, vx<v0).
        // Following: direct T=(gap-s0)/vx — uses the gap the driver actually holds.
        //   No circular dependency: HumanRef is computed independently of Nash output.
        if (leader != null)
        {
            float dv = ego.GetVx() - leader.GetVx();
            if (Phase == MergePhase.GapSearch || Phase == MergePhase.Merge)
                _driverCalib.UpdateGapSearch(ego.GetVx(), _longSafety.LastGap, dv);
            else if (Phase == MergePhase.Following)
                _driverCalib.UpdateFollowing(ego.GetVx(), _longSafety.LastGap, dv);
        }

        // Phase-aware R1:
        //   GapSearch (vel_err small) / Merge : R1_long_Merge (10000) — closes gap error aggressively
        //   Following                         : R1_long (75000)       — quiet steady-state, less chattering
        if (Phase == MergePhase.Following)
        {
            _longNash.UpdateR1(settings.Nash.R1_long);
        }
        else if (Phase == MergePhase.GapSearch &&
                 Mathf.Abs(velErr) < settings.RefGen.CatchupVelThreshold)
        {
            _longNash.UpdateR1(settings.Nash.R1_long_Merge);
        }

        // Reference trajectories
        float[] r1 = _longRef.SystemRef(
            ego, leader,
            platoonManager.GetTargetVelocity(),
            Phase,
            settings.Timing.NashNp, settings.Timing.NashDtLong,
            usingVirtual ? _virtualLeader : null);

        float[] r2 = _longRef.HumanRef(
            ego, _lockedLeader, Phase, platoonManager.GetTargetVelocity(),
            settings.Timing.NashNp, settings.Timing.NashDtLong,
            _driverCalib.T);

        // Tighten box constraints to physical vehicle limits before solving.
        // Du1Max is the per-step jerk limit passed to the QP solver.
        // Following: tight 0.5 m/s³ (ISO 15622:2018 §A.1 ACC comfort) — steady-state.
        // Merge, near steady-state (|vel_err|<0.5, |gap_err|<3m): 1.5 m/s³ — soft cap
        //   damps residual chattering while preserving enough authority for late corrections.
        // Merge/GapSearch, transient: full Du1Max (15) — emergency catchup, comfort secondary.
        float du1Max;
        if (Phase == MergePhase.Following)
            du1Max = 0.5f;
        else if (Phase == MergePhase.Merge &&
                 Mathf.Abs(velErr) < 0.5f &&
                 Mathf.Abs(gapErr) < 3f)
            du1Max = 1.5f;
        else
            du1Max = settings.Nash.Du1Max;
        float du1 = Mathf.Min(settings.Nash.Du1Max, du1Max) * settings.Timing.NashDtLong;
        _longNash.UpdatePhysicalBounds(aMin, aMax,
            du1,
            settings.Nash.Du2Max * settings.Timing.NashDtLong);

        // Nash solve
        float[] x0 = ego.GetLongState();
        var (u1, u2) = _longNash.SolveNashEquilibrium(x0, r1, r2, LongLambda);
        float uShared = Mathf.Clamp(u1 + u2, aMin, aMax);

        // Inject: acceleration → (throttle, brake) via vehicle's inverse engine map
        var (thr, brk) = ego.AccelerationToInputs(uShared);
        ego.NashThrottle   = thr;
        ego.NashBrake      = brk;
        ego.NashModeActive = true;

        // During APPROACH: lateral Nash not running → pass driver steering through.
        // During GAP_SEARCH: lateral Nash runs with latTarget=current lane (straight-keeping),
        // so its output already suppresses driver drift — do NOT pass SteeringInput through.
        if (Phase == MergePhase.Approach)
            ego.NashSteerNorm = VehicleInputs.Instance != null
                              ? VehicleInputs.Instance.SteeringInput : 0f;

        float diagGap = _longSafety.LastGap;
        float diagTTC = _longSafety.LastTTC;
        float diagTHW = _longSafety.LastTHW;
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
        Data.THW          = diagTHW;
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

        // GapSearch R1/R2 override — applied once after solver is ready.
        // Reduces effort penalty so Nash can apply enough δ to damp a large entry ψ.
        // Restored to default on the first Merge tick so lane-change effort is normal.
        if (Phase == MergePhase.GapSearch && !_latGsRApplied)
        {
            lat.UpdateR1(settings.Nash.R1_lat_GapSearch);
            lat.UpdateR2(settings.Nash.R2_lat_GapSearch);
            _latGsRApplied = true;
        }
        else if (Phase != MergePhase.GapSearch && _latGsRApplied)
        {
            lat.UpdateR1(settings.Nash.R1_lat);
            lat.UpdateR2(settings.Nash.R2_lat);
            _latGsRApplied = false;
        }

        // Keep HumanRef cubic origin in sync with ego position during GapSearch.
        if (Phase == MergePhase.GapSearch)
            _latRef.UpdateGapSearchOrigin(-ego.GetY(), -ego.GetYDot());

        // During GAP_SEARCH track the current ego y as the lateral target.
        // Goal: stabilise ψ→0 and vy→0 wherever the vehicle is, not return to _gsLockedY.
        // Chasing a fixed entry point when vy≠0 at J-press creates large yErr → large λ_lat
        // → the solver fights to pull back to a position already passed, amplifying ψ and
        // causing the vehicle to overshoot far to the opposite side (→ collision with Car3).
        // Tracking ego.GetY() keeps yErr≈0 so authority stays low and the solver focuses
        // on zeroing ψ/vy. Once stabilised the vehicle holds its current y naturally.
        // During MERGE/FOLLOWING use the actual platoon lane target.
        if (Phase == MergePhase.GapSearch)
            _gsLockedY = -ego.GetY();
        float latTarget = (Phase == MergePhase.GapSearch)
                        ? _gsLockedY
                        : platoonLaneY;

        float force = _latSafety.Compute(ego, platoon);
        _latForceEma = force;

        // yErr in solver convention: positive = ego right of target (needs to go left)
        float yErr = -ego.GetY() - latTarget;

        // Authority signal for ComputeLat — phase-dependent.
        //
        // GapSearch: latTarget tracks ego.GetY() so yErr≡0, meaning the lane-offset
        // sigmoid in ComputeLat gives no signal. DSF force is also irrelevant here
        // (it reflects physical repulsion from platoon vehicles in an adjacent lane,
        // not a lateral collision risk while ego is longitudinally behind the platoon).
        // Authority must instead come from ψ: large heading → large λ so Nash damps it fast.
        // Encode ψ as an equivalent yErr: at vx*sin(ψ)≈vx*ψ lateral speed, the vehicle
        // will drift vx*ψ*T metres in T seconds — pass that as the yErr surrogate.
        // Force surrogate is set to 0 so only the performance sigmoid drives λ.
        //
        // Following: quadratic fade on DSF force so λ→0 when already centred.
        float forceForAuth = force;
        float yErrForAuth  = yErr;
        if (Phase == MergePhase.GapSearch)
        {
            forceForAuth = 0f;
            // Projected lateral drift over NashDtLat: surrogate yErr that grows with ψ
            float psiSurrogate = ego.GetVx() * Mathf.Sin(ego.GetPsi()) * settings.Timing.NashDtLat;
            yErrForAuth = psiSurrogate;
        }
        else if (Phase == MergePhase.Following)
        {
            float latThr  = laneWidth * 0.5f;
            float beyondS = Mathf.Min(1f, (Mathf.Abs(yErr) / latThr) * (Mathf.Abs(yErr) / latThr));
            forceForAuth = force * beyondS;
        }

        LatLambda = _authority.ComputeLat(forceForAuth, yErrForAuth);

        // Quarter/half-lane geometry
        float halfLane    = laneWidth * 0.5f;
        float quarterLane = halfLane * 0.5f;

        // beyondFrac: 0 inside deadband, ramps to 1 at ±halfLane.
        // In Following use FollowingYDeadband (3 cm) so even small lane offsets
        // from steering activity trigger Q_y/R2 correction — prevents permanent
        // u1/u2 cancellation when y_err is small but nonzero.
        // In Merge keep the wider quarterLane deadband for a softer transition.
        float beyondDeadband = (Phase == MergePhase.Following)
                             ? settings.Nash.FollowingYDeadband
                             : quarterLane;
        float beyondFrac = Mathf.Clamp01(
            (Mathf.Abs(yErr) - beyondDeadband) / Mathf.Max(halfLane - beyondDeadband, 1e-3f));

        // ── Adaptive R2 lateral ───────────────────────────────────────────────
        // Inside quarter-lane: R2=R2LatFree (large) → driver has authority.
        // Outside quarter-lane: R2 lerps toward R2LatStart (default large) while
        // Q rises (see below), making u1 stronger without stripping u2 entirely.
        // During Merge: use original l_n_lat ramp (approach/lane-change logic).
        float l_n_lat   = Mathf.Clamp01((halfLane - Mathf.Abs(yErr)) / Mathf.Max(halfLane, 1e-3f));
        float r2LaTarget;
        if (Phase == MergePhase.Following)
        {
            // R2 ramps from R2LatFree (inside deadband) → R2LatFollow (at halfLane).
            // R2LatFollow < R1_lat so system gains real authority for residual correction.
            // Previously used R2LatStart as endpoint (= R2LatFree = 250000) → no change → oscillation.
            r2LaTarget = Mathf.Lerp(settings.Nash.R2LatFree,
                                    settings.Nash.R2LatFollow,
                                    beyondFrac);
        }
        else
        {
            r2LaTarget = settings.Nash.R2LatStart
                       + (settings.Nash.R2LatFollow - settings.Nash.R2LatStart) * l_n_lat;
        }
        _r2LatCurrent = settings.Nash.R2LatEmaAlpha * r2LaTarget
                      + (1f - settings.Nash.R2LatEmaAlpha) * _r2LatCurrent;

        // Conflict detection (moving average over last 5 Nash ticks)
        float conflictMav = 0f;
        for (int ci = 0; ci < ConflictBufLen; ci++) conflictMav += _conflictBuf[ci];
        conflictMav /= ConflictBufLen;

        // ── Activity-based R2 scaling (Lazcano et al. 2021) ──────────────────
        // Active steering → reduce R2 to amplify u2 (driver feels control).
        // During contested Merge → no reduction (conflict boost handles suppression).
        float latSteering = VehicleInputs.Instance != null
                          ? Mathf.Abs(VehicleInputs.Instance.SteeringInput) : 0f;
        float latActScale;
        if (Phase == MergePhase.Merge && conflictMav > 0.2f)
            latActScale = 1f;
        else
            latActScale = 1f - Mathf.Clamp01(latSteering / 0.3f)
                              * (1f - settings.Nash.R2LatActivityMinScale);
        float r2LatEffective = _r2LatCurrent * latActScale;

        // ── Conflict boost (Merge only) ───────────────────────────────────────
        // Following boundary correction is handled by beyondFrac→R2+Q, not conflict boost.
        if (Phase == MergePhase.Merge)
        {
            float conflictBoost = conflictMav <= 0.2f ? 0f
                                : conflictMav >= 0.8f ? ConflictBoostMag
                                : ConflictBoostMag * (conflictMav - 0.2f) / 0.6f;
            r2LatEffective *= (1f + conflictBoost);
        }

        if (Mathf.Abs(r2LatEffective - _r2LatPrev) > settings.Nash.R2LatUpdateThreshold)
        {
            lat.UpdateR2(r2LatEffective);
            _r2LatPrev = r2LatEffective;
        }

        // ── Q1 adaptation for Following boundary ─────────────────────────────
        // Inside quarter-lane: Q_y stays at W default (low) → driver barely feels pull.
        // Outside quarter-lane: Q_y lerps to Q_y_BeyondBound (large) → both players
        // pay a high cost for the offset, so u1 grows and u2 naturally aligns.
        // Gated by Q_LatUpdateThreshold to avoid per-tick RebuildCostMatrices.
        if (Phase == MergePhase.Following)
        {
            float qyEff     = Mathf.Lerp(settings.Nash.Q_y,      settings.Nash.Q_y_BeyondBound,      beyondFrac);
            float qpsiEff   = Mathf.Lerp(settings.Nash.Q_psi,    settings.Nash.Q_psi_BeyondBound,    beyondFrac);
            float qyTermEff = Mathf.Lerp(settings.Nash.Q_y_terminal, settings.Nash.Q_yTerm_BeyondBound, beyondFrac);
            float qpTermEff = Mathf.Lerp(settings.Nash.Q_psi_terminal, settings.Nash.Q_psiTerm_BeyondBound, beyondFrac);
            if (Mathf.Abs(qyEff - _qyPrev) > settings.Nash.Q_LatUpdateThreshold)
            {
                lat.UpdateQ1(qyEff, qpsiEff, qyTermEff, qpTermEff);
                _qyPrev = qyEff;
            }
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
            lat.SetALat(ego.GetALat());
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
                // J-press triggers EnableNash() which locks position, joins platoon,
                // and transitions directly to GapSearch — nothing to do here.
                break;

            case MergePhase.GapSearch:
                // MOBIL checks for a safe lateral gap every Nash tick.
                // When approved AND longitudinal speed and gap are matched → begin Merge.
                if (NearestPlatoonDist(platoon) < approachCheckDist)
                {
                    if (!_mobilApproved)
                        _mobilApproved = _mobil.CheckPlatoonMerge(ego, platoon);
                }

                if (_mobilApproved)
                {
                    var gsLeader   = _mergeLocked ? _lockedLeader : FindLeader(platoon);
                    float gsVelErr = gsLeader != null ? gsLeader.GetVx() - ego.GetVx() : 0f;
                    bool velReady  = Mathf.Abs(gsVelErr) < settings.Timing.MergeVelThreshold;

                    bool gapReady = true;
                    if (gsLeader != null)
                    {
                        float gsGap    = gsLeader.GetPosition().z - ego.GetPosition().z
                                       - gsLeader.GetLength() * 0.5f - ego.GetLength() * 0.5f;
                        float gsDesGap = settings.SafetyField.StandstillDist
                                       + settings.SafetyField.PlatoonTimeGap * ego.GetVx();
                        gapReady = Mathf.Abs(gsGap - gsDesGap) / Mathf.Max(gsDesGap, 1f)
                                 < settings.Timing.MergeGapThreshold;
                    }

                    if (t - _gapSearchStart >= 2.0f && velReady && gapReady)
                    {
                        _prevGapErr     = 0f;
                        _prevGapErrTime = t;
                        _latRef.OnMergeEntry(-ego.GetY(), -ego.GetYDot(), ego.GetPsi(), ego.GetPsiDot(), ego.GetVx(), platoonLaneY, t);
                        // Switch to Merge R1: vel_err≈0 at this point so R1=75000 gives u1≈0.32 m/s²
                        // (insufficient to close gap_err≈14m). R1_long_Merge=10000 gives u1≈1.36 m/s².
                        // GapSearch used R1_long=75000 safely: large vel_err clips to aMax regardless of R1.
                        _longNash.UpdateR1(settings.Nash.R1_long_Merge);
                        Phase = MergePhase.Merge;
                        Debug.Log($"[NashCoordinator] GAP_SEARCH → MERGE at t={t:F1}s (velErr={gsVelErr:F2} m/s) | " +
                                  $"Calibrator end-of-GapSearch: {_driverCalib.GetStatusString()}");
                    }
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
                        _latNash?.Reset();         // clear warm-start from Merge so Following starts fresh
                        _latWarmStartDone = false; // re-seed from current a_lat at first Following tick
                        System.Array.Clear(_conflictBuf, 0, ConflictBufLen);
                        _conflictIdx = 0;
                        _qyPrev = 0f;  // force Q1 update on first Following tick

                        Debug.Log($"[NashCoordinator] MERGE → FOLLOWING at t={t:F1}s | " +
                                  $"Calibrator: {_driverCalib.GetStatusString()}");
                    }
                }
                else
                {
                    _phaseHoldTimer = 0f;
                }
                break;

            case MergePhase.Following:
            {
                // Hysteresis: if gap diverges significantly for long enough → re-enter Merge.
                // Threshold is intentionally looser than Merge→Following entry conditions.
                var fLeader = _mergeLocked ? _lockedLeader : FindLeader(platoon);
                if (fLeader != null)
                {
                    float fGap    = fLeader.GetPosition().z - ego.GetPosition().z
                                  - fLeader.GetLength() * 0.5f - ego.GetLength() * 0.5f;
                    float fDesGap = settings.SafetyField.StandstillDist
                                  + settings.SafetyField.PlatoonTimeGap * ego.GetVx();
                    float fGapFrac = Mathf.Abs(fGap - fDesGap) / Mathf.Max(fDesGap, 1e-3f);

                    // Divergence rate: positive = gap growing (ego falling behind)
                    float elapsed = t - _prevGapTime;
                    float gapDivRate = elapsed > 0.05f ? (fGap - _prevGap) / elapsed : 0f;
                    _prevGap     = fGap;
                    _prevGapTime = t;

                    // Lateral exit: immediate re-Merge when ego leaves the quarter-lane band.
                    float fYErr       = -ego.GetY() - platoonLaneY;
                    float quarterLane = laneWidth * 0.25f;
                    if (Mathf.Abs(fYErr) > quarterLane)
                    {
                        _prevGapErr        = fGap - fDesGap;
                        _prevGapErrTime    = t;
                        _followingDivTimer = 0f;
                        _phaseHoldTimer    = 0f;
                        _latRef.OnMergeEntry(-ego.GetY(), -ego.GetYDot(), ego.GetPsi(), ego.GetPsiDot(), ego.GetVx(), platoonLaneY, t);
                        Phase = MergePhase.Merge;
                        Debug.Log($"[NashCoordinator] FOLLOWING → MERGE (lateral) at t={t:F1}s yErr={fYErr:F3}m > quarterLane={quarterLane:F3}m");
                        break;
                    }

                    // Longitudinal divergence: hold timer before re-entering Merge.
                    bool tooBig    = fGapFrac > settings.Timing.FollowingHysteresisGapFrac;
                    bool diverging = gapDivRate > settings.Timing.FollowingHysteresisDivRate;

                    if (tooBig || diverging)
                        _followingDivTimer += dt;
                    else
                        _followingDivTimer = 0f;

                    if (_followingDivTimer >= settings.Timing.FollowingHysteresisHoldTime)
                    {
                        _prevGapErr        = fGap - fDesGap;
                        _prevGapErrTime    = t;
                        _followingDivTimer = 0f;
                        _phaseHoldTimer    = 0f;
                        _latRef.OnMergeEntry(-ego.GetY(), -ego.GetYDot(), ego.GetPsi(), ego.GetPsiDot(), ego.GetVx(), platoonLaneY, t);
                        Phase = MergePhase.Merge;
                        Debug.Log($"[NashCoordinator] FOLLOWING → MERGE (longitudinal) at t={t:F1}s " +
                                  $"gapFrac={fGapFrac:F2} divRate={gapDivRate:F2} " +
                                  $"(tooBig={tooBig} diverging={diverging})");
                    }
                }
                break;
            }
        }
    }

    bool IsMergeComplete(AdvancedBicycleModel[] platoon)
    {
        bool yOk   = Mathf.Abs(-ego.GetY() - platoonLaneY) < 0.3f;
        bool psiOk = Mathf.Abs(ego.GetPsi())                < 0.05f;

        var leader = _mergeLocked ? _lockedLeader : FindLeader(platoon);
        bool gapOk         = true;
        bool velOk         = true;
        bool convergingOk  = true;
        if (leader != null)
        {
            float gap    = leader.GetPosition().z - ego.GetPosition().z
                         - leader.GetLength() / 2f - ego.GetLength() / 2f;
            float desGap = settings.SafetyField.StandstillDist
                         + settings.SafetyField.PlatoonTimeGap * ego.GetVx();
            float gapErr = gap - desGap;

            gapOk = Mathf.Abs(gapErr) / Mathf.Max(desGap, 1e-3f)
                  < settings.Timing.MergeGapTolerance;
            velOk = Mathf.Abs(ego.GetVx() - leader.GetVx())
                  < settings.Timing.MergeVelTolerance;

            // Gap convergence: |d(gap_err)/dt| must be small (gap settling, not drifting)
            float elapsed = Time.fixedTime - _prevGapErrTime;
            if (elapsed > 0.05f)
            {
                float gapErrDot    = (gapErr - _prevGapErr) / elapsed;
                convergingOk       = Mathf.Abs(gapErrDot) < settings.Timing.MergeGapConvergenceRate;
                _prevGapErr        = gapErr;
                _prevGapErrTime    = Time.fixedTime;
            }
        }
        return yOk && psiOk && gapOk && velOk && convergingOk;
    }

    // =========================================================================
    // Utility
    // =========================================================================

    float LongGapError(AdvancedBicycleModel leader)
    {
        float gap    = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength() / 2f - ego.GetLength() / 2f;
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
        // A platoon vehicle whose bumper-to-bumper gap to ego is negative is
        // laterally parallel — ego will merge ahead of it, so it is a follower.
        foreach (var v in platoon)
        {
            if (v == null) continue;
            float d = v.GetPosition().z - egoZ;
            float halfSum = v.GetLength() * 0.5f + ego.GetLength() * 0.5f;
            // Skip vehicles laterally parallel to ego (|d| <= halfSum):
            // ego is merging ahead of them, so they become followers.
            if (Mathf.Abs(d) <= halfSum) continue;
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
            float halfSum = v.GetLength() * 0.5f + ego.GetLength() * 0.5f;
            // Vehicles parallel to ego (|d| <= halfSum) or behind are followers.
            if (d > halfSum) continue;  // clearly ahead → leader
            if (d > maxD) { maxD = d; best = v; }
        }
        return best;
    }
}
