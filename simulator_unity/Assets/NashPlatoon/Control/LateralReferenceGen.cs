// LateralReferenceGen — R1 (system) and R2 (human) lateral reference trajectories.
//
// R1 SystemRef: 5th-order polynomial lane-change to targetY (Gu & Dolan).
//   Mirrors coordinator._lat_sys_ref():
//     T = 1.875·|dy| / (vx·tan(ψ_max))   (scales naturally: large dy → slow, tiny dy → immediate)
//     MERGE phase: time-based polynomial locked to entry state (τ advances with wall clock)
//     FOLLOWING: settling cubic from current (y, ẏ) to (targetY, 0)
//     APPROACH/GAP_SEARCH: position-based 5th-order polynomial
//   Returns (Np*2,) float[] as [y0,psi0, y1,psi1, …]
//
// R2 HumanRef (Unity HITL, no Stanley/driver_type):
//   Steering wheel angle from G29 → front wheel angle → Neuromuscular IIR (Swain & Rath Eq.7)
//   Lateral state propagated Np steps using linearised bicycle model (Forward Euler, ZOH δ).
//   Mirrors coordinator._lat_hum_ref() but uses real steering input instead of synthetic driver.
using UnityEngine;

public class LateralReferenceGen
{
    readonly ReferenceGeneratorConfig _c;

    // IIR / neuromuscular filter state (Swain & Rath 2023, Eq. 7 Euler-discretised)
    float _psiKm1;
    float _psiKm2;

    // Lateral state-space cache (linearised at current vx)
    float _vxCache = -1f;
    float[] _Ac;   // 4×4 row-major
    float[] _Bc;   // 4×1

    // MERGE entry state for locked polynomial
    bool  _mergeEntryLocked;
    float _mergeEntryY;
    float _mergeEntryYDot;   // yDot at merge entry (solver convention) for smooth polynomial
    float _mergeEntryPsi;    // psi at merge entry — fade-out over T_lc for continuity
    float _mergeEntryTime;
    float _mergeTlcSys = -1f;
    float _mergeTlcHum = -1f;   // human T_lc (factor 1.5, locked at merge entry)

    // GAP_SEARCH entry state for psi/psiDot fade-out
    float _gsEntryPsi    = 0f;
    float _gsEntryPsiDot = 0f;
    float _gsEntryYDot   = 0f;
    float _gsEntryY      = 0f;
    float _gsSettleT     = 0f;   // computed from psi/psiDot at entry
    float _gsEntryTime   = -1f;

    // FOLLOWING entry time
    float _followingEntryTime = -1f;

    public LateralReferenceGen(ReferenceGeneratorConfig c)
    {
        _c = c;
    }

    // Called by NashCoordinator when entering MERGE (locks the polynomial base point)
    public void OnMergeEntry(float egoY, float egoYDot, float egoPsi, float egoPsiDot, float egoVx, float targetY, float wallTime)
    {
        _mergeEntryLocked = true;
        _mergeEntryY      = egoY;
        _mergeEntryYDot   = egoYDot;
        _mergeEntryPsi    = egoPsi;
        _mergeEntryTime   = wallTime;

        // Seed IIR from the actual physical heading (not yDot/vx).
        // GetLatStateVector() passes x0[2]=psi to the solver, so seeding with egoPsi
        // keeps HumanRef consistent with the solver's initial state — no step jump at t=0.
        _psiKm1 = egoPsi;
        _psiKm2 = egoPsi;

        float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;
        float vx      = Mathf.Max(egoVx, 1f);
        float dySign  = targetY - egoY;                 // signed lateral distance
        float dy      = Mathf.Abs(dySign);
        float vyTowardTarget = egoYDot * Mathf.Sign(dySign); // >0 if moving toward target

        // Physical T_lc: for a cubic Hermite with (y0, vy0) → (targetY, 0),
        // integral(ẏ dτ) = dy iff T = 2·dy/vy0. This yields the natural time for
        // vy to decay linearly from vy0 to 0 while covering exactly dy — no overshoot,
        // no requested acceleration. When vy0 opposes dy, fall back to the geometric
        // formula (vehicle must first reverse vy, then approach), scaled by max heading.
        const float tMinAbs = 4.0f;   // absolute floor [s] — never faster than 4s
        float tPhysical;
        if (vyTowardTarget > 0.05f && dy > 1e-4f)
        {
            // Natural approach: vy already points toward target
            tPhysical = 2f * dy / vyTowardTarget;
        }
        else
        {
            // vy neutral or opposing target — polynomial must first reverse motion.
            // Geometric fallback: reach dy at max heading angle.
            tPhysical = dy > 1e-4f
                      ? 1.875f * dy / (vx * Mathf.Tan(maxHeadRad))
                      : tMinAbs;
        }

        _mergeTlcSys = Mathf.Max(tMinAbs, tPhysical);
        _mergeTlcHum = Mathf.Max(tMinAbs, tPhysical * (1.5f / 1.875f));
    }

    // Called every GapSearch Nash tick to update the cubic origin in HumanRef.
    // Because latTarget tracks ego.GetY() (not a fixed lock), the cubic must also
    // start from the current position and velocity so R2 is consistent with R1.
    public void UpdateGapSearchOrigin(float egoY, float egoYDot)
    {
        _gsEntryY    = egoY;
        _gsEntryYDot = egoYDot;
    }

    // Called by NashCoordinator when entering GAP_SEARCH (locks psi/psiDot for fade-out)
    public void OnGapSearchEntry(float egoY, float egoYDot, float egoPsi, float egoPsiDot, float egoVx, float wallTime)
    {
        const float omegaMax_gs   = 0.15f;
        const float alphaPsiMax_gs = 0.20f;
        const float tMinAbs_gs    = 1.0f;

        _gsEntryY      = egoY;
        _gsEntryYDot   = egoYDot;
        _gsEntryPsi    = egoPsi;
        _gsEntryPsiDot = egoPsiDot;
        _gsEntryTime   = wallTime;

        const float aLatMax_gs = 0.8f;
        float tFromVy     = Mathf.Abs(egoYDot)  / aLatMax_gs;
        float tFromPsi    = Mathf.Abs(egoPsi)    / omegaMax_gs;
        float tFromPsiDot = Mathf.Abs(egoPsiDot) / alphaPsiMax_gs;
        _gsSettleT = Mathf.Max(tMinAbs_gs, Mathf.Max(tFromVy, Mathf.Max(tFromPsi, tFromPsiDot)));

        // Seed IIR from actual heading so HumanRef starts consistently
        _psiKm1 = egoPsi;
        _psiKm2 = egoPsi;
    }

    public void OnFollowingEntry(float wallTime)
    {
        _mergeEntryLocked = false;
        _followingEntryTime = wallTime;
    }

    // ── System reference ──────────────────────────────────────────────────────

    public float[] SystemRef(
        AdvancedBicycleModel ego,
        float targetY,
        MergePhase phase,
        float wallTime,
        int Np, float dt)
    {
        float y0   = -ego.GetY();   // solver convention: y = pos.x - _x0 = -GetY()
        float dy   = targetY - y0;
        float vx   = Mathf.Max(ego.GetVx(), 1f);
        float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;

        float[] ref_ = new float[Np * 2];

        if (phase == MergePhase.Following && _followingEntryTime >= 0f)
        {
            // FOLLOWING: same T_lc formula as Merge (OnMergeEntry) so the reference
            // horizon scales naturally with the remaining lateral error.
            // Floor at 3 s (vs 4 s in Merge) because Following corrections are small.
            const float aLatMax     = 0.8f;
            const float omegaMax    = 0.15f;
            const float alphaPsiMax = 0.20f;
            const float tMinAbs     = 3.0f;

            float vy0cur    = -ego.GetYDot();
            float psi0      = ego.GetPsi();
            float psiDot0   = ego.GetPsiDot();
            float tStopVy   = Mathf.Abs(vy0cur) / aLatMax;
            float tStopPsi  = Mathf.Abs(psi0)   / omegaMax;
            float tStopPsiD = Mathf.Abs(psiDot0) / alphaPsiMax;
            float tMin      = Mathf.Max(tMinAbs, Mathf.Max(tStopVy, Mathf.Max(tStopPsi, tStopPsiD)));

            float T_settle = Mathf.Abs(dy) > 1e-4f
                           ? Mathf.Max(tMin, 1.875f * Mathf.Abs(dy) / (vx * Mathf.Tan(maxHeadRad)))
                           : tMin;

            float vy0 = vy0cur;
            if (Mathf.Abs(dy) > 1e-6f)
            {
                float vyMax = 1.5f * Mathf.Abs(dy) / T_settle;
                vy0 = Mathf.Clamp(vy0, -vyMax, vyMax);
            }
            else { vy0 = 0f; }

            float a0 = y0;
            float a1 = vy0 * T_settle;
            float a2 = 3f * dy - 2f * vy0 * T_settle;
            float a3 = -2f * dy + vy0 * T_settle;

            for (int k = 0; k < Np; k++)
            {
                float tPred = (k + 1) * dt;
                float tau   = Mathf.Min(tPred / T_settle, 1f);
                float y, psi;
                if (tau < 1f)
                {
                    y   = a0 + a1 * tau + a2 * tau * tau + a3 * tau * tau * tau;
                    float ydot = (a1 + 2f * a2 * tau + 3f * a3 * tau * tau) / T_settle;
                    psi = Mathf.Clamp(ydot / vx, -0.3f, 0.3f);
                }
                else { y = targetY; psi = 0f; }
                ref_[2 * k]     = y;
                ref_[2 * k + 1] = psi;
            }
        }
        else if (phase == MergePhase.Merge && _mergeEntryLocked)
        {
            // Time-based polynomial locked to entry state.
            // Uses quintic Hermite with non-zero entry yDot so the polynomial
            // matches the vehicle's actual lateral velocity at merge entry.
            float yStart   = _mergeEntryY;
            float vy0      = _mergeEntryYDot;   // solver-convention yDot at entry
            float dyMerge  = targetY - yStart;
            if (_mergeTlcSys < 0f)
                OnMergeEntry(y0, -ego.GetYDot(), ego.GetPsi(), ego.GetPsiDot(), vx, targetY, wallTime);  // fallback init

            float T_lc     = _mergeTlcSys;
            float tElapsed = wallTime - _mergeEntryTime;

            // Quintic Hermite: y(0)=yStart, y'(0)=vy0, y''(0)=0
            //                  y(T)=targetY, y'(T)=0,   y''(T)=0
            // Normalised τ = t/T:
            //   a0 = yStart
            //   a1 = vy0·T
            //   a2 = 0
            //   a3 = 10·dy - 6·vy0·T
            //   a4 = -15·dy + 8·vy0·T
            //   a5 = 6·dy - 3·vy0·T
            float vT  = vy0 * T_lc;
            float c3  = 10f * dyMerge - 6f * vT;
            float c4  = -15f * dyMerge + 8f * vT;
            float c5  = 6f * dyMerge - 3f * vT;

            for (int k = 0; k < Np; k++)
            {
                float tTotal = tElapsed + (k + 1) * dt;
                float tau    = Mathf.Min(tTotal / T_lc, 1f);
                float t2     = tau * tau;
                float t3     = t2 * tau;
                float t4     = t3 * tau;
                float t5     = t4 * tau;
                float y, psi;
                if (tau < 1f)
                {
                    y = yStart + vT * tau + c3 * t3 + c4 * t4 + c5 * t5;
                    float ydot = (vT + 3f * c3 * t2 + 4f * c4 * t3 + 5f * c5 * t4) / T_lc;
                    // psi_ref is the natural heading angle of the polynomial motion:
                    // arctan(ẏ/vx) ≈ ẏ/vx for small angles. The polynomial already
                    // starts with ẏ(0)=vy0, so ẏ(0)/vx ≈ egoPsi implicitly — no need
                    // to add a fade-out of egoPsi on top (that caused clamp saturation
                    // when egoPsi > maxHeadRad, freezing psi_ref at 0.3 rad).
                    psi = ydot / vx;
                }
                else
                {
                    y   = targetY;
                    psi = 0f;
                }
                ref_[2 * k]     = y;
                ref_[2 * k + 1] = psi;
            }
        }
        else
        {
            // APPROACH / GAP_SEARCH: position-based 5th-order polynomial with psi fade-out.
            //
            // GapSearch: psiEntry re-anchors to current ψ every tick (not locked _gsEntryPsi).
            // This prevents overshoot accumulation: when ψ crosses zero the reference immediately
            // reverses direction toward 0 from the new side, so Nash corrects rather than continues.
            // T is recomputed from current |ψ| and |ψ̇| — short settling window each tick.
            float psiEntry;
            float T;
            if (phase == MergePhase.GapSearch)
            {
                // GapSearch: cubic settling for vy — T derived physically from current vy0.
                //
                // Cubic Hermite with (y0, vy0) → (y0, 0):  a1=vT, a2=-2vT, a3=vT (dy=0).
                // Max |ÿ| = 4|vy0|/T at τ=0. Requiring |ÿ| ≤ ayLimit gives T ≥ 4|vy0|/ayLimit.
                // With ayLimit=0.8 m/s² → T ≈ 5·|vy0|. This ensures the reference never
                // demands lateral acceleration beyond the tire capability, so the vehicle
                // can physically follow it without Nash saturating δ.
                //
                // Recomputed every tick from current vy0 (no locking): when vy is small,
                // T is short and the polynomial converges quickly; when vy is large,
                // T is long and the reference is gentle. Both are physically consistent
                // with the current state — no feedback instability.
                float vy0gs    = -ego.GetYDot();          // current vy0, solver convention
                psiEntry       = ego.GetPsi();             // re-anchored every tick

                const float ayLimit = 0.8f;                // lateral accel budget [m/s²]
                const float tMinGs  = 1.0f;                // absolute floor [s]
                float T_gs = Mathf.Max(tMinGs, 4f * Mathf.Abs(vy0gs) / ayLimit);

                float yStart = y0;   // current ego position (dy=0 target)

                float vT  = vy0gs * T_gs;
                float a1c =  vT;
                float a2c = -2f * vT;
                float a3c =  vT;

                for (int k = 0; k < Np; k++)
                {
                    float tPred = (k + 1) * dt;
                    float tau   = Mathf.Min(tPred / T_gs, 1f);
                    float y, psi;
                    if (tau < 1f)
                    {
                        float tau2 = tau * tau;
                        float tau3 = tau2 * tau;
                        y = yStart + a1c * tau + a2c * tau2 + a3c * tau3;
                        float ydot    = (a1c + 2f * a2c * tau + 3f * a3c * tau2) / T_gs;
                        // psi_ref = polynomial heading. No fade-out of psiEntry — same
                        // reasoning as Merge: ẏ/vx already reflects the natural heading
                        // for this motion. Adding (1-τ)·psiEntry caused clamp saturation
                        // when psi was large, freezing the reference.
                        psi = ydot / Mathf.Max(vx, 0.1f);
                    }
                    else { y = yStart; psi = 0f; }
                    ref_[2 * k]     = y;
                    ref_[2 * k + 1] = psi;
                }
                return ref_;
            }
            else
            {
                psiEntry = ego.GetPsi();
                T = Mathf.Abs(dy) > 1e-4f
                  ? Mathf.Max(dt, 1.875f * Mathf.Abs(dy) / (vx * Mathf.Tan(maxHeadRad)))
                  : dt;
            }

            for (int k = 0; k < Np; k++)
            {
                float t   = (k + 1) * dt;
                float tau = Mathf.Min(t / T, 1f);
                float t2  = tau * tau;
                float t3  = t2 * tau;
                float t4  = t3 * tau;
                float t5  = t4 * tau;
                float y   = y0 + dy * (10f * t3 - 15f * t4 + 6f * t5);
                float psi = 0f;
                if (tau < 1f)
                {
                    float ydot    = (dy / Mathf.Max(T, 1e-6f)) * (30f * t2 - 60f * t3 + 30f * t4);
                    float psiPoly = Mathf.Abs(dy) > 1e-4f ? ydot / Mathf.Max(vx, 0.1f) : 0f;
                    psi = Mathf.Clamp(psiPoly + (1f - tau) * psiEntry, -0.3f, 0.3f);
                }
                ref_[2 * k]     = y;
                ref_[2 * k + 1] = psi;
            }
        }
        return ref_;
    }

    // ── Human reference ───────────────────────────────────────────────────────
    //
    // MERGE phase: cubic polynomial + driver offset → IIR.
    //   y_ref  = polynomial cubic (locked at merge entry, like Python _lat_hum_ref).
    //   psi_des[k] = psi_poly[k] + steeringOffset
    //     where psi_poly[k] = polynomial derivative at step k (→ 0 at lane-change end),
    //     and steeringOffset = clip(psi_steering − psi_poly_now, ±R2MaxHeadingDeg)
    //     captures how much the driver currently deviates from the expected trajectory.
    //   Driver input appears via: IIR initial conditions (seeded from actual psi) AND
    //   the steeringOffset term (driver's real deviation modulates each predicted step).
    //
    // Other phases: original G29 → psi_des → IIR (unchanged).
    //
    // To revert MERGE to the old path: remove the `if` branch and keep only the else block.

    public float[] HumanRef(
        AdvancedBicycleModel ego,
        float targetY, MergePhase phase, float wallTime,
        int Np, float dt)
    {
        var vp = ego.GetVehicleParameters();
        float vx = Mathf.Max(ego.GetVx(), 0.5f);

        // Actual driver steering → front wheel angle → psiSteering [rad]
        float rawSteering = VehicleInputs.Instance != null
                          ? VehicleInputs.Instance.SteeringInput : 0f;
        // OLD (V1) — physical model gives ~0.08° at 30 m/s, too small to influence R2:
        // float deltaHuman = rawSteering * vp.maxWheelAngle * Mathf.Deg2Rad / vp.steeringRatio;
        // deltaHuman = Mathf.Clamp(deltaHuman, -0.436f, 0.436f);
        // float psiSteering = (float)System.Math.Atan(vp.lf * deltaHuman / vx);

        // NEW (V2) — direct heading intent: full wheel → R2MaxHeadingDeg (2.6°)
        float psiSteering = rawSteering * _c.R2MaxHeadingDeg * Mathf.Deg2Rad;

        // Neuromuscular IIR coefficients (Swain & Rath 2023, Eq. 7, Euler-discretised)
        float neuroJ = _c.NeuroJ, neuroB = _c.NeuroB, neuroK = _c.NeuroK;
        float nd  = neuroJ + neuroB * dt + neuroK * dt * dt;
        float na0 = (dt * dt) / nd;
        float na1 = (2f * neuroJ + neuroB * dt) / nd;
        float na2 = -neuroJ / nd;

        // Seed IIR: during MERGE/GAP_SEARCH use polynomial psi to break the circular dependency
        // where large actual psi seeds the IIR → R2 predicts large psi → Nash can't correct.
        // In GapSearch, psi decays linearly from _gsEntryPsi matching the SystemRef fade-out.
        // Outside MERGE/GAP_SEARCH, seed from actual heading (reflects driver inputs normally).
        float psiNow;
        if (phase == MergePhase.Merge && _mergeEntryLocked && _mergeTlcHum > 0f)
        {
            float tauSeed  = Mathf.Min((wallTime - _mergeEntryTime) / _mergeTlcHum, 1f);
            float yDotSeed = (targetY - _mergeEntryY) * (6f * tauSeed - 6f * tauSeed * tauSeed) / _mergeTlcHum;
            psiNow = tauSeed < 1f ? yDotSeed / vx : 0f;
        }
        else if (phase == MergePhase.GapSearch)
        {
            // Mirror GapSearch SystemRef: re-anchor to current ψ every tick so IIR seed
            // is consistent with R1 and immediately reverses when ψ overshoots zero.
            psiNow = ego.GetPsi();
        }
        else
        {
            psiNow = ego.GetPsi();
        }
        _psiKm2 = _psiKm1;
        _psiKm1 = psiNow;

        float[] ref_     = new float[Np * 2];
        float psiK       = _psiKm1;
        float psiKm1Pred = _psiKm2;

        if (phase == MergePhase.Merge && _mergeEntryLocked)
        {
            // ── MERGE: polynomial + driver offset ─────────────────────────────────
            float deltaY = targetY - _mergeEntryY;

            if (_mergeTlcHum < 0f)   // lazy init (OnMergeEntry normally sets this)
            {
                float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;
                float dy = Mathf.Abs(deltaY);
                _mergeTlcHum = dy > 1e-4f
                             ? Mathf.Max(dt, 1.5f * dy / (vx * Mathf.Tan(maxHeadRad)))
                             : dt;
            }

            float T_lc_h   = _mergeTlcHum;
            float tElapsed = wallTime - _mergeEntryTime;

            // psiPolyNow from the same quintic Hermite used by R1 (with vy0 = _mergeEntryYDot)
            float tauNow = Mathf.Min(tElapsed / T_lc_h, 1f);
            float psiPolyNow;
            if (tauNow < 1f)
            {
                float vy0h = _mergeEntryYDot;
                float vTh  = vy0h * T_lc_h;
                float c3h  = 10f * deltaY - 6f * vTh;
                float c4h  = -15f * deltaY + 8f * vTh;
                float c5h  = 6f * deltaY - 3f * vTh;
                float tau2 = tauNow * tauNow;
                float tau3 = tau2 * tauNow;
                float tau4 = tau3 * tauNow;
                float ydotNow = (vTh + 3f * c3h * tau2 + 4f * c4h * tau3 + 5f * c5h * tau4) / T_lc_h;
                psiPolyNow = ydotNow / vx;
            }
            else { psiPolyNow = 0f; }

            // Steering offset: driver's intentional deviation from the polynomial.
            // Scale by |rawSteering| so a passive/neutral driver produces zero offset
            // and only active steering inputs from the driver modulate R2.
            float maxOffset      = _c.R2MaxHeadingDeg * Mathf.Deg2Rad;
            float steerMagMerge  = Mathf.Clamp01(Mathf.Abs(rawSteering) / 0.15f);
            float steeringOffset = steerMagMerge * Mathf.Clamp(psiSteering - psiPolyNow, -maxOffset, maxOffset);

            // Cubic Hermite with non-zero entry yDot — continuity at merge entry
            float vy0R2 = _mergeEntryYDot;
            float vTR2  = vy0R2 * T_lc_h;
            float a2R2  = 3f * deltaY - 2f * vTR2;
            float a3R2  = -2f * deltaY + vTR2;

            for (int k = 0; k < Np; k++)
            {
                float tTotal = tElapsed + (k + 1) * dt;
                float tau    = Mathf.Min(tTotal / T_lc_h, 1f);
                float yRef, psiDes;
                if (tau < 1f)
                {
                    float tau2 = tau * tau;
                    float tau3 = tau2 * tau;
                    yRef = _mergeEntryY + vTR2 * tau + a2R2 * tau2 + a3R2 * tau3;
                    float yDot      = (vTR2 + 2f * a2R2 * tau + 3f * a3R2 * tau2) / T_lc_h;
                    float offsetAtK = steeringOffset * (1f - tau);
                    // psi_ref = polynomial heading (ẏ/vx) + driver's steering intent.
                    // No fade-out of _mergeEntryPsi — the polynomial's ẏ(0)/vx already
                    // matches egoPsi implicitly (vy0 = _mergeEntryYDot).
                    psiDes = yDot / vx + offsetAtK;
                }
                else
                {
                    yRef   = targetY;
                    // Polynomial done: hand off to driver's actual steering intent.
                    // Without this, psiExec=0 for all τ≥1 → u2≈0 → driver feels no authority.
                    psiDes = psiSteering;
                }
                float psiExec = na0 * psiDes + na1 * psiK + na2 * psiKm1Pred;
                psiExec = Mathf.Clamp(psiExec, -maxOffset, maxOffset);
                psiKm1Pred = psiK;
                psiK       = psiExec;
                ref_[2 * k]     = yRef;
                ref_[2 * k + 1] = psiExec;
            }
        }
        else if (phase == MergePhase.GapSearch)
        {
            // ── GAP_SEARCH: cubic settling — mirrors SystemRef GapSearch branch ────────
            // T derived physically from current vy0 (recomputed every tick), no locking.
            // The polynomial always describes the natural motion from the current state.
            float r2MaxPsi  = _c.R2MaxHeadingDeg * Mathf.Deg2Rad;
            float maxOffset = r2MaxPsi;

            float psiNowGs = ego.GetPsi();
            float vy0gs    = -ego.GetYDot();          // solver convention, current
            float yEntry   = -ego.GetY();              // current ego position (dy=0)

            const float ayLimit_r2 = 0.8f;
            const float tMinGs_r2  = 1.0f;
            float T_gs = Mathf.Max(tMinGs_r2, 4f * Mathf.Abs(vy0gs) / ayLimit_r2);

            float vTgs = vy0gs * T_gs;
            float a2gs = -2f * vTgs;
            float a3gs =  vTgs;

            float steerMagGs    = Mathf.Clamp01(Mathf.Abs(rawSteering) / 0.15f);
            float steeringOffGs = steerMagGs * Mathf.Clamp(psiSteering - psiNowGs, -maxOffset, maxOffset);

            for (int k = 0; k < Np; k++)
            {
                float tPred = (k + 1) * dt;
                float tau   = Mathf.Min(tPred / T_gs, 1f);
                float yRef, psiDes;
                if (tau < 1f)
                {
                    float tau2 = tau * tau;
                    float tau3 = tau2 * tau;
                    yRef = yEntry + vTgs * tau + a2gs * tau2 + a3gs * tau3;
                    float ydot      = (vTgs + 2f * a2gs * tau + 3f * a3gs * tau2) / T_gs;
                    float offsetAtK = steeringOffGs * (1f - tau);
                    // psi_ref = polynomial heading + driver offset. No fade-out of psiNowGs.
                    psiDes = ydot / vx + offsetAtK;
                }
                else
                {
                    yRef   = yEntry;
                    psiDes = psiSteering;
                }
                float psiExec = na0 * psiDes + na1 * psiK + na2 * psiKm1Pred;
                psiExec = Mathf.Clamp(psiExec, -maxOffset, maxOffset);
                psiKm1Pred = psiK;
                psiK       = psiExec;
                ref_[2 * k]     = yRef;
                ref_[2 * k + 1] = psiExec;
            }
        }
        else
        {
            // ── APPROACH / FOLLOWING: cubic settling to targetY ───────────────────────
            // R2 uses the same T_lc formula as R1/Merge so the human reference horizon
            // scales with the remaining lateral error — identical logic to SystemRef Following.
            const float aLatMax_h     = 0.8f;
            const float omegaMax_h    = 0.15f;
            const float alphaPsiMax_h = 0.20f;
            const float tMinAbs_h     = 3.0f;

            float dy   = targetY - (-ego.GetY());
            float vx_h = Mathf.Max(ego.GetVx(), 1f);
            float maxHeadRad_h = _c.MaxHeadingDeg * Mathf.Deg2Rad;

            float vy0_h   = -ego.GetYDot();
            float psi0_h  = ego.GetPsi();
            float psiD0_h = ego.GetPsiDot();
            float tStopVy_h   = Mathf.Abs(vy0_h)  / aLatMax_h;
            float tStopPsi_h  = Mathf.Abs(psi0_h)  / omegaMax_h;
            float tStopPsiD_h = Mathf.Abs(psiD0_h) / alphaPsiMax_h;
            float tMin_h = Mathf.Max(tMinAbs_h, Mathf.Max(tStopVy_h, Mathf.Max(tStopPsi_h, tStopPsiD_h)));

            float T_h = Mathf.Abs(dy) > 1e-4f
                      ? Mathf.Max(tMin_h, 1.875f * Mathf.Abs(dy) / (vx_h * Mathf.Tan(maxHeadRad_h)))
                      : tMin_h;

            float vy0 = vy0_h;
            if (Mathf.Abs(dy) > 1e-6f)
            {
                float vyMax = 1.5f * Mathf.Abs(dy) / T_h;
                vy0 = Mathf.Clamp(vy0, -vyMax, vyMax);
            }
            else { vy0 = 0f; }

            float a0 = -ego.GetY();  // solver convention
            float a1 = vy0 * T_h;
            float a2 = 3f * dy - 2f * vy0 * T_h;
            float a3 = -2f * dy + vy0 * T_h;

            float r2MaxPsi = _c.R2MaxHeadingDeg * Mathf.Deg2Rad;

            for (int k = 0; k < Np; k++)
            {
                float tPred = (k + 1) * dt;
                float tau   = Mathf.Min(tPred / T_h, 1f);
                float yRef, psiRef;
                if (tau < 1f)
                {
                    yRef = a0 + a1 * tau + a2 * tau * tau + a3 * tau * tau * tau;
                    float ydot = (a1 + 2f * a2 * tau + 3f * a3 * tau * tau) / T_h;
                    // R2 psi: blend polynomial heading with driver's IIR intent.
                    float psiExec = na0 * psiSteering + na1 * psiK + na2 * psiKm1Pred;
                    psiExec       = Mathf.Clamp(psiExec, -r2MaxPsi, r2MaxPsi);
                    psiKm1Pred    = psiK;
                    psiK          = psiExec;
                    // Weight driver intent by steering magnitude; at rest → same as R1
                    float steerMag = Mathf.Abs(rawSteering);
                    float blend    = Mathf.Clamp01(steerMag / 0.2f);
                    psiRef = Mathf.Clamp(
                        (1f - blend) * (ydot / vx_h) + blend * psiExec,
                        -r2MaxPsi, r2MaxPsi);
                }
                else { yRef = targetY; psiRef = 0f; }

                ref_[2 * k]     = yRef;
                ref_[2 * k + 1] = psiRef;
            }
        }
        return ref_;
    }

    void RebuildJacobian(VehicleParameters vp, float vx)
    {
        _vxCache = vx;
        float m   = vp.mass;
        float Iz  = vp.Iz;
        float lf  = vp.lf;
        float lr  = vp.lr;
        float Caf = vp.Caf;
        float Car = vp.Car;

        // Linearised bicycle model Ac (4×4 row-major: [y, vy, psi, psiDot])
        // y = _x0 - position.x → ẏ = -vy - vx·ψ (body y opposite to world X)
        _Ac = new float[16];
        // Row 0: ẏ = -vy - vx·ψ
        _Ac[0 * 4 + 1] = -1f;
        _Ac[0 * 4 + 2] = -vx;
        // Caf/Car are per-wheel; factor 2 accounts for both wheels on each axle (Belousov Eq. 3.12–3.13)
        // Row 1: v̇y
        _Ac[1 * 4 + 0] = 0f;
        _Ac[1 * 4 + 1] = -2f * (Caf + Car) / (m * vx);
        _Ac[1 * 4 + 2] = 0f;
        _Ac[1 * 4 + 3] = 2f * (lr * Car - lf * Caf) / (m * vx) - vx;
        // Row 2: ψ̇ = psiDot
        _Ac[2 * 4 + 3] = 1f;
        // Row 3: ψ̈
        _Ac[3 * 4 + 0] = 0f;
        _Ac[3 * 4 + 1] = 2f * (lr * Car - lf * Caf) / (Iz * vx);
        _Ac[3 * 4 + 2] = 0f;
        _Ac[3 * 4 + 3] = -2f * (lf * lf * Caf + lr * lr * Car) / (Iz * vx);

        // Bc (4×1) — per-wheel Caf × 2 for full axle
        _Bc = new float[] { 0f, 2f * Caf / m, 0f, 2f * lf * Caf / Iz };
    }
}
