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
    float _mergeEntryTime;
    float _mergeTlcSys = -1f;
    float _mergeTlcHum = -1f;   // human T_lc (factor 1.5, locked at merge entry)

    // FOLLOWING entry time
    float _followingEntryTime = -1f;

    public LateralReferenceGen(ReferenceGeneratorConfig c)
    {
        _c = c;
    }

    // Called by NashCoordinator when entering MERGE (locks the polynomial base point)
    public void OnMergeEntry(float egoY, float egoYDot, float egoVx, float targetY, float wallTime)
    {
        _mergeEntryLocked = true;
        _mergeEntryY      = egoY;
        _mergeEntryYDot   = egoYDot;
        _mergeEntryTime   = wallTime;

        // Reset IIR so the prediction starts from the polynomial heading,
        // not from whatever large heading accumulated during APPROACH.
        _psiKm1 = egoYDot / Mathf.Max(egoVx, 1f);
        _psiKm2 = _psiKm1;

        float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;
        float vx  = Mathf.Max(egoVx, 1f);
        // remaining distance to cover: if driver already moving toward target,
        // effective dy shrinks; use actual remaining gap for T_lc.
        float dy  = Mathf.Abs(targetY - egoY);
        // T_lc: time needed to cover dy at max heading, but at least long enough
        // for the entry yDot to decay naturally (dy_from_yDot = yDot²/(2·a_lat_max)).
        _mergeTlcSys = dy > 1e-4f
                     ? Mathf.Max(Time.fixedDeltaTime, 1.875f * dy / (vx * Mathf.Tan(maxHeadRad)))
                     : Time.fixedDeltaTime;

        // Human T_lc uses factor 1.5
        _mergeTlcHum = dy > 1e-4f
                     ? Mathf.Max(Time.fixedDeltaTime, 1.5f * dy / (vx * Mathf.Tan(maxHeadRad)))
                     : Time.fixedDeltaTime;
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
            // FOLLOWING: R1 always aims at targetY (platoon lane centre) so that the
            // Nash game naturally rewards the driver when steering toward the target
            // (u1 and u2 agree → driver feels authority) and corrects when opposing.
            //
            // T_eff is fixed at T_settle rather than shrinking — the old shrinking
            // T_rem → 0 caused ydot/T_rem → ∞ oscillation.
            const float T_settle = 20f;

            float vy0 = -ego.GetYDot();  // solver convention: vy_solver = -GetYDot()

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
                OnMergeEntry(y0, -ego.GetYDot(), vx, targetY, wallTime);  // fallback init

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
                    psi = Mathf.Clamp(ydot / vx, -0.3f, 0.3f);
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
            // APPROACH / GAP_SEARCH: position-based 5th-order polynomial
            float T = Mathf.Abs(dy) > 1e-4f
                    ? Mathf.Max(dt, 1.875f * Mathf.Abs(dy) / (vx * Mathf.Tan(maxHeadRad)))
                    : dt;

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
                    float ydot = (dy / Mathf.Max(T, 1e-6f)) * (30f * t2 - 60f * t3 + 30f * t4);
                    psi = Mathf.Clamp(ydot / Mathf.Max(vx, 0.1f), -0.3f, 0.3f);
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

        // Seed IIR: during MERGE use polynomial psi to break the circular dependency
        // where large actual psi seeds the IIR → R2 predicts large psi → Nash can't correct.
        // Outside MERGE, seed from actual heading (reflects driver inputs normally).
        float psiNow;
        if (phase == MergePhase.Merge && _mergeEntryLocked && _mergeTlcHum > 0f)
        {
            float tauSeed = Mathf.Min((wallTime - _mergeEntryTime) / _mergeTlcHum, 1f);
            // ẏ/vx = psi when y = _x0 - pos.x and rightward is positive
            float yDotSeed = (targetY - _mergeEntryY) * (6f * tauSeed - 6f * tauSeed * tauSeed) / _mergeTlcHum;
            psiNow = tauSeed < 1f ? yDotSeed / vx : 0f;
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

            for (int k = 0; k < Np; k++)
            {
                float tTotal = tElapsed + (k + 1) * dt;
                float tau    = Mathf.Min(tTotal / T_lc_h, 1f);
                float yRef, psiDes;
                if (tau < 1f)
                {
                    float tau2 = tau * tau;
                    yRef   = _mergeEntryY + deltaY * (3f * tau2 - 2f * tau2 * tau);
                    float yDot = deltaY * (6f * tau - 6f * tau2) / T_lc_h;
                    float offsetAtK = steeringOffset * (1f - tau);
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
        else
        {
            // ── NON-MERGE (APPROACH / GAP_SEARCH / FOLLOWING): cubic settling to targetY ──
            // The Forward-Euler bicycle propagation was numerically unstable at high vx
            // (dt·vx ≈ 0.05·30 = 1.5, near the Euler stability limit for the -vx·ψ coupling).
            // Instead R2 uses the same cubic form as R1 but seeded from the driver's current
            // state (vy0 from GetYDot(), psi from IIR-filtered steering intent).
            // This keeps R1 and R2 in the same function space so IBR converges, while still
            // letting the driver's actual heading intention (psiSteering via IIR) modulate R2.
            const float T_h = 20f;  // same settling horizon as R1 SystemRef

            float vy0 = -ego.GetYDot();  // solver convention
            float dy  = targetY - (-ego.GetY());

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

            // IIR-filtered driver heading intent — this is what differentiates R2 from R1:
            // R1 uses psi=ydot/vx from the polynomial; R2 uses the driver's actual heading.
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
                    // IIR step: neuromuscular filter on psiSteering
                    float psiExec = na0 * psiSteering + na1 * psiK + na2 * psiKm1Pred;
                    psiExec       = Mathf.Clamp(psiExec, -r2MaxPsi, r2MaxPsi);
                    psiKm1Pred    = psiK;
                    psiK          = psiExec;
                    // Weight driver intent by steering magnitude; at rest → same as R1
                    float steerMag = Mathf.Abs(rawSteering);
                    float blend    = Mathf.Clamp01(steerMag / 0.2f);
                    psiRef = Mathf.Clamp(
                        (1f - blend) * (ydot / vx) + blend * psiExec,
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
