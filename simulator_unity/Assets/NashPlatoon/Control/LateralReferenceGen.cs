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
    public void OnMergeEntry(float egoY, float egoVx, float targetY, float wallTime)
    {
        _mergeEntryLocked = true;
        _mergeEntryY      = egoY;
        _mergeEntryTime   = wallTime;

        // Reset IIR so the prediction starts from the polynomial (psi=0 at tau=0),
        // not from whatever large heading accumulated during APPROACH.
        _psiKm1 = 0f;
        _psiKm2 = 0f;

        float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;
        float dy  = Mathf.Abs(targetY - egoY);
        float vx  = Mathf.Max(egoVx, 1f);
        _mergeTlcSys = dy > 1e-4f
                     ? Mathf.Max(Time.fixedDeltaTime, 1.875f * dy / (vx * Mathf.Tan(maxHeadRad)))
                     : Time.fixedDeltaTime;

        // Human T_lc uses factor 1.5 (mirrors Python _lat_hum_ref: factor 1.5 × human max_heading_deg)
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
        float y0   = ego.GetY();
        float dy   = targetY - y0;
        float vx   = Mathf.Max(ego.GetVx(), 1f);
        float maxHeadRad = _c.MaxHeadingDeg * Mathf.Deg2Rad;

        float[] ref_ = new float[Np * 2];

        if (phase == MergePhase.Following && _followingEntryTime >= 0f)
        {
            // Settling cubic from (y0, ẏ0) → (targetY, 0)
            float T_settle   = 20f;  // DRIVER_PARAMS['normal']['system_settle_time']
            float tInFollow  = wallTime - _followingEntryTime;
            float T_rem      = Mathf.Max(T_settle - tInFollow, dt);

            float vy0 = ego.GetYDot();
            if (Mathf.Abs(dy) > 1e-6f)
            {
                float vyMax = 1.5f * Mathf.Abs(dy) / T_rem;
                vy0 = Mathf.Clamp(vy0, -vyMax, vyMax);
            }
            else { vy0 = 0f; }

            float a0 = y0;
            float a1 = vy0 * T_rem;
            float a2 = 3f * dy - 2f * vy0 * T_rem;
            float a3 = -2f * dy + vy0 * T_rem;

            for (int k = 0; k < Np; k++)
            {
                float tPred = k * dt;
                float tau   = Mathf.Min(tPred / T_rem, 1f);
                float y, psi;
                if (tau < 1f)
                {
                    y   = a0 + a1 * tau + a2 * tau * tau + a3 * tau * tau * tau;
                    float ydot = (a1 + 2f * a2 * tau + 3f * a3 * tau * tau) / T_rem;
                    psi = Mathf.Clamp(-ydot / vx, -0.3f, 0.3f);  // ψ = -ẏ/vx (y = _x0 - pos.x)
                }
                else { y = targetY; psi = 0f; }
                ref_[2 * k]     = y;
                ref_[2 * k + 1] = psi;
            }
        }
        else if (phase == MergePhase.Merge && _mergeEntryLocked)
        {
            // Time-based polynomial locked to entry state (τ advances with wall clock)
            float yStart  = _mergeEntryY;
            float dyMerge = targetY - yStart;
            if (_mergeTlcSys < 0f)
                OnMergeEntry(y0, vx, targetY, wallTime);  // fallback init

            float T_lc    = _mergeTlcSys;
            float tElapsed = wallTime - _mergeEntryTime;

            for (int k = 0; k < Np; k++)
            {
                float tTotal = tElapsed + (k + 1) * dt;
                float tau    = Mathf.Min(tTotal / T_lc, 1f);
                float t2     = tau * tau;
                float t3     = t2 * tau;
                float t4     = t3 * tau;
                float t5     = t4 * tau;
                float y      = yStart + dyMerge * (10f * t3 - 15f * t4 + 6f * t5);
                float psi    = 0f;
                if (tau < 1f)
                {
                    float ydot = (dyMerge / T_lc) * (30f * t2 - 60f * t3 + 30f * t4);
                    psi = Mathf.Clamp(-ydot / vx, -0.3f, 0.3f);  // ψ = -ẏ/vx
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
                    psi = Mathf.Clamp(-ydot / Mathf.Max(vx, 0.1f), -0.3f, 0.3f);  // ψ = -ẏ/vx
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
            psiNow = tauSeed < 1f
                   ? (targetY - _mergeEntryY) * (6f * tauSeed - 6f * tauSeed * tauSeed) / _mergeTlcHum / vx
                   : 0f;
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

            // Polynomial psi at current time → used to compute driver's deviation offset
            // ψ = -ẏ/vx because y = _x0 - position.x (body y opposite to world X)
            float tauNow     = Mathf.Min(tElapsed / T_lc_h, 1f);
            float psiPolyNow = tauNow < 1f
                             ? -deltaY * (6f * tauNow - 6f * tauNow * tauNow) / T_lc_h / vx
                             : 0f;

            // Steering offset: driver's deviation from the polynomial at this moment
            float maxOffset     = _c.R2MaxHeadingDeg * Mathf.Deg2Rad;
            float steeringOffset = Mathf.Clamp(psiSteering - psiPolyNow, -maxOffset, maxOffset);

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
                    // Na & Cole 2021: driver offset fades linearly as lane change completes.
                    // At τ=0: full steeringOffset; at τ=1: zero offset (merge done).
                    float offsetAtK = steeringOffset * (1f - tau);
                    psiDes = -yDot / vx + offsetAtK;
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
            // ── NON-MERGE: road centerline + driver steering overlay ───────────────
            // Na, Cole & Li (Veh. Sys. Dyn. 2021): R2 y-component = desired path geometry
            // (road centerline = targetY), not the propagated ego position.
            // Rationale: the driver's GOAL is to be at targetY; the steering input captures
            // HOW they are getting there (transient heading), not WHERE they want to end up.
            // Propagating y via psiExec caused R2 to drift away from the lane target whenever
            // the driver corrected — conflicting with the system's lane-centering goal.
            float r2MaxPsi = _c.R2MaxHeadingDeg * Mathf.Deg2Rad;

            if (Mathf.Abs(vx - _vxCache) > 0.5f) RebuildJacobian(vp, vx);

            for (int k = 0; k < Np; k++)
            {
                float psiExec = na0 * psiSteering + na1 * psiK + na2 * psiKm1Pred;
                psiExec = Mathf.Clamp(psiExec, -r2MaxPsi, r2MaxPsi);
                ref_[2 * k]     = targetY;   // driver's lateral goal = road centerline
                ref_[2 * k + 1] = psiExec;   // driver's current heading intent (transient)
                psiKm1Pred = psiK;
                psiK       = psiExec;
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
        // Row 1: v̇y
        _Ac[1 * 4 + 0] = 0f;
        _Ac[1 * 4 + 1] = -(Caf + Car) / (m * vx);
        _Ac[1 * 4 + 2] = 0f;
        _Ac[1 * 4 + 3] = (lr * Car - lf * Caf) / (m * vx) - vx;
        // Row 2: ψ̇ = psiDot
        _Ac[2 * 4 + 3] = 1f;
        // Row 3: ψ̈
        _Ac[3 * 4 + 0] = 0f;
        _Ac[3 * 4 + 1] = (lr * Car - lf * Caf) / (Iz * vx);
        _Ac[3 * 4 + 2] = 0f;
        _Ac[3 * 4 + 3] = -(lf * lf * Caf + lr * lr * Car) / (Iz * vx);

        // Bc (4×1)
        _Bc = new float[] { 0f, Caf / m, 0f, lf * Caf / Iz };
    }
}
