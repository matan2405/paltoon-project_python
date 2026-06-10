// LongitudinalReferenceGen — R1 (system) and R2 (human) longitudinal reference trajectories.
//
// R1 SystemRef: Rajamani Ch. 6.7 transitional controller propagated Np steps.
//   Mirrors coordinator._long_sys_ref():
//     CATCHUP: gap > 2×desGap AND lv > vx+2 → reference = platoon velocity immediately
//              (creates large tracking error → Nash outputs U1Max, jerk-limited only)
//     COLLISION_AVOIDANCE: TTC < TtcThreshold → emergency decel
//     CRUISE: far from leader or no leader → free-road IDM toward platoon speed
//     TRANSITIONAL (default): parabola target velocity + CTG feedback
//   Returns (Np*2,) float[] as [x0,vx0, x1,vx1, …]
//
// R2 HumanRef (Unity HITL, no driver_type profiles):
//   Uses actual G29 throttle/brake inputs from VehicleInputs.Instance.
//   Propagates a ZOH human acceleration Np steps.
//   Mirrors coordinator._long_hum_ref() but replaces IDM+GP with real pedal input.
using UnityEngine;

public class LongitudinalReferenceGen
{
    readonly ReferenceGeneratorConfig _c;
    readonly SafetyFieldConfig        _sf;

    public LongitudinalReferenceGen(ReferenceGeneratorConfig c, SafetyFieldConfig sf)
    {
        _c  = c;
        _sf = sf;
    }

    // ── System reference (Rajamani transitional) ──────────────────────────────

    public float[] SystemRef(
        AdvancedBicycleModel ego,
        AdvancedBicycleModel leader,
        float platoonTargetVelocity,
        int Np, float dt)
    {
        float x  = ego.GetX();
        float vx = ego.GetVx();

        float lx = leader != null ? leader.GetPosition().z : float.MaxValue;
        float lv = leader != null ? leader.GetVx() : platoonTargetVelocity;

        float desGap = _sf.StandstillDist + _c.RajamaniH * vx;
        float detectionRange = _c.DetectionRange;

        float[] ref_ = new float[Np * 2];

        // CATCHUP: leader exists, far behind (gap > 2×desGap), AND platoon significantly
        // faster (lv > vx+2). Set reference = platoon velocity for all steps so the Nash
        // QP sees a ~(lv-vx) velocity tracking error at every step and saturates at U1Max.
        if (leader != null)
        {
            float initGap = lx - x - leader.GetLength();
            if (initGap > 2f * desGap && lv > vx + 2f)
            {
                float xk = x;
                for (int k = 0; k < Np; k++)
                {
                    xk += lv * dt;
                    ref_[2 * k]     = xk;
                    ref_[2 * k + 1] = lv;
                }
                return ref_;
            }
        }

        for (int k = 0; k < Np; k++)
        {
            float a;
            float gap = lx - x - (leader != null ? leader.GetLength() : 0f);

            if (leader == null || gap > detectionRange)
            {
                // CRUISE: IDM free-road toward platoon speed
                float ratio = platoonTargetVelocity > 0.01f
                            ? vx / (platoonTargetVelocity * _c.CatchupMultiplier)
                            : 0f;
                a = _c.K1Ctg > 0
                    ? Mathf.Clamp(
                        _c.IdmAMax * (1f - Mathf.Pow(Mathf.Clamp01(ratio), _c.FreeDelta)),
                        _c.MaxEmergDecel, _c.IdmAMax)
                    : 0f;
            }
            else
            {
                // TTC check
                float relVel = vx - lv;
                float ttc    = relVel > 0.1f ? gap / relVel : 1e6f;
                float minCritGap = _c.MinCritGapOffset
                                 + (leader != null ? leader.GetLength() * 0.5f : 0f);

                if (ttc < _c.TtcThreshold || gap < minCritGap)
                {
                    // COLLISION_AVOIDANCE: emergency decel
                    a = _c.MaxEmergDecel;
                }
                else
                {
                    // TRANSITIONAL: Rajamani parabolic + CTG
                    float gapError  = gap - desGap;     // >0 too far, <0 too close
                    float vTarget   = lv + Mathf.Sign(gapError)
                                    * Mathf.Sqrt(2f * _c.AComfort * Mathf.Abs(gapError));
                    float aParabola = _c.KV * (vTarget - vx);

                    // Blend with CTG for small gap errors (quadratic — mirrors coordinator._long_sys_ref)
                    if (Mathf.Abs(gapError) < 5f)
                    {
                        float aDot  = lv - vx;
                        float aCtg  = _c.K1Ctg * gapError + _c.K2Ctg * aDot;
                        float gapMag = Mathf.Abs(gapError);
                        float blend  = Mathf.Pow(1f - gapMag / 5f, 2f);
                        a = (1f - blend) * aParabola + blend * aCtg;
                    }
                    else
                    {
                        a = aParabola;
                    }
                    a = Mathf.Clamp(a, _c.MaxEmergDecel, _c.IdmAMax);
                }
            }

            vx = Mathf.Clamp(vx + a * dt, 0f, 50f);
            x += vx * dt;
            if (leader != null) lx += lv * dt;

            ref_[2 * k]     = x;
            ref_[2 * k + 1] = vx;
        }
        return ref_;
    }

    // ── Human reference (actual pedal input, AR(1) + truncated horizon) ─────
    //
    // Three research findings implemented here:
    //
    // 1. AR(1) decay toward IDM attractor (Flad et al. 2017; Vehicle Sys. Dyn. 2021):
    //      a[k] = α·a[k-1] + (1-α)·a_idm,   α = exp(-dt/τ),  τ = 1.51 s
    //    Steps 1-5 (~0.5 s): a[k] ≈ 0.72·a0 — driver intent dominates.
    //    Steps 6-20 (~2 s):  a[k] → a_idm   — physical attractor dominates.
    //
    // 2. Truncated human horizon at Np_human = 5 (Na, Cole & Li, Veh. Sys. Dyn. 2021):
    //    Empirically, drivers only plan ~0.5 s ahead in following/interaction tasks.
    //    Beyond step 5 the AR(1) value is frozen — no further decay.
    //    This prevents R2 from projecting implausible long-range intent.
    //
    // 3. IDM attractor is leader-aware (current gap, velocity):
    //    Computed once from current state; correctly handles car-following and free-road.

    public float[] HumanRef(AdvancedBicycleModel ego,
                             AdvancedBicycleModel leader,
                             MergePhase phase,
                             float targetVelocity,
                             int Np, float dt)
    {
        float aMax = _c.IdmAMax;
        float aMin = -8f;

        float x  = ego.GetX();
        float vx = ego.GetVx();

        // ── Step 1: measured acceleration at t=now ───────────────────────────
        float a0;
        if (VehicleInputs.Instance != null)
        {
            float thr = VehicleInputs.Instance.ThrottleInput;
            float brk = VehicleInputs.Instance.BrakeInput;

            // Approach + low-speed idle: IDM free-road prevents anchoring Nash at rest.
            // Merge/Following: always use real pedal input — even idle drag (≈−0.12 m/s²)
            // is the honest R2 signal and keeps the Nash game non-degenerate.
            if (phase == MergePhase.Approach && thr < 0.02f && brk < 0.02f && vx < 5f)
            {
                float vRatio = vx / Mathf.Max(targetVelocity, 1f);
                a0 = aMax * (1f - Mathf.Pow(Mathf.Clamp01(vRatio), _c.IdmDelta));
            }
            else
            {
                a0 = ego.InputsToAcceleration(thr, brk);
            }
            a0 = Mathf.Clamp(a0, aMin, aMax);
        }
        else
        {
            // No input device: Approach uses IDM free-road; Merge/Following uses drag
            // deceleration from (0,0) pedal (≈−0.12 m/s²) to keep R2 honest.
            if (phase == MergePhase.Approach)
            {
                float vRatio = vx / Mathf.Max(targetVelocity, 1f);
                a0 = aMax * (1f - Mathf.Pow(Mathf.Clamp01(vRatio), _c.IdmDelta));
            }
            else
            {
                a0 = ego.InputsToAcceleration(0f, 0f);
            }
            a0 = Mathf.Clamp(a0, aMin, aMax);
        }

        // ── Step 2: IDM attractor — long-term equilibrium for this driver/situation ─
        bool hasLeader = (leader != null) &&
                         (phase != MergePhase.Approach) &&
                         (phase != MergePhase.GapSearch);
        float vRatioIdm = vx / Mathf.Max(targetVelocity, 1f);
        float a_idm     = aMax * (1f - Mathf.Pow(Mathf.Clamp01(vRatioIdm), _c.IdmDelta));
        if (hasLeader)
        {
            float gap   = leader.GetX() - x - leader.GetLength();
            float dv    = vx - leader.GetVx();
            float sStar = _c.IdmS0 + vx * _c.IdmPlanT
                        + vx * dv / (2f * Mathf.Sqrt(Mathf.Max(aMax * _c.IdmPlanB, 1e-6f)));
            a_idm -= aMax * Mathf.Pow(Mathf.Max(sStar, 0f) / Mathf.Max(gap, 0.1f), 2f);
        }
        a_idm = Mathf.Clamp(a_idm, aMin, aMax);

        // ── Step 3: AR(1) rollout with horizon truncation at k=5 ────────────
        // α ≈ 0.936 per step (dt=0.1 s, τ=1.51 s)
        // k < 5  : AR(1) active — driver intent decays toward IDM attractor
        // k >= 5 : freeze at a[4] — human preview horizon exhausted (Na 2021)
        float ar1 = Mathf.Exp(-dt / 1.51f);
        float ak  = a0;
        float[] ref_ = new float[Np * 2];
        for (int k = 0; k < Np; k++)
        {
            if (k < 5)
                ak = Mathf.Clamp(ar1 * ak + (1f - ar1) * a_idm, aMin, aMax);
            // k >= 5: ak stays frozen at value reached at k=4

            vx = Mathf.Clamp(vx + ak * dt, 0f, targetVelocity * 1.2f);
            x += vx * dt;
            ref_[2 * k]     = x;
            ref_[2 * k + 1] = vx;
        }
        return ref_;
    }
}
