// AuthorityAllocator — maps safety force + gap/velocity errors → Nash authority ratio λ.
//
// Mirrors coordinator.py _long_authority() and _lat_authority():
//
//   Long three-stage:
//     1. Safety sigmoid:   λ_s  = f(|risk_force|, ForceMidpoint, Steepness)
//     2. Gap+vel Eq.15:    l_n  = min(1-|gap_err|/GapErrMax, 1-|vel_err|/VelErrMax)
//                          γ₁  = 1/(1+exp(M1·(-l_n+M2)));  λ_gap = (1-γ₁)/γ₁
//     3. Fusion + adaptive EMA:  λ = EMA(max(λ_s, λ_gap))
//        In FOLLOWING:  cap at LongLambdaMaxFollowing, alpha = LongAlphaFollowing
//
//   Lat three-stage (Swain & Rath 2023, Eq. 15):
//     1. Safety sigmoid on |force|
//     2. Lane-offset sigmoid: l_n = (LaneHalfWidth - |y_err|) / LaneHalfWidth
//     3. Fusion + adaptive EMA on |y_error|
using UnityEngine;

public class AuthorityAllocator
{
    readonly AuthorityConfig   _a;
    readonly SafetyFieldConfig _s;  // for LaneWidth

    float _longLambda;
    float _latLambda;

    public AuthorityAllocator(AuthorityConfig auth, SafetyFieldConfig sf)
    {
        _a          = auth;
        _s          = sf;
        _longLambda = auth.LongLambdaMin;
        _latLambda  = auth.LatLambdaMin;
    }

    // ── Longitudinal authority ────────────────────────────────────────────────

    public float ComputeLong(float force, float gapError, float velError, MergePhase phase)
    {
        // 1. Safety sigmoid
        float exponent  = -_a.LongSteepness * (Mathf.Abs(force) - _a.LongForceMidpoint);
        float lamSafety = (_a.LongLambdaMax - _a.LongLambdaMin) / (1f + Mathf.Exp(exponent))
                         + _a.LongLambdaMin;

        // 2. Gap + velocity Swain & Rath Eq. 15
        float absGap = Mathf.Abs(gapError);
        float absVel = Mathf.Abs(velError);
        float lnGap  = (_a.AuthGapErrorMax - absGap) / _a.AuthGapErrorMax;
        float lnVel  = (_a.AuthVelErrorMax - absVel) / _a.AuthVelErrorMax;
        float ln     = Mathf.Clamp(Mathf.Min(lnGap, lnVel), 0f, 1f);

        float gamma1  = 1f / (1f + Mathf.Exp(_a.SigmoidM1 * (-ln + _a.SigmoidM2)));
        float gamma2  = 1f - gamma1;
        float lamGap  = gamma2 / Mathf.Max(gamma1, 1e-6f);

        // 3. Fusion
        float lamRaw = Mathf.Min(Mathf.Max(lamSafety, lamGap), _a.LongLambdaMax);

        // FOLLOWING cap
        if (phase == MergePhase.Following)
            lamRaw = Mathf.Min(lamRaw, _a.LongLambdaMaxFollowing);

        // Adaptive EMA
        float combinedErr = Mathf.Max(absGap, absVel);
        float alpha;
        if (phase == MergePhase.Following)
            alpha = _a.LongAlphaFollowing;
        else if (combinedErr > 5f)
            alpha = _a.LongAlphaFast;
        else if (combinedErr > 2f)
            alpha = _a.LongAlphaBase
                  + (_a.LongAlphaFast - _a.LongAlphaBase) * (combinedErr - 2f) / 3f;
        else
            alpha = _a.LongAlphaBase;

        _longLambda = alpha * lamRaw + (1f - alpha) * _longLambda;
        return _longLambda;
    }

    // ── Lateral authority ─────────────────────────────────────────────────────

    public float ComputeLat(float force, float yError)
    {
        // 1. Safety sigmoid on risk force
        float expS = -_a.LatSteepness * (Mathf.Abs(force) - _a.LatForceMidpoint);
        float lamS = (_a.LatLambdaMax - _a.LatLambdaMin) / (1f + Mathf.Exp(expS))
                    + _a.LatLambdaMin;

        // 2. Lane-offset sigmoid (Swain & Rath Eq. 15)
        float laneHalf = _s.LaneWidth / 2f;
        float ln       = Mathf.Clamp((laneHalf - Mathf.Abs(yError)) / laneHalf, 0f, 1f);

        float gamma1 = 1f / (1f + Mathf.Exp(_a.SigmoidM1 * (-ln + _a.SigmoidM2)));
        float gamma2 = 1f - gamma1;
        float lamP   = Mathf.Clamp(gamma2 / Mathf.Max(gamma1, 1e-6f),
                                   _a.LatLambdaMin, _a.LatLambdaMax);

        float lamRaw = Mathf.Max(lamS, lamP);

        // Adaptive EMA on |y_error|
        float eAbs = Mathf.Abs(yError);
        float alpha;
        if (eAbs > _a.LatAlphaFastThr)
            alpha = _a.LatAlphaFast;
        else if (eAbs < _a.LatAlphaBaseThr)
            alpha = _a.LatAlphaBase;
        else
        {
            float blend = (eAbs - _a.LatAlphaBaseThr)
                        / (_a.LatAlphaFastThr - _a.LatAlphaBaseThr);
            alpha = _a.LatAlphaBase + (_a.LatAlphaFast - _a.LatAlphaBase) * blend;
        }

        _latLambda = alpha * lamRaw + (1f - alpha) * _latLambda;
        return _latLambda;
    }
}
