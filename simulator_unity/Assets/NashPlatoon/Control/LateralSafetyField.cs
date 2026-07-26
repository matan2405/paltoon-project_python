// LateralSafetyField — Wang et al. (2015/2016) / Li et al. (2019) elliptic DSF.
//
// Wang 2016 Eq.17:  M  = DsfVehicleMass * (SpeedCoeff * |v|^SpeedExp + SpeedOffset)
// Li 2019 Eq.12-13: r* = sqrt((dx/a)^2 + (dy/b)^2);  a = max(|Δv|·Ts, AMin), b = Tau
// Li 2019 Eq.11:    E  = G * R_obs * M_obs / r*
// Li 2019 Eq.18:    E *= exp(K1 * v_obs * cos(θ_v))        (kinetic correction, obstacle)
// Li 2019 Eq.21:    Fr = E * M_ego * R_ego * exp(-K2 * v_ego * cos(θ_i)) * (1 + DR)
// Direction:        Fr *= tanh(dy / σ)  (smooth lateral sign; σ=LatDsfSigma=0.5m, mirrors Python lateral_safety_field.py)
// Road boundaries:  1/dist repulsion near lane edges
// Output:           EMA-filtered and clamped to ±LatMaxForce
using UnityEngine;

public class LateralSafetyField
{
    readonly SafetyFieldConfig _c;
    float _forceFilt;

    public LateralSafetyField(SafetyFieldConfig cfg) => _c = cfg;

    public float Compute(AdvancedBicycleModel ego, AdvancedBicycleModel[] platoon)
    {
        float egoX  = ego.GetPosition().z;
        float egoY  = ego.GetY();
        float egoVx = Mathf.Max(ego.GetVx(), 1f);

        // Virtual mass of ego (Wang 2016, Eq. 17)
        float mEgo = _c.DsfVehicleMass
                   * (_c.DsfSpeedCoeff * Mathf.Pow(Mathf.Abs(egoVx), _c.DsfSpeedExponent)
                      + _c.DsfSpeedOffset);

        float total = 0f;

        foreach (var pv in platoon)
        {
            if (pv == null) continue;
            float obsX  = pv.GetPosition().z;
            float obsY  = pv.GetY();
            float obsVx = pv.GetVx();

            float dx = egoX - obsX;    // positive = ego ahead
            float dy = egoY - obsY;    // positive = ego LEFT of obstacle (Belousov convention)

            // Elliptic distance (Li 2019, Eq. 12-13)
            float a  = Mathf.Max(Mathf.Abs(egoVx - obsVx) * _c.LatDsfTs, _c.LatDsfAMin);
            float b  = _c.LatDsfTau;
            float dxa = dx / a;
            float dyb = dy / b;
            float rStar = Mathf.Max(Mathf.Sqrt(dxa * dxa + dyb * dyb), _c.DsfEpsilon);

            // Elliptic gradient unit vector (r* direction, used for kinetic correction)
            float rStarUx = (dx / a) / rStar;
            float rStarUy = (dy / b) / rStar;

            // Virtual mass of obstacle (Wang 2016, Eq. 17)
            float mObs = _c.DsfVehicleMass
                       * (_c.DsfSpeedCoeff * Mathf.Pow(Mathf.Abs(obsVx), _c.DsfSpeedExponent)
                          + _c.DsfSpeedOffset);

            // Field strength (Li 2019, Eq. 11) — includes R_obs road condition factor
            float E = _c.LatDsfG * _c.LatRoadConditionObs * mObs / rStar;

            // Kinetic correction (Li 2019, Eq. 18): obstacle velocity projected onto gradient
            float obsVxDir  = Mathf.Abs(obsVx) > 0.1f ? 1f : 0f;  // unit vec [1,0] if moving
            float cosThetaV = obsVxDir * rStarUx;                   // dot([1,0], rStarU)
            E *= Mathf.Exp(_c.LatDsfK1 * Mathf.Abs(obsVx) * cosThetaV);

            // Elliptic gradient ∇(r*²) (unnormalised) — used only for cos(θ_i) (Eq. 21)
            float fcx = dx / (a * a);
            float fcy = dy / (b * b);
            float fcNorm = Mathf.Sqrt(fcx * fcx + fcy * fcy);
            float fcUx = fcNorm > 1e-9f ? fcx / fcNorm : 0f;

            // cos(θ_i): ego velocity [1,0] projected onto gradient direction (Eq. 21)
            float cosThetaI = fcUx;

            // Field force (Li 2019, Eq. 21) — includes R_ego road condition factor
            float Fr = E * mEgo * _c.LatRoadConditionEgo
                     * Mathf.Exp(-_c.LatDsfK2 * Mathf.Abs(egoVx) * cosThetaI)
                     * (1f + _c.LatDsfDr);

            // Lateral direction: tanh(dy/σ) — smooth sign function (mirrors Python lateral_safety_field.py).
            // Li 2019 Eq.21 gives |Fr| only; this maps dy→±1 continuously so Fr=0 when dy=0
            // (no lateral bias when ego is directly ahead/behind) and saturates beyond σ=0.5m.
            float latDir = (float)System.Math.Tanh(dy / _c.LatDsfSigma);
            total += Fr * latDir;
        }

        // Road boundary forces
        float distRight = _c.RoadHalfWidth - egoY;
        if (distRight > 0f && distRight < _c.BoundaryProximity)
        {
            float pot = 1f / Mathf.Max(distRight, _c.BoundaryEpsilon);
            total -= _c.BoundaryGain * (float)System.Math.Tanh(pot / _c.BoundaryScale);
        }
        float distLeft = _c.RoadHalfWidth + egoY;
        if (distLeft > 0f && distLeft < _c.BoundaryProximity)
        {
            float pot = 1f / Mathf.Max(distLeft, _c.BoundaryEpsilon);
            total += _c.BoundaryGain * (float)System.Math.Tanh(pot / _c.BoundaryScale);
        }

        total = Mathf.Clamp(total, -_c.LatMaxForce, _c.LatMaxForce);
        _forceFilt = _c.LatEmaAlpha * total + (1f - _c.LatEmaAlpha) * _forceFilt;
        return _forceFilt;
    }
}
