// LateralSafetyField — Wang et al. (2015/2016) / Li et al. (2019) elliptic DSF.
//
// Mirrors coordinator.py _lat_safety_force():
//   Virtual mass:     M = DsfVehicleMass * (SpeedCoeff * |v|^SpeedExp + SpeedOffset)
//   Elliptic dist:    r* = sqrt((dx/a)^2 + (dy/b)^2);  a = max(|Δv|·Ts, AMin), b = Tau
//   Field strength:   E  = G * M_obs / r*
//   Kinetic correction: E *= exp(K1 * v_obs * cos(θ_v))
//   Field force:      Fr = E * M_ego * exp(-K2 * v_ego * cos(θ_i)) * (1 + DR)
//   Direction:        Fr *= tanh(dy / sigma)  (sign: pushes ego away from obstacle)
//   Road boundaries:  1/dist repulsion near lane edges
//   Output:           EMA-filtered and clamped to ±LatMaxForce
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
            float dy = egoY - obsY;    // positive = ego to right

            // Elliptic distance (Li 2019, Eq. 12-13)
            float a  = Mathf.Max(Mathf.Abs(egoVx - obsVx) * _c.LatDsfTs, _c.LatDsfAMin);
            float b  = _c.LatDsfTau;
            float dxa = dx / a;
            float dyb = dy / b;
            float rStar = Mathf.Max(Mathf.Sqrt(dxa * dxa + dyb * dyb), _c.DsfEpsilon);

            // Elliptic gradient unit vector
            float rStarUx = (dx / a) / rStar;
            float rStarUy = (dy / b) / rStar;

            // Virtual mass of obstacle
            float mObs = _c.DsfVehicleMass
                       * (_c.DsfSpeedCoeff * Mathf.Pow(Mathf.Abs(obsVx), _c.DsfSpeedExponent)
                          + _c.DsfSpeedOffset);

            // Field strength (Li 2019, Eq. 11)
            float E = _c.LatDsfG * mObs / rStar;

            // Kinetic correction (Eq. 18): obstacle velocity along gradient
            float obsVxDir = obsVx > 0.1f ? 1f : 0f;  // unit vec [1,0] if moving
            float cosThetaV = obsVxDir * rStarUx;      // dot([1,0], rStarU)
            E *= Mathf.Exp(_c.LatDsfK1 * Mathf.Abs(obsVx) * cosThetaV);

            // Field force (Eq. 21): ego projection on Cartesian gradient
            float fcx = dx / (a * a);
            float fcy = dy / (b * b);
            float fcNorm = Mathf.Sqrt(fcx * fcx + fcy * fcy);
            float fcUx = fcNorm > 1e-9f ? fcx / fcNorm : 0f;
            // dot([1,0], fcU) — ego moves along x-axis
            float cosThetaI = fcUx;

            float Fr = E * mEgo
                     * Mathf.Exp(-_c.LatDsfK2 * Mathf.Abs(egoVx) * cosThetaI)
                     * (1f + _c.LatDsfDr);

            // Lateral direction: smooth tanh (push ego away from obstacle)
            float latDir = (float)System.Math.Tanh(dy / _c.LatDsfSigma);
            total += Fr * latDir;
        }

        // Road boundary forces (mirrors coordinator _lat_safety_force)
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
