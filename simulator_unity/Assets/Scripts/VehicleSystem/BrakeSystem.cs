using UnityEngine;

public class BrakeSystem
{
    private VehicleParameters vehicleParams;

    public BrakeSystem(VehicleParameters vehicleParams)
    {
        this.vehicleParams = vehicleParams;
    }

    // Calculate brake force based on brake input and vehicle velocity.
    // Level 1 (physical): friction-limited by μ·N per axle, with quasi-static
    // weight transfer derived from the actual braking deceleration (iterative
    // one-step approximation).  The old code used a hardcoded 1.25 g for the
    // weight-transfer estimate, which created a circular ceiling and capped
    // Fbrake_max at ~50 % of the physical limit.
    public float CalculateBrakeForce(float brakeInput, float vx)
    {
        float g            = vehicleParams.gravity;
        float m            = vehicleParams.mass;
        float mu           = vehicleParams.tireFrictionCoefficient;
        float staticWeight = m * g;

        // Static axle loads (symmetric wheelbase assumed)
        float Nf0 = staticWeight * vehicleParams.lr / vehicleParams.wheelbase;
        float Nr0 = staticWeight * vehicleParams.lf / vehicleParams.wheelbase;

        // Maximum friction-limited brake force with static loads (no weight transfer)
        float Ff0 = Nf0 * mu * vehicleParams.brakeBalance       * brakeInput;
        float Fr0 = Nr0 * mu * (1f - vehicleParams.brakeBalance) * brakeInput;
        float F0  = Ff0 + Fr0;

        // One-step weight-transfer correction: use the deceleration implied by F0
        // instead of a hardcoded 1.25 g.  This avoids the circular assumption and
        // correctly scales with μ, brake balance, and pedal input.
        float decel        = F0 / m;   // first-order estimate [m/s²]
        float weightTransfer = (m * decel * vehicleParams.height * 0.5f) / vehicleParams.wheelbase;

        float frontNormalForce = Nf0 + weightTransfer;
        float rearNormalForce  = Mathf.Max(Nr0 - weightTransfer, 0f);   // rear never negative

        float actualFrontBrakeForce = frontNormalForce * mu * vehicleParams.brakeBalance       * brakeInput;
        float actualRearBrakeForce  = rearNormalForce  * mu * (1f - vehicleParams.brakeBalance) * brakeInput;

        return actualFrontBrakeForce + actualRearBrakeForce;
    }

    // Returns (front, rear) brake forces separately.
    // Used by Derivatives6D to correctly assign Fxf/Fxr per Belousov Eq. 3.11b/c:
    // only the front contact-patch force enters the sinδ terms.
    // Uses the same one-step weight-transfer approximation as CalculateBrakeForce.
    public (float front, float rear) CalculateBrakeForceSplit(float brakeInput, float vx)
    {
        float g            = vehicleParams.gravity;
        float m            = vehicleParams.mass;
        float mu           = vehicleParams.tireFrictionCoefficient;
        float staticWeight = m * g;

        float Nf0 = staticWeight * vehicleParams.lr / vehicleParams.wheelbase;
        float Nr0 = staticWeight * vehicleParams.lf / vehicleParams.wheelbase;

        float Ff0 = Nf0 * mu * vehicleParams.brakeBalance       * brakeInput;
        float Fr0 = Nr0 * mu * (1f - vehicleParams.brakeBalance) * brakeInput;
        float decel        = (Ff0 + Fr0) / m;
        float weightTransfer = (m * decel * vehicleParams.height * 0.5f) / vehicleParams.wheelbase;

        float frontNormalForce = Nf0 + weightTransfer;
        float rearNormalForce  = Mathf.Max(Nr0 - weightTransfer, 0f);

        float front = frontNormalForce * mu * vehicleParams.brakeBalance       * brakeInput;
        float rear  = rearNormalForce  * mu * (1f - vehicleParams.brakeBalance) * brakeInput;

        return (front, rear);
    }
}