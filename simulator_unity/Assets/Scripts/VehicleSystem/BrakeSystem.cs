using UnityEngine;

public class BrakeSystem
{
    private VehicleParameters vehicleParams;

    public BrakeSystem(VehicleParameters vehicleParams)
    {
        this.vehicleParams = vehicleParams;
    }

    // Calculate brake force based on brake input and vehicle velocity.
    // Ideal brake distribution (Rajamani §4.3): each axle is independently limited
    // to μ·N_axle, where N_axle is the instantaneous normal force after weight
    // transfer.  This removes the fixed brakeBalance split and allows the total
    // braking force to reach μ·(Nf+Nr) = μ·m·g ≈ 7.85 m/s² at full pedal.
    //
    // Implementation uses the same one-step weight-transfer approximation as before,
    // but the per-axle force is now brakeInput·μ·N_axle (not balance·brakeInput·μ·N).
    public float CalculateBrakeForce(float brakeInput, float vx)
    {
        float g            = vehicleParams.gravity;
        float m            = vehicleParams.mass;
        float mu           = vehicleParams.tireFrictionCoefficient;
        float staticWeight = m * g;

        // Static axle loads
        float Nf0 = staticWeight * vehicleParams.lr / vehicleParams.wheelbase;
        float Nr0 = staticWeight * vehicleParams.lf / vehicleParams.wheelbase;

        // First-order weight-transfer estimate using static loads
        float F0_static      = (Nf0 + Nr0) * mu * brakeInput;   // ideal upper bound
        float decel0         = F0_static / m;
        float weightTransfer = (m * decel0 * vehicleParams.height * 0.5f) / vehicleParams.wheelbase;

        float frontNormalForce = Nf0 + weightTransfer;
        float rearNormalForce  = Mathf.Max(Nr0 - weightTransfer, 0f);

        // Ideal distribution: each axle at its own friction limit (Rajamani §4.3)
        float actualFrontBrakeForce = frontNormalForce * mu * brakeInput;
        float actualRearBrakeForce  = rearNormalForce  * mu * brakeInput;

        return actualFrontBrakeForce + actualRearBrakeForce;
    }

    // Returns (front, rear) brake forces separately.
    // Used by Derivatives6D to correctly assign Fxf/Fxr per Belousov Eq. 3.11b/c:
    // only the front contact-patch force enters the sinδ terms.
    // Uses ideal brake distribution (same as CalculateBrakeForce).
    public (float front, float rear) CalculateBrakeForceSplit(float brakeInput, float vx)
    {
        float g            = vehicleParams.gravity;
        float m            = vehicleParams.mass;
        float mu           = vehicleParams.tireFrictionCoefficient;
        float staticWeight = m * g;

        float Nf0 = staticWeight * vehicleParams.lr / vehicleParams.wheelbase;
        float Nr0 = staticWeight * vehicleParams.lf / vehicleParams.wheelbase;

        float F0_static      = (Nf0 + Nr0) * mu * brakeInput;
        float decel0         = F0_static / m;
        float weightTransfer = (m * decel0 * vehicleParams.height * 0.5f) / vehicleParams.wheelbase;

        float frontNormalForce = Nf0 + weightTransfer;
        float rearNormalForce  = Mathf.Max(Nr0 - weightTransfer, 0f);

        float front = frontNormalForce * mu * brakeInput;
        float rear  = rearNormalForce  * mu * brakeInput;

        return (front, rear);
    }
}