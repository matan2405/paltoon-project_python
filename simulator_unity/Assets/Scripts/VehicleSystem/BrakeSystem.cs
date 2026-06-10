using UnityEngine;

public class BrakeSystem
{
    private VehicleParameters vehicleParams;

    public BrakeSystem(VehicleParameters vehicleParams)
    {
        this.vehicleParams = vehicleParams;
    }

    // Calculate brake force based on brake input and vehicle velocity
    public float CalculateBrakeForce(float brakeInput, float vx)
    {
        // Calculate maximum possible brake force based on tire grip
        float staticWeight = vehicleParams.mass * vehicleParams.gravity;
        float maxBrakeForce = staticWeight * vehicleParams.tireFrictionCoefficient; // Maximum theoretical brake force
        
        // Calculate weight transfer during braking
        float deceleration = brakeInput * vehicleParams.gravity * 1.25f; // Maximum deceleration around 1.25*g
        float weightTransfer = (vehicleParams.mass * deceleration * vehicleParams.height * 0.5f) / vehicleParams.wheelbase;

        // Calculate normal forces with weight transfer
        float frontNormalForce = (staticWeight * vehicleParams.lr / vehicleParams.wheelbase) + weightTransfer;
        float rearNormalForce = (staticWeight * vehicleParams.lf / vehicleParams.wheelbase) - weightTransfer;

        // Calculate maximum brake forces per axle based on normal force and friction
        float maxFrontBrakeForce = frontNormalForce * vehicleParams.tireFrictionCoefficient;
        float maxRearBrakeForce = rearNormalForce * vehicleParams.tireFrictionCoefficient;

        // Apply brake balance and pedal input
        float actualFrontBrakeForce = maxFrontBrakeForce * vehicleParams.brakeBalance * brakeInput;
        float actualRearBrakeForce = maxRearBrakeForce * (1 - vehicleParams.brakeBalance) * brakeInput;

        // Total brake force
        return actualFrontBrakeForce + actualRearBrakeForce;
    }

    // Returns (front, rear) brake forces separately.
    // Used by Derivatives6D to correctly assign Fxf/Fxr per Belousov Eq. 3.11b/c:
    // only the front contact-patch force enters the sinδ terms.
    public (float front, float rear) CalculateBrakeForceSplit(float brakeInput, float vx)
    {
        float staticWeight    = vehicleParams.mass * vehicleParams.gravity;
        float deceleration    = brakeInput * vehicleParams.gravity * 1.25f;
        float weightTransfer  = (vehicleParams.mass * deceleration * vehicleParams.height * 0.5f)
                              / vehicleParams.wheelbase;

        float frontNormalForce = (staticWeight * vehicleParams.lr / vehicleParams.wheelbase) + weightTransfer;
        float rearNormalForce  = (staticWeight * vehicleParams.lf / vehicleParams.wheelbase) - weightTransfer;

        float front = frontNormalForce * vehicleParams.tireFrictionCoefficient
                    * vehicleParams.brakeBalance * brakeInput;
        float rear  = rearNormalForce  * vehicleParams.tireFrictionCoefficient
                    * (1f - vehicleParams.brakeBalance) * brakeInput;

        return (front, rear);
    }
}