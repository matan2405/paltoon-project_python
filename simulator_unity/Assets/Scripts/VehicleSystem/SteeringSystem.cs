using UnityEngine;

public class SteeringSystem
{
    private VehicleParameters vehicleParams;
    public Transform frontLeftWheel;
    public Transform frontRightWheel;
    public Transform SteeringWheel;

    public struct SteeringAngles
    {
        public float inner;  // Inner wheel steering angle
        public float outer;  // Outer wheel steering angle
        public float average; // Average angle (bicycle model approximation)
        public float steeringWheelAngle; // Actual steering wheel angle
    }
    
    public SteeringSystem(VehicleParameters vehicleParams)
    {
        this.vehicleParams = vehicleParams;
    }

    // Calculate Ackerman steering angles based on steering input
    public SteeringAngles CalculateAckermanSteering(float steerInput)
    {
        SteeringAngles angles = new SteeringAngles();

        angles.steeringWheelAngle = steerInput * vehicleParams.maxSteeringWheelAngle;
        float wheelAngleDegrees = angles.steeringWheelAngle / vehicleParams.steeringRatio;
        wheelAngleDegrees = Mathf.Clamp(wheelAngleDegrees, -vehicleParams.maxWheelAngle, vehicleParams.maxWheelAngle);
        float baseWheelAngle = wheelAngleDegrees * Mathf.Deg2Rad;
        float turnRadius = vehicleParams.wheelbase / Mathf.Tan(Mathf.Abs(baseWheelAngle));

        if (baseWheelAngle > 0)
        {
            angles.inner = Mathf.Atan(vehicleParams.wheelbase / (turnRadius - (vehicleParams.trackWidth / 2f)));
            angles.outer = Mathf.Atan(vehicleParams.wheelbase / (turnRadius + (vehicleParams.trackWidth / 2f)));
        }
        else
        {
            angles.inner = -Mathf.Atan(vehicleParams.wheelbase / (turnRadius - (vehicleParams.trackWidth / 2f)));
            angles.outer = -Mathf.Atan(vehicleParams.wheelbase / (turnRadius + (vehicleParams.trackWidth / 2f)));
        }

        angles.average = (angles.inner + angles.outer) / 2f;
        return angles;
    }

    // Update wheel rotation based on calculated steering angles
    public void UpdateWheelRotation(SteeringAngles angles)
    {
        if (frontLeftWheel != null)
            frontLeftWheel.localRotation = Quaternion.Euler(0f, angles.inner * Mathf.Rad2Deg, 0f);
        if (frontRightWheel != null)
            frontRightWheel.localRotation = Quaternion.Euler(0f, angles.outer * Mathf.Rad2Deg, 0f);
        // if (SteeringWheel != null)
        //     SteeringWheel.localRotation = Quaternion.Euler(0f, angles.steeringWheelAngle, 0f);
    }
}