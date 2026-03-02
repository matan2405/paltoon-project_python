using UnityEngine;

public class VehicleParameters 
{
    [Header("Audi TT 2.0 TFSI - Official Specifications")]
    public float steeringRatio = 14.6f;       // From specs: 14.6:1
    public float maxSteeringWheelAngle = 450f; // Typical ±450 degrees (1.25 turns each way)
    public float maxWheelAngle = 30.0f;       // Maximum wheel angle in degrees
    public float maxSteeringAngle = 40.0f;    // Maximum steering wheel angle in degrees
    public float maxVelocity = 249f;          // From specs: 155 mph = 249 km/h

    [Header("Audi TT Vehicle Parameters - From Official Specs")]
    public float mass = 1305f;                // From specs: 1305 kg (including driver)
    public float height = 1.353f;             // From specs: 1353mm

    public float width = 1.832f;              // From specs: 1832mm
    public float length = 4.177f;             // From specs: 4177mm
    public float Iz = 2500f;                  // Moment of inertia around Z axis (estimated)
    public float wheelbase = 2.505f;          // From specs: 2505mm
    public float lf = 1.2525f;                // Distance from CG to front axle (wheelbase/2)
    public float lr = 1.2525f;                // Distance from CG to rear axle (wheelbase/2)
    public float trackWidth = 1.572f;         // From specs: 1572mm (front)
    public float Caf = 15000f;                // Front tire cornering stiffness
    public float Car = 18000f;                // Rear tire cornering stiffness

    [Header("Environmental Parameters")]
    public float gravity = 9.81f;             // Gravity acceleration
    public float roadSlope = 0f;              // Road slope angle (theta) in degrees
    public float dragCoefficient = 0.30f;     // From specs: 0.30
    public float frontalArea = 2.09f;         // From specs: 2.09 m²

    [Header("Wheel Parameters - 225/50 R17 (From Specs)")]
    public float wheelRadius = 0.3175f;       // Calculated: 225mm width, 50% profile, 17" rim
    public float wheelAngularVelocity = 0f;
    public float wheelInertia = 1.5f;         // Approximated wheel rotational inertia

    [Header("Brake System Parameters - From Specs")]
    public float frontBrakeDiscDiameter = 0.312f;  // From specs: 312mm front
    public float rearBrakeDiscDiameter = 0.300f;   // From specs: 300mm rear
    public float brakeDiscFrictionCoefficient = 0.4f;  // Brake pad friction coefficient
    public float brakeBalance = 0.7f;              // Front/rear brake force distribution
    public float tireFrictionCoefficient = 0.8f;   // Tire-road friction coefficient (dry asphalt)
}