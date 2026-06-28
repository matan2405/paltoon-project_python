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
    // Lateral (cornering) stiffness — per single wheel [N/rad].
    // Belousov Eq. 3.12–3.13 use per-axle (2·Cα), so Derivatives6D must multiply by 2.
    // Source: load-based estimate (Pacejka rule: Cα ≈ 9.2·Fz at nominal load).
    //   Audi TT FWD, mass=1305 kg, 62/38 weight split:
    //   Fz_front_wheel ≈ 1305·0.62·9.81/2 ≈ 3,963 N  → Caf ≈ 9.2·3963 ≈ 36,500 N/rad
    //   Fz_rear_wheel  ≈ 1305·0.38·9.81/2 ≈  2,428 N  → Car ≈ 9.2·2428 ≈ 22,300 N/rad
    // Cross-checked against Rajamani "Vehicle Dynamics and Control" Ch.2 (compact FWD range).
    //
    // Longitudinal stiffness (Cs) is NOT stored here because Fx is computed directly
    // from engine/brake forces — slip-based longitudinal tire model is not used.
    public float Caf = 36500f;               // Front tire lateral (cornering) stiffness, per wheel [N/rad]
    public float Car = 22300f;               // Rear  tire lateral (cornering) stiffness, per wheel [N/rad]

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