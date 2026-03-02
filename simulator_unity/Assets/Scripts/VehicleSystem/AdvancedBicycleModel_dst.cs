using System;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.XR;
using static SteeringSystem;

[RequireComponent(typeof(Rigidbody))]
public class AdvancedBicycleModel : MonoBehaviour
{
    float steeringInput;
    float throttleInput;
    float brakeInput;

    // Internal variables
    private float y = 0f;                          // Lateral position
    private float yDot = 0f;                       // Lateral velocity
    private float psi = 0f;                        // Yaw angle
    private float psiDot = 0f;                     // Yaw rate
    private float vx = 0;                          // Longitudinal velocity
    private float ax = 0;                          // Longitudinal acceleration
    private Vector4 state;                         // State vector [y, yDot, psi, psiDot]
    private float Fx_tot = 0f;                     // Total force
    private SteeringAngles angles = new SteeringAngles { inner = 0, outer = 0, average = 0, steeringWheelAngle = 0 };
    private VehicleParameters vehicleParams;
    public Powertrain powertrain { get; private set; }
    private BrakeSystem brakeSystem;
    private SteeringSystem steeringSystem;
    private Transmission transmission;
    private Rigidbody rb;
    private Matrix4x4 A;                          // State matrix
    private Vector4 B;                            // Input matrix
    private Vector3 position;                     // Global position

    // Wheel transforms for steering visualization
    public Transform frontLeftWheel;
    public Transform frontRightWheel;
    public Transform backLeftWheel;
    public Transform backRightWheel;
    public Transform SteeringWheel;

    public bool AutoMode = false; // Whether to use automatic driving mode
    public int IndexPlatoon=-1; // Index in platoon, -1 means not in a platoon
    private float Totaldistance = 0f; // Total distance traveled
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.isKinematic = true;  // We'll handle the physics ourselves
        position = transform.position;

        // Initialize vehicle systems
        vehicleParams = new VehicleParameters();

        // Get EngineAudio first
        EngineAudio engineAudio = GetComponent<EngineAudio>();
        if (engineAudio == null)
        {
            Debug.LogWarning("EngineAudio component not found!");
        }

        steeringInput = 0;
        throttleInput = 0;
        brakeInput = 0;

        // Initialize powertrain with engineAudio
        powertrain = new Powertrain(vehicleParams, engineAudio);

        // Get transmission reference directly from powertrain
        transmission = powertrain.GetTransmission();

        // Initialize other systems
        brakeSystem = new BrakeSystem(vehicleParams);
        steeringSystem = new SteeringSystem(vehicleParams);

        // Assign wheel transforms to the steering system
        steeringSystem.frontLeftWheel = frontLeftWheel;
        steeringSystem.frontRightWheel = frontRightWheel;
        steeringSystem.SteeringWheel = SteeringWheel;

        InitializeStateSpace();
        vx = 0f; // Starting from rest
        ax = 0f; // No acceleration initially

        powertrain.StartEngine(); // Start the engine

        // הוסף את רכיב לוח הבקרה אם לא קיים
        if (GetComponent<VehicleDashboard>() == null)
        {
            gameObject.AddComponent<VehicleDashboard>();
            Debug.Log("VehicleDashboard component added automatically.");
        }

        // וודא שיש DisplayManager בסצנה
        if (FindFirstObjectByType<DisplayManager>() == null)
        {
            GameObject displayManagerGO = new GameObject("DisplayManager");
            displayManagerGO.AddComponent<DisplayManager>();
            Debug.Log("DisplayManager created automatically.");
        }
    }

    void InitializeStateSpace()
    {
        // Initialize state-space matrices
        A = new Matrix4x4();

        // First row
        A.m00 = 0f; A.m01 = 1f; A.m02 = 0f; A.m03 = 0f;

        // Second row
        A.m10 = 0f;
        A.m11 = -(2f * vehicleParams.Caf + 2f * vehicleParams.Car) / (vehicleParams.mass * vx);
        A.m12 = 0f;
        A.m13 = -vx - (2f * vehicleParams.Caf * vehicleParams.lf - 2f * vehicleParams.Car * vehicleParams.lr) / (vehicleParams.mass * vx);

        // Third row
        A.m20 = 0f; A.m21 = 0f; A.m22 = 0f; A.m23 = 1f;

        // Fourth row
        A.m30 = 0f;
        A.m31 = -(2f * vehicleParams.lf * vehicleParams.Caf - 2f * vehicleParams.lr * vehicleParams.Car) / (vehicleParams.Iz * vx);
        A.m32 = 0f;
        A.m33 = -(2f * vehicleParams.lf * vehicleParams.lf * vehicleParams.Caf + 2f * vehicleParams.lr * vehicleParams.lr * vehicleParams.Car) / (vehicleParams.Iz * vx);

        B = new Vector4(
            0f,
            2f * vehicleParams.Caf / vehicleParams.mass,
            0f,
            2f * vehicleParams.lf * vehicleParams.Caf / vehicleParams.Iz
        );
    }

    void UpdateAngleWheels()
    {
        // Update wheel angular velocity
        float wheelAngularVelocity = vx / vehicleParams.wheelRadius;
        frontLeftWheel.Rotate(Vector3.right, wheelAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);
        frontRightWheel.Rotate(Vector3.right, wheelAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);
        backLeftWheel.Rotate(Vector3.right, wheelAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);
        backRightWheel.Rotate(Vector3.right, wheelAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        float Rxf = 0, Rxr = 0;

        if (AutoMode && IndexPlatoon >= 0)
        {
            // If in auto mode and part of a platoon, get inputs from the platoon manager
            var platoonManager = FindFirstObjectByType<PlatoonManager>();
            if (platoonManager != null)
            {
                var platoon = platoonManager.GetPlatoonVehicles()[IndexPlatoon]; // Get the platoon vehicle at this index
                if (platoon != null)
                {
                    //throttleInput = platoonManager.GetThrottleInput_Pl(IndexPlatoon);
                    //steeringInput = platoonManager.GetSteeringInput_Pl(IndexPlatoon);
                    //brakeInput = platoonManager.GetBrakeInput_Pl(IndexPlatoon);
                    throttleInput = 0f; // Reset throttle input
                    steeringInput = 0f; // Reset steering input
                    brakeInput = 0f; // Reset brake input
                }
            }
        }
        else
        {
            if (!AutoMode)
            {
                Debug.LogWarning("AutoMode is enabled but not in a platoon. Using manual inputs.");
                // Use manual inputs if AutoMode is enabled but not in a platoon
                throttleInput = VehicleInputs.Instance.ThrottleInput;
                steeringInput = VehicleInputs.Instance.SteeringInput;
                brakeInput = VehicleInputs.Instance.BrakeInput;
            }

        }

        
        // if (!PressPedal && (rawThrottleInput < 0.01f || rawBrakeInput < 0.01f))
        // {
        //     PressPedal = true;
        // }
        // if (PressPedal)
        // {
        //     throttleInput = rawThrottleInput;
        //     steeringInput = rawSteeringInput;
        //     brakeInput = rawBrakeInput;
        // }
        // else
        // {
        //     throttleInput = 0f;
        //     steeringInput = 0f;
        //     brakeInput = 0f;
        // }

        // Calculate forces - with accurate data
        float airDensity = 1.225f; // Air density at sea level
        // Air resistance with real data: Cd=0.30, A=2.09m²
        float Faero = 0.5f * airDensity * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;

        // Engine force now calculated with proper logic
        float Fxf = powertrain.CalculateEngineForce(throttleInput, vx);

        // Brake force
        float Fbrake = brakeSystem.CalculateBrakeForce(brakeInput, vx);

        // Steering angle
        angles = steeringSystem.CalculateAckermanSteering(steeringInput);

        // Rolling resistance for 225/50 R17 tires
        if (vx > 0.01f)
        {
            float rollingResistanceCoeff = 0.012f;
            float speedFactor = 1.0f + (vx * 0.008f);
            rollingResistanceCoeff *= speedFactor;

            Rxf = 0.5f * vehicleParams.mass * vehicleParams.gravity * rollingResistanceCoeff;
            Rxr = Rxf;
        }

        // Slope resistance
        float slopeRadians = vehicleParams.roadSlope * Mathf.Deg2Rad;
        float slopeForce = vehicleParams.mass * vehicleParams.gravity * Mathf.Sin(slopeRadians);

        // Natural deceleration - engine braking does most of the work
        float naturalDeceleration = 0f;
        if (throttleInput < 0.001f && brakeInput < 0.001f && vx > 0.1f)
        {
            naturalDeceleration = powertrain.CalculateNaturalDeceleration(vx, powertrain.EngineRpm);
            // Almost zero: Only minimal air resistance at very high speeds
            float additionalAirResistance = vx * vx * 0.01f;
            naturalDeceleration += additionalAirResistance;
        }

        // Total longitudinal forces
        Fx_tot = Fxf - Fbrake - Faero - Rxf - Rxr - slopeForce - naturalDeceleration;

        
        if (AutoMode)// If in autonomous mode, update engine RPM based on platoon manager           
        {
            // Update engine RPM based on throttle input and current speed
            var platoonManager = FindFirstObjectByType<PlatoonManager>();
            if (platoonManager != null)
            {
                // In auto mode, get acceleration from platoon manager
                float desiredAcceleration = platoonManager.GetAccelerationVehicle(IndexPlatoon);
                // Calculate total force based on desired acceleration
                Fx_tot = desiredAcceleration * vehicleParams.mass;
                // Calculate engine force needed (excluding resistance forces)
                float engineForce = Fx_tot + Faero + Rxf + Rxr + slopeForce + naturalDeceleration;
                // Update engine RPM according to required engine force
                powertrain.UpdateRpmForAutonomousMode(engineForce, vx);
                
            }
        }

        // Longitudinal acceleration (F = ma)
        ax = Fx_tot / vehicleParams.mass;
        vx += ax * dt;
        // Limit speed to maximum speed from specs: 155 mph = 249 km/h
        float maxSpeedMS = vehicleParams.maxVelocity / 3.6f;
        vx = Mathf.Clamp(vx, 0, maxSpeedMS);

        if (vx >= maxSpeedMS)
        {
            ax = 0;
        }

        // Update state-space matrices only if speed is significant
        if (Mathf.Abs(vx) > 0.001f)
        {
            InitializeStateSpace();
            Vector4 stateDot = A * state + B * angles.average;
            state += stateDot * dt;

            y = state[0];
            yDot = state[1];
            psi = state[2];
            psiDot = state[3];
        }

        // Update global position
        float cosRot = Mathf.Cos(psi);
        float sinRot = Mathf.Sin(psi);
        position.z += (vx * cosRot - yDot * sinRot) * dt;
        position.x += (vx * sinRot + yDot * cosRot) * dt;

        Totaldistance+=math.sqrt(math.pow(vx * dt, 2) + math.pow(yDot * dt, 2));

        // Apply updates to transform
        transform.position = position;
        transform.rotation = Quaternion.Euler(0f, psi * Mathf.Rad2Deg, 0f);

        // Rotate front wheels based on Ackerman steering angles
        steeringSystem.UpdateWheelRotation(angles);

        // Update wheel angular velocity
        UpdateAngleWheels();
    }

    // Public getters for other scripts to access the internal state
    public Vector3 GetPosition() => position;
    public float GetVx() => vx;
    public float SetVx(float newVx) => vx = newVx; // Allows setting velocity directly, useful for testing
    public float GetAx() => ax;
    public float SetAx(float newAx) => ax = newAx;
    public float GetY() => y;
    public float GetYDot() => yDot;
    public float GetPsi() => psi;
    public float GetPsiDot() => psiDot;
    public float GetEngineRpm() => powertrain.EngineRpm;
    public float GetFx_tot() => Fx_tot;
    public float GetLambda() => angles.average;
    public int GetIndexPlatoon() => IndexPlatoon;
    public void SetIndexPlatoon(int index) => IndexPlatoon = index;
    public bool IsAutoMode() => AutoMode;

    // Additional getters for dashboard and systems access
    public Transmission GetTransmission() => transmission;
    public VehicleParameters GetVehicleParameters() => vehicleParams;
    public BrakeSystem GetBrakeSystem() => brakeSystem;
    public SteeringSystem GetSteeringSystem() => steeringSystem;
    public SteeringAngles GetCurrentSteeringAngles() => angles;
    public int GetCurrentGear() => transmission.GetCurrentGear();
    public float GetTotalDistance() => Totaldistance;

    // Input getters for dashboard
    public float GetThrottleInput() => throttleInput;
    public float GetBrakeInput() => brakeInput;
    public float GetSteeringInput() => steeringInput;
    public float GetSteeringWheelAngle() => angles.steeringWheelAngle;
}