//using System;
//using System.Linq;
//using Unity.Mathematics;
//using UnityEngine;


//[RequireComponent(typeof(Rigidbody))]
//public class AdvancedBicycleModel : MonoBehaviour
//{
//    public LogitechG29Controller controller;
//    public LogitechG29Connection connection;
//    private struct SteeringAngles
//    {
//        public float inner;  // Inner wheel steering angle
//        public float outer;  // Outer wheel steering angle
//        public float average; // Average angle (bicycle model approximation)
//        public float steeringWheelAngle; // Actual steering wheel angle
//    }

//    float steeringInput;
//    float throttleInput;
//    float brakeInput;
//    [Header("Audi TT Vehicle Steer And Velocity Parameters")]
//    public float steeringRatio = 14.6f;       // From Audi TT specs: 14.6:1
//    public float maxSteeringWheelAngle = 450f; // Typical ±450 degrees (1.25 turns each way)
//    public float maxWheelAngle = 30.0f;       // Maximum wheel angle in degrees
//    public float maxSteeringAngle = 40.0f;    // Maximum steering wheel angle in degrees
//    public float maxVelocity = 120f;          // Maximum velocity in km/h

//    [Header("Audi TT Vehicle Parameters")]
//    public float mass = 1305f;                // Mass of the vehicle (kg) - Audi TT 2.0 TFSI with driver
//    public float height = 1.353f;             //height of the car (meters)
//    public float Iz = 2500f;                  // Moment of inertia around Z axis (approximated)
//    public float wheelbase = 2.505f;          // Wheelbase in meters from specs
//    public float lf = 1.2525f;                // Distance from CG to front axle (approximated as wheelbase/2)
//    public float lr = 1.2525f;                // Distance from CG to rear axle (approximated as wheelbase/2)
//    public float trackWidth = 1.572f;         // Front track width in meters
//    public float Caf = 11000f;                // Front tire cornering stiffness (approximated)
//    public float Car = 13000f;                // Rear tire cornering stiffness (approximated)

//    [Header("Audi TT Environmental Parameters")]
//    public float gravity = 9.81f;             // Gravity acceleration
//    public float roadSlope = 0f;              // Road slope angle (theta) in degrees
//    public float dragCoefficient = 0.30f;     // From Audi TT specs
//    public float frontalArea = 2.09f;         // Frontal area in m² from specs

//    [Header("Powertrain Parameters")]
//    private const float maxTorque = 370f;      // Nm
//    private const float maxPower = 169f;  // kW 
//    const float idleRPM = 800f;          // Idle RPM
//    const float torqueStartRPM = 1600f;  // Start of max torque plateau
//    const float torqueEndRPM = 4300f;    // End of max torque plateau
//    const float powerStartRPM = 4500f;   // Start of max power
//    const float powerEndRPM = 6200f;     // End of max power
//    const float redlineRPM = 6500f;      // Redline RPM
//    const float maxPowerKW = 230f * 0.735f;  // Max power in kW (converted from PS)
//    float angularVelocity;
//    private float[] gearRatios = new float[] { // Gear ratios
//    2.923f,  // 1st
//    1.792f,  // 2nd
//    1.185f,  // 3rd
//    0.829f,  // 4th
//    0.862f,  // 5th
//    0.686f   // 6th
//};

//    private float[] finalDriveRatios = new float[] { // Final drive ratios from diiferential
//    4.769f,  // 1st-4th
//    4.769f,  // 2nd
//    4.769f,  // 3rd
//    4.769f,  // 4th
//    3.444f,  // 5th
//    3.444f   // 6th
//};
//    // Transmission state variables
//    private bool isShifting = false;
//    private float shiftTimer = 0f;
//    private float shiftDuration = 0.2f;  // S-tronic shift time

//    // RPM thresholds for gear shifts
//    private float normalUpshiftRPM = 4000f;
//    private float aggressiveUpshiftRPM = 6000f;
//    private float normalDownshiftRPM = 1800f;
//    private float aggressiveDownshiftRPM = 2500f;

//    // Kickdown threshold
//    private float kickdownThreshold = 0.8f;

//    [Header("Brake System Parameters")]
//    public float frontBrakeDiscDiameter = 0.312f;  // 312mm front brake discs
//    public float rearBrakeDiscDiameter = 0.300f;   // 300mm rear brake discs
//    public float brakeDiscFrictionCoefficient = 0.4f;  // Brake pad friction coefficient
//    public float brakeBalance = 0.7f;              // Front/rear brake force distribution
//    public float tireFrictionCoefficient = 0.8f;   // Tire-road friction coefficient (dry asphalt)


//    [Header("Wheel Parameters")]
//    public float wheelRadius = 0.3175f;            // For 225/50 R17 tires
//    private float wheelAngularVelocity = 0f;
//    private float wheelInertia = 1.5f;             // Approximated wheel rotational inertia
//    public Transform frontLeftWheel;
//    public Transform frontRightWheel;


//    [Header("State Variables")]
//    private float currentPoint =0; // Current point on the path
//    private float y = 0f;                          // Lateral position
//    private float yDot = 0f;                       // Lateral velocity
//    private float psi = 0f;                        // Yaw angle
//    private float psiDot = 0f;                     // Yaw rate
//    private float vx = 0;                         // Longitudinal velocity
//    public float ax = 0;                         // Longitudinal acceleration
//    private Vector4 state;                    // State vector [y, yDot, psi, psiDot]

//    private Matrix4x4 A;                      // State matrix
//    private Vector4 B;                        // Input matrix
//    private Vector3 position;                 // Global position
//    private int currentGear = 1;
//    private float engineRpm = 0f;

//    // Internal variables
//    private Rigidbody rb;
//    void Start()
//    {
//        rb = GetComponent<Rigidbody>();
//        rb.isKinematic = true;  // We'll handle the physics ourselves
//        position = transform.position;
//        InitializeStateSpace();
//        vx = 0f; // Starting from rest
//        ax = 0f; // Starting from rest
//        connection = GetComponent<LogitechG29Connection>();
//        controller = GetComponent<LogitechG29Controller>();
//        Debug.Log(currentPoint);
//        Debug.Log(position);
//        Debug.Log("Velocity: " + vx);
//        Debug.Log("Vy: " + yDot);
//        Debug.Log("ax: " + ax);
//        currentPoint++;
//    }

//    void InitializeStateSpace()
//    {
//        // Initialize state-space matrices based on Image 1
//        A = new Matrix4x4();

//        // First row
//        A.m00 = 0f; A.m01 = 1f; A.m02 = 0f; A.m03 = 0f;

//        // Second row
//        A.m10 = 0f;
//        A.m11 = -(2f * Caf + 2f * Car) / (mass * vx);
//        A.m12 = 0f;
//        A.m13 = -vx - (2f * Caf * lf - 2f * Car * lr) / (mass * vx);

//        // Third row
//        A.m20 = 0f; A.m21 = 0f; A.m22 = 0f; A.m23 = 1f;

//        // Fourth row
//        A.m30 = 0f;
//        A.m31 = -(2f * lf * Caf - 2f * lr * Car) / (Iz * vx);
//        A.m32 = 0f;
//        A.m33 = -(2f * lf * lf * Caf + 2f * lr * lr * Car) / (Iz * vx);

//        B = new Vector4(
//            0f,
//            2f * Caf / mass,
//            0f,
//            2f * lf * Caf / Iz
//        );
//    }
//    //private void HandleTransmission(float throttle, float deltaTime)
//    //{
//    //    // 1. מטפל בזמן החלפת הילוך
//    //    if (isShifting)
//    //    {
//    //        shiftTimer += deltaTime;
//    //        if (shiftTimer >= shiftDuration)  // shiftDuration = 0.2f עבור S-tronic
//    //        {
//    //            isShifting = false;
//    //            shiftTimer = 0f;
//    //        }
//    //        return;
//    //    }

//    //    // 2. חישוב נקודות החלפת הילוכים בהתאם למצערת
//    //    float upshiftRPM = Mathf.Lerp(normalUpshiftRPM, aggressiveUpshiftRPM, throttle);
//    //    float downshiftRPM = Mathf.Lerp(normalDownshiftRPM, aggressiveDownshiftRPM, throttle);

//    //    // 3. קיקדאון - הורדת הילוך במצערת מלאה
//    //    if (throttle > kickdownThreshold && currentGear > 1)
//    //    {
//    //        // מחשב מה יהיה הסל"ד בהילוך הנמוך יותר לפי יחסי S-tronic
//    //        float rpmAfterDownshift = engineRpm * gearRatios[currentGear - 2] / gearRatios[currentGear - 1];
//    //        // לדוגמה: אם אנחנו בהילוך 3 (1.185) ורוצים לרדת ל-2 (1.792)
//    //        // rpmAfterDownshift = engineRpm * 1.792 / 1.185

//    //        if (rpmAfterDownshift < redlineRPM)  // בדיקה שלא נחרוג מהסל"ד המקסימלי
//    //        {
//    //            ChangeGear(currentGear - 1);
//    //            return;
//    //        }
//    //    }

//    //    // 4. העלאת הילוך רגילה
//    //    if (engineRpm > upshiftRPM && currentGear < gearRatios.Length)
//    //    {
//    //        // מחשב מה יהיה הסל"ד בהילוך הבא
//    //        float nextGearRPM = engineRpm * gearRatios[currentGear] / gearRatios[currentGear - 1];
//    //        // לדוגמה: אם אנחנו בהילוך 2 (1.792) ורוצים לעלות ל-3 (1.185)
//    //        // nextGearRPM = engineRpm * 1.185 / 1.792

//    //        if (nextGearRPM > downshiftRPM)
//    //        {
//    //            ChangeGear(currentGear + 1);
//    //        }
//    //    }
//    //    // 5. הורדת הילוך רגילה
//    //    else if (engineRpm < downshiftRPM && currentGear > 1)
//    //    {
//    //        float prevGearRPM = engineRpm * gearRatios[currentGear - 2] / gearRatios[currentGear - 1];
//    //        if (prevGearRPM < redlineRPM)
//    //        {
//    //            ChangeGear(currentGear - 1);
//    //        }
//    //    }

//    //    // 6. הגנה מפני סל"ד נמוך מדי
//    //    if (engineRpm < idleRPM * 1.2f && currentGear > 1)
//    //    {
//    //        ChangeGear(currentGear - 1);
//    //    }
//    //}
//    //private void ChangeGear(int newGear)
//    //{
//    //    if (newGear < 1 || newGear > gearRatios.Length || isShifting) return;

//    //    currentGear = newGear;
//    //    isShifting = true;
//    //    shiftTimer = 0f;
//    //}
//    private float CalculateEngineTorque(float rpm)
//    {
//        // Below idle or above redline
//        if (rpm < idleRPM || rpm > redlineRPM)
//            return 0f;

//        // Idle to peak torque (linear increase)
//        if (rpm < torqueStartRPM)
//            return Mathf.Lerp(maxTorque * 0.3f, maxTorque, (rpm - idleRPM) / (torqueStartRPM - idleRPM));

//        // Peak torque plateau
//        if (rpm <= torqueEndRPM)
//            return maxTorque;

//        // Torque decrease to power peak
//        if (rpm <= powerStartRPM)
//            return Mathf.Lerp(maxTorque, maxTorque * 0.9f, (rpm - torqueEndRPM) / (powerStartRPM - torqueEndRPM));

//        // Power peak region (constant power)
//        if (rpm <= powerEndRPM)
//        {
//            angularVelocity = (2f * Mathf.PI * rpm) / 60f;
//            return (maxPowerKW * 1000f) / angularVelocity;
//        }

//        // Power drop to redline
//        float powerDrop = Mathf.Lerp(maxPowerKW, maxPowerKW * 0.85f, (rpm - powerEndRPM) / (redlineRPM - powerEndRPM));
//        angularVelocity = (2f * Mathf.PI * rpm) / 60f;
//        return (powerDrop * 1000f) / angularVelocity;
//    }
//    private float CalculateEngineForce(float throttle)
//    {
//        if (Mathf.Abs(throttle) < 0.001f) return 0f;

//        // Calculate engine RPM
//        float wheelRpm = (Mathf.Abs(vx) * 60f) / (2f * Mathf.PI * wheelRadius);
//        float currentFinalDrive = finalDriveRatios.Average();//finalDriveRatios[currentGear];
//        engineRpm = Mathf.Max(wheelRpm * gearRatios.Average() * currentFinalDrive, 800f);
//        //engineRpm = Mathf.Max(wheelRpm * gearRatios[currentGear - 1] * currentFinalDrive, 800f);

//        // Get torque and apply transmission
//        float engineTorque = CalculateEngineTorque(engineRpm);
//        float torqueAtWheels = engineTorque * gearRatios[currentGear - 1] * currentFinalDrive;

//        return (torqueAtWheels / wheelRadius) * throttle;
//    }

//    private float CalculateBrakeForce(float brakeInput)
//    {
//        // Calculate maximum possible brake force based on tire grip
//        float staticWeight = mass * gravity;
//        float maxBrakeForce = staticWeight * tireFrictionCoefficient; // Maximum theoretical brake force

//        // Calculate weight transfer during braking
//        float deceleration = brakeInput * gravity * 1.25f;  // Maximum deceleration around 1.25*g
//        float weightTransfer = (mass * deceleration * height * 0.5f) / wheelbase;

//        // Calculate normal forces with weight transfer
//        float frontNormalForce = (staticWeight * lr / wheelbase) + weightTransfer;
//        float rearNormalForce = (staticWeight * lf / wheelbase) - weightTransfer;

//        // Calculate maximum brake forces per axle based on normal force and friction
//        float maxFrontBrakeForce = frontNormalForce * tireFrictionCoefficient;
//        float maxRearBrakeForce = rearNormalForce * tireFrictionCoefficient;

//        // Apply brake balance and pedal input
//        float actualFrontBrakeForce = maxFrontBrakeForce * brakeBalance * brakeInput;
//        float actualRearBrakeForce = maxRearBrakeForce * (1 - brakeBalance) * brakeInput;

//        // Total brake force
//        return actualFrontBrakeForce + actualRearBrakeForce;
//    }
//    private SteeringAngles CalculateAckermanSteering(float steerInput)
//    {
//        SteeringAngles angles = new SteeringAngles();

//        // Convert input (-1 to 1) to steering wheel angle in degrees
//        angles.steeringWheelAngle = steerInput * maxSteeringWheelAngle;

//        // Convert steering wheel angle to wheel angle in degrees
//        float wheelAngleDegrees = angles.steeringWheelAngle / steeringRatio;

//        // Clamp wheel angle to maximum (still in degrees)
//        wheelAngleDegrees = Mathf.Clamp(wheelAngleDegrees, -maxWheelAngle, maxWheelAngle);

//        // Convert to radians for further calculations
//        float baseWheelAngle = wheelAngleDegrees * Mathf.Deg2Rad;

//        // Calculate turn radius based on the base wheel angle
//        float turnRadius = wheelbase / Mathf.Tan(Mathf.Abs(baseWheelAngle));

//        // Calculate Ackerman angles based on turn direction
//        if (baseWheelAngle > 0)
//        {
//            // Left turn
//            angles.inner = Mathf.Atan(wheelbase / (turnRadius - (trackWidth / 2f)));
//            angles.outer = Mathf.Atan(wheelbase / (turnRadius + (trackWidth / 2f)));
//        }
//        else
//        {
//            // Right turn
//            angles.inner = -Mathf.Atan(wheelbase / (turnRadius - (trackWidth / 2f)));
//            angles.outer = -Mathf.Atan(wheelbase / (turnRadius + (trackWidth / 2f)));
//        }

//        // Calculate average angle for bicycle model
//        angles.average = (angles.inner + angles.outer) / 2f;

//        return angles;
//    }
//    void FixedUpdate()
//    {
//        float dt = Time.fixedDeltaTime;
//        float Rxf = 0, Rxr = 0;
//        steeringInput = 0;
//        throttleInput = 0;
//        brakeInput = 0;
//        // Get inputs
//        if (connection.IsWheelConnected())
//        {
//            throttleInput = controller.GetThrottle();
//            steeringInput = controller.GetSteering();
//            brakeInput = controller.GetBrake();
//        }
//        else
//        {
//            // Get input (you can modify this based on your input system)
//            throttleInput = Mathf.Clamp(Input.GetAxis("Vertical"), 0, 1);
//            steeringInput = Input.GetAxis("Horizontal");
//            brakeInput = Input.GetAxis("Jump");
//        }
//        //float steeringInput = Input.GetAxis("Horizontal");
//        //float throttleInput = Input.GetAxis("Vertical");
//        //float brakeInput = Input.GetAxis("Jump");
//        // Gear changes
//        //if (Input.GetKeyDown(KeyCode.E) && currentGear < 6) currentGear++;
//        //if (Input.GetKeyDown(KeyCode.Q) && currentGear > 1) currentGear--;

//        // Handle automatic transmission
//        //HandleTransmission(throttleInput, dt);

//        // Calculate longitudinal dynamics
//        float airDensity = 1.225f; // kg/m³ at sea level
//        float Faero = 0.5f * airDensity * dragCoefficient * frontalArea * vx * vx;
//        float Fxf = CalculateEngineForce(throttleInput);
//        float Fbrake = CalculateBrakeForce(brakeInput);
//        SteeringAngles angles = CalculateAckermanSteering(steeringInput);
//        //Debug.Log("Fxf: " + Fxf);
//        //Debug.Log("Fbrake: " + Fbrake);

//        if (vx > 0)// Calculate rolling resistance
//        {
//            Rxf = 0.5f * mass * gravity * 0.015f * 10; // Rolling resistance coefficient ≈ 0.015
//            Rxr = Rxf;
//        }
//        // Calculate slope resistance
//        float slopeRadians = roadSlope * Mathf.Deg2Rad;

//        // Longitudinal acceleration (from Longitudinal dynamic equation)
//        ax = (Fxf - Fbrake - Faero - Rxf - Rxr - mass * gravity * Mathf.Sin(slopeRadians)) / mass;
//        vx += ax * dt;
//        // Clamp vx to be between 0 and maxVelocity
//        vx = Mathf.Clamp(vx, 0, maxVelocity / 3.6f);

//        // Update state-space matrices if velocity changed significantly
//        if (Mathf.Abs(vx) > 0.001f)
//        {
//            InitializeStateSpace();
//            // State-space update
//            Vector4 stateDot = A * state + B * angles.average;
//            state += stateDot * dt;

//            // Extract states
//            y = state[0];
//            yDot = state[1];
//            psi = state[2];
//            psiDot = state[3];
//        }

//        // Update position
//        float cosRot = Mathf.Cos(psi);
//        float sinRot = Mathf.Sin(psi);
//        position.z += (vx * cosRot - yDot * sinRot) * dt;
//        position.x += (vx * sinRot + yDot * cosRot) * dt;

//        Debug.Log(currentPoint);
//        Debug.Log(position);
//        Debug.Log("Velocity: " + vx);
//        Debug.Log("Vy: " + yDot);
//        Debug.Log("ax: " + ax);
//        currentPoint++;


//        // Apply to transform
//        transform.position = position;
//        transform.rotation = Quaternion.Euler(0f, psi * Mathf.Rad2Deg, 0f);

//        // Rotate front wheels based on Ackerman steering angles
//        frontLeftWheel.localRotation = Quaternion.Euler(0f, angles.inner * Mathf.Rad2Deg, 0f);
//        frontRightWheel.localRotation = Quaternion.Euler(0f, angles.outer * Mathf.Rad2Deg, 0f);
//    }


//    void OnGUI()
//    {
//        // Debug information

//        GUI.Label(new Rect(10, 10, 200, 20), $"Speed: {vx * 3.6f:F1} km/h");
//        GUI.Label(new Rect(10, 30, 200, 20), $"Engine RPM: {engineRpm:F0}");
//        //GUI.Label(new Rect(10, 50, 200, 20), $"Current Gear: {currentGear}");
//        GUI.Label(new Rect(10, 70, 200, 20), $"Lateral Velocity: {yDot:F2} m/s");
//        GUI.Label(new Rect(10, 90, 200, 20), $"Yaw Rate: {psiDot * Mathf.Rad2Deg:F1} deg/s");
//        GUI.Label(new Rect(10, 110, 200, 20), $"throttleInput: {throttleInput} ");
//        GUI.Label(new Rect(10, 130, 200, 20), $"brakeInput: {brakeInput}");
//        GUI.Label(new Rect(10, 150, 200, 20), $"steeringInput: {steeringInput}");
//    }
//}