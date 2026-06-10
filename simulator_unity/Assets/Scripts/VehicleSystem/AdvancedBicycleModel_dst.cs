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
    private float y = 0f;
    private float yDot = 0f;
    private float _x0 = 0f;   // world-X at spawn; y = _x0 - position.x
    private float psi = 0f;
    private float psiDot = 0f;
    private float vx = 0;
    private float ax = 0;
    private Vector4 state;
    private float Fx_tot = 0f;
    private SteeringAngles angles = new SteeringAngles { inner = 0, outer = 0, average = 0, steeringWheelAngle = 0 };
    private VehicleParameters vehicleParams;
    public Powertrain powertrain { get; private set; }
    private BrakeSystem brakeSystem;
    private SteeringSystem steeringSystem;
    private Transmission transmission;
    private Rigidbody rb;
    private Matrix4x4 A;
    private Vector4 B;
    private Vector3 position;

    // Wheel transforms for steering visualization
    public Transform frontLeftWheel;
    public Transform frontRightWheel;
    public Transform backLeftWheel;
    public Transform backRightWheel;
    public Transform SteeringWheel;

    public bool AutoMode = false;
    public int IndexPlatoon = -1;

    [HideInInspector] public bool  NashModeActive;
    [HideInInspector] public float NashThrottle;
    [HideInInspector] public float NashBrake;
    [HideInInspector] public float NashSteerNorm;
    private float _prevDelta = 0f;
    private float Totaldistance = 0f;

    void Awake()
    {
        // Initialize early so PlatoonManager.Start() can call GetVehicleParameters/GetCoastDeceleration
        vehicleParams = new VehicleParameters();
        EngineAudio engineAudio = GetComponent<EngineAudio>();
        powertrain = new Powertrain(vehicleParams, engineAudio);
        brakeSystem = new BrakeSystem(vehicleParams);
        steeringSystem = new SteeringSystem(vehicleParams);
    }

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.isKinematic = true;
        position = transform.position;
        _x0 = position.x;

        if (GetComponent<EngineAudio>() == null)
                    Debug.LogWarning("EngineAudio component not found!");
        
        steeringInput = 0;
        throttleInput = 0;
        brakeInput = 0;

        // powertrain already created in Awake — just get transmission reference
        transmission = powertrain.GetTransmission();

                steeringSystem.frontLeftWheel = frontLeftWheel;
        steeringSystem.frontRightWheel = frontRightWheel;
        steeringSystem.SteeringWheel = SteeringWheel;

        InitializeStateSpace();
        vx = 0f;
        ax = 0f;

        powertrain.StartEngine();

                if (GetComponent<VehicleDashboard>() == null)
        {
            gameObject.AddComponent<VehicleDashboard>();
            Debug.Log("VehicleDashboard component added automatically.");
        }

                if (FindFirstObjectByType<DisplayManager>() == null)
        {
            GameObject displayManagerGO = new GameObject("DisplayManager");
            displayManagerGO.AddComponent<DisplayManager>();
            Debug.Log("DisplayManager created automatically.");
        }
    }

    void InitializeStateSpace()
    {
                A = new Matrix4x4();

                A.m00 = 0f; A.m01 = 1f; A.m02 = 0f; A.m03 = 0f;

                A.m10 = 0f;
        A.m11 = -(2f * vehicleParams.Caf + 2f * vehicleParams.Car) / (vehicleParams.mass * vx);
        A.m12 = 0f;
        A.m13 = -vx - (2f * vehicleParams.Caf * vehicleParams.lf - 2f * vehicleParams.Car * vehicleParams.lr) / (vehicleParams.mass * vx);

                A.m20 = 0f; A.m21 = 0f; A.m22 = 0f; A.m23 = 1f;

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

        // AutoMode: inputs come from PlatoonManager (via NashThrottle/NashBrake)
        // Manual mode: inputs come from the player or NashCoordinator
        if (AutoMode) {
            throttleInput = NashThrottle;
            steeringInput = NashSteerNorm;
            brakeInput    = NashBrake;
        } else {
                if (NashModeActive) {
                    throttleInput = NashThrottle;
                    steeringInput = NashSteerNorm;
                    brakeInput    = NashBrake;
                } else {
                    throttleInput = VehicleInputs.Instance.ThrottleInput;
                    steeringInput = VehicleInputs.Instance.SteeringInput;
                    brakeInput    = VehicleInputs.Instance.BrakeInput;
                }
            }

        float airDensity = 1.225f;
        float Faero = 0.5f * airDensity * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;

        float Fxf_engine = powertrain.CalculateEngineForce(throttleInput, vx);
        var (FbrakeFront, FbrakeRear) = brakeSystem.CalculateBrakeForceSplit(brakeInput, vx);

        angles = steeringSystem.CalculateAckermanSteering(steeringInput);

        if (vx > 0.01f)
        {
            float rollingResistanceCoeff = 0.012f * (1.0f + vx * 0.008f);
            Rxf = 0.5f * vehicleParams.mass * vehicleParams.gravity * rollingResistanceCoeff;
            Rxr = Rxf;
        }

        float slopeRadians = vehicleParams.roadSlope * Mathf.Deg2Rad;
        float slopeForce = vehicleParams.mass * vehicleParams.gravity * Mathf.Sin(slopeRadians);

        float naturalDeceleration = 0f;
        if (throttleInput < 0.001f && brakeInput < 0.001f && vx > 0.1f)
        {
            naturalDeceleration = powertrain.CalculateNaturalDeceleration(vx, powertrain.EngineRpm);
            naturalDeceleration += vx * vx * 0.01f;
        }

        // Fxf_contact / Fxr_contact: forces at each axle contact patch (Belousov Eq. 3.11b/c).
        // Audi TT is FWD: engine force acts only at front axle.
        // Body-level resistances (Faero, Rr, Rg) are NOT contact-patch forces — they enter
        // only Eq. 3.11a via Fx_tot, not the sinδ terms of 3.11b/c.
        float Fxf_contact = Fxf_engine - FbrakeFront;
        float Fxr_contact = -FbrakeRear;

        Fx_tot = Fxf_contact + Fxr_contact - Faero - Rxf - Rxr - slopeForce - naturalDeceleration;

        if (AutoMode) {
            float engineForce = Fx_tot + Faero + Rxf + Rxr + slopeForce + naturalDeceleration;
            powertrain.UpdateRpmForAutonomousMode(engineForce, vx);
        }

        UpdateDynamics6D(Fx_tot, Fxf_contact, angles.average, dt);

        Totaldistance += math.sqrt(math.pow(vx * dt, 2) + math.pow(yDot * dt, 2));

                transform.position = position;
        transform.rotation = Quaternion.Euler(0f, psi * Mathf.Rad2Deg, 0f);

                steeringSystem.UpdateWheelRotation(angles);
        UpdateAngleWheels();
    }

    // Public getters
    public Vector3 GetPosition() => position;
    public float GetVx() => vx;
    public float SetVx(float newVx) => vx = newVx;
    public float GetAx() => ax;

    // Total passive deceleration at given speed when no pedal pressed [m/s²], always negative
    public float GetCoastDeceleration(float vxS) {
        float Faero   = 0.5f * 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vxS * vxS;
        float Cr      = 0.012f * (1f + vxS * 0.008f);
        float Frr     = vehicleParams.mass * vehicleParams.gravity * Cr;
        float Fengbrk = powertrain.GetEngineBrakingForce(vxS);
        return -(Faero + Frr + Fengbrk) / vehicleParams.mass;
    }

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

        public Transmission GetTransmission() => transmission;
    public VehicleParameters GetVehicleParameters() => vehicleParams;
    public BrakeSystem GetBrakeSystem() => brakeSystem;
    public SteeringSystem GetSteeringSystem() => steeringSystem;
    public SteeringAngles GetCurrentSteeringAngles() => angles;
    public int GetCurrentGear() => transmission.GetCurrentGear();
    public float GetTotalDistance() => Totaldistance;

        public float GetThrottleInput() => throttleInput;
    public float GetBrakeInput() => brakeInput;
    public float GetSteeringInput() => steeringInput;
    public float GetSteeringWheelAngle() => angles.steeringWheelAngle;

    // ── Belousov RK4 6D dynamics ──────────────────────────────────────────
    // Fx_tot     : net longitudinal force on body [N] — enters Eq. 3.11a
    // Fxf_contact: front axle contact-patch force [N] — enters Eq. 3.11b/c sinδ terms
    void UpdateDynamics6D(float Fx_tot, float Fxf_contact, float delta, float dt) {
        float deltaRL = RateLimitDelta(delta, dt);
        float[] s    = GetState6DArray();
        float[] sNew = RK4Step(s, Fx_tot, Fxf_contact, deltaRL, dt);

                sNew[1] = Mathf.Clamp(sNew[1], 0f, vehicleParams.maxVelocity / 3.6f);
        sNew[3] = Mathf.Clamp(sNew[3], -2f, 2f);
        sNew[4] = Mathf.Clamp(sNew[4], -0.5236f, 0.5236f);
        sNew[5] = Mathf.Clamp(sNew[5], -0.3f, 0.3f);

        if (float.IsNaN(sNew[0]) || float.IsNaN(sNew[1]) || float.IsNaN(sNew[2]) ||
            float.IsNaN(sNew[3]) || float.IsNaN(sNew[4]) || float.IsNaN(sNew[5])) {
            Debug.LogWarning("[Belousov] NaN in 6D state — resetting lateral velocities");
            sNew[3] = 0f; sNew[5] = 0f;
        }

                position.z = sNew[0];
        ax         = (sNew[1] - s[1]) / dt;
        vx         = sNew[1];
        position.x = sNew[2];
        psi        = sNew[4];
        psiDot     = sNew[5];
        float xDotNew = vx * Mathf.Sin(psi) + sNew[3] * Mathf.Cos(psi);
        yDot = -xDotNew;           // ẏ = -d(position.x)/dt
        y    = _x0 - position.x;  // exact world-frame lateral tracking
    }

    float[] RK4Step(float[] s, float Fx_tot, float Fxf_contact, float delta, float dt) {
        float[] k1 = Derivatives6D(s, Fx_tot, Fxf_contact, delta);
        float[] k2 = Derivatives6D(Add6(s, Scale6(k1, dt / 2f)), Fx_tot, Fxf_contact, delta);
        float[] k3 = Derivatives6D(Add6(s, Scale6(k2, dt / 2f)), Fx_tot, Fxf_contact, delta);
        float[] k4 = Derivatives6D(Add6(s, Scale6(k3, dt)),       Fx_tot, Fxf_contact, delta);
        float[] wsum = Add6(Add6(k1, Scale6(k2, 2f)), Add6(Scale6(k3, 2f), k4));
        return Add6(s, Scale6(wsum, dt / 6f));
    }

    // Belousov Eq. 3.11a/b/c:
    //   Fx_tot      → 3.11a only (net body force incl. drag, rolling, grade)
    //   Fxf_contact → 3.11b/c sinδ terms (front axle contact-patch force only)
    float[] Derivatives6D(float[] s, float Fx_tot, float Fxf_contact, float delta) {
        float vxS    = s[1], vyS = s[3], psiS = s[4], psiDotS = s[5];
        float vxSafe = Mathf.Max(Mathf.Abs(vxS), 1.0f);

        float fv = ((float)System.Math.Tanh(10.0 * Mathf.Abs(vxS) - 8.0) + 1f) / 2f;

        float alphaF = fv * (delta - Mathf.Atan2(vyS + vehicleParams.lf * psiDotS, vxSafe));
        float alphaR = fv * Mathf.Atan2(vehicleParams.lr * psiDotS - vyS, vxSafe);

        float Fyf = vehicleParams.Caf * alphaF;
        float Fyr = vehicleParams.Car * alphaR;
        float m   = vehicleParams.mass;

        float cosD = Mathf.Cos(delta), sinD = Mathf.Sin(delta);
        float cosP = Mathf.Cos(psiS),  sinP = Mathf.Sin(psiS);

        float zDot = vxS * cosP - vyS * sinP;
        float xDot = vxS * sinP + vyS * cosP;

        // Eq. 3.11a: full net body force drives longitudinal acceleration
        float vxDot = vyS * psiDotS + (Fx_tot * cosD - Fyf * sinD) / m;

        float k_vy  = vehicleParams.tireFrictionCoefficient * vehicleParams.gravity / 0.5f;
        float k_psi = vehicleParams.tireFrictionCoefficient * vehicleParams.mass
                    * vehicleParams.gravity * vehicleParams.wheelbase
                    / (4f * vehicleParams.Iz * 0.3f);

        // Eq. 3.11b: only front contact-patch force contributes via sinδ
        float vyDot = (Fyf * cosD + Fyr + Fxf_contact * sinD) / m - vxS * psiDotS
                    - (1f - fv) * k_vy * vyS;

        // Eq. 3.11c: only front contact-patch force generates yaw moment via sinδ
        float psiDotDot = (vehicleParams.lf * Fyf * cosD
                         - vehicleParams.lr * Fyr
                         + vehicleParams.lf * Fxf_contact * sinD) / vehicleParams.Iz
                         - (1f - fv) * k_psi * psiDotS;

        return new[] { zDot, vxDot, xDot, vyDot, psiDotS, psiDotDot };
    }

    float RateLimitDelta(float delta, float dt) {
        float maxChange = SimCfg.I != null ? SimCfg.I.Timing.MaxSteerRate : 0.087f;
        float limited   = Mathf.Clamp(delta, _prevDelta - maxChange, _prevDelta + maxChange);
        _prevDelta      = limited;
        return limited;
    }

        float[] GetState6DArray() => new[] { position.z, vx, position.x, yDot, psi, psiDot };

    static float[] Add6(float[] a, float[] b) =>
        new[] { a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5] };

    static float[] Scale6(float[] v, float sc) =>
        new[] { v[0]*sc, v[1]*sc, v[2]*sc, v[3]*sc, v[4]*sc, v[5]*sc };

    // ── Nash linearization methods ────────────────────────────────────────
    // Linearised bicycle model — Belousov Eq. 3.11b/c — for state [y, ẏ, ψ, ψ̇].
    // state[1] = ẏ (world-frame dy/dt); slip angles via vy_Belousov ≈ −ẏ (small-ψ, Unity sign).
    public (float[,] Ac_lat, float[] Bc_lat) GetLatJacobian() {
        float vxS = Mathf.Max(Mathf.Abs(vx), 0.5f);
        float m   = vehicleParams.mass, Iz = vehicleParams.Iz;
        float lf  = vehicleParams.lf,   lr = vehicleParams.lr;
        float Caf = vehicleParams.Caf,  Car = vehicleParams.Car;
        float fv  = ((float)System.Math.Tanh(10.0 * vxS - 8.0) + 1f) / 2f;

        float[,] Ac = {
            { 0,  1,  0,  0 },
            { 0, -fv*(2f*Caf + 2f*Car)/(m*vxS),       0,  fv*(2f*Caf*lf - 2f*Car*lr)/(m*vxS) },
            { 0,  0,  0,  1 },
            { 0,  fv*(2f*lf*Caf - 2f*lr*Car)/(Iz*vxS), 0, -fv*(2f*lf*lf*Caf + 2f*lr*lr*Car)/(Iz*vxS) }
        };
        float[] Bc = { 0f, -fv*2f*Caf/m, 0f, fv*2f*lf*Caf/Iz };
        return (Ac, Bc);
    }

    // Linearised longitudinal dynamics — Belousov Eq. 3.11a, state [z, vx].
    // v̇x ≈ vy·ψ̇ + (Fxf·cosδ − Fyf·sinδ)/m  linearised around straight (δ≈0, ψ̇≈0).
    // Remaining nonlinear term: aerodynamic drag  ∂v̇x/∂vx = −(ρ·Cd·A·|vx|)/m.
    // Control input u = a_des [m/s²]: AccelerationToInputs performs feedforward
    // cancellation of drag+rolling+grade, so from the Nash solver's perspective
    // v̇x = −dragLin·vx + 1·u  →  Bc = [0, 1].
    public (float[,] Ac_long, float[] Bc_long) GetLongJacobian() {
        float dragLin = 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea
                      * Mathf.Abs(vx) / vehicleParams.mass;
        float[,] Ac = { { 0, 1 }, { 0, -dragLin } };
        float[]  Bc = { 0f, 1f };   // u = a_des [m/s²], not force [N]
        return (Ac, Bc);
    }

    public float[] GetLongState()      => new[] { position.z, vx };
    public float[] GetLatStateVector() => new[] { y, yDot, psi, psiDot };
    public float   GetX()              => position.z;
    public float   GetLength()         => vehicleParams.length;

    // (throttle, brake) ∈ [0,1] → net acceleration [m/s²] at current vx.
    // Forward model of FixedUpdate force calculation (ignores road slope).
    public float InputsToAcceleration(float thr, float brk)
    {
        float Faero  = 0.5f * 1.225f * vehicleParams.dragCoefficient
                     * vehicleParams.frontalArea * vx * vx;
        float Fxf    = powertrain.CalculateEngineForce(thr, vx);
        float Fbrake = brakeSystem.CalculateBrakeForce(brk, vx);
        float Cr     = 0.012f * (1f + vx * 0.008f);
        float Rxf    = 0.5f * vehicleParams.mass * vehicleParams.gravity * Cr;

        float naturalDecel = 0f;
        if (thr < 0.001f && brk < 0.001f && vx > 0.1f)
        {
            naturalDecel  = powertrain.CalculateNaturalDeceleration(vx, powertrain.EngineRpm);
            naturalDecel += vx * vx * 0.01f;
        }

        float Fx_tot = Fxf - Fbrake - Faero - 2f * Rxf - naturalDecel;
        return Fx_tot / vehicleParams.mass;
    }

    // a_des [m/s²] → (throttle, brake) ∈ [0,1]
    // Three zones: throttle | coast (engine braking sufficient) | brake
    public (float throttle, float brake) AccelerationToInputs(float a_des) {
        float Faero = 0.5f * 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;
        float Cr    = 0.012f * (1f + vx * 0.008f);
        float Frr   = vehicleParams.mass * vehicleParams.gravity * Cr;

        float F_net = a_des * vehicleParams.mass + Faero + Frr;
        float F_eng = -GetCoastDeceleration(vx) * vehicleParams.mass - Faero - Frr;

        if (F_net >= 0f) {
            float maxF = powertrain.CalculateMaxForceAtCurrentSpeed(vx);
            return (Mathf.Clamp01(maxF > 1f ? F_net / maxF : 0f), 0f);
        } else if (F_net >= -F_eng)
            return (0f, 0f);
        else {
            float maxBrakeF = brakeSystem.CalculateBrakeForce(1.0f, vx);
            return (0f, Mathf.Clamp01(maxBrakeF > 1f ? (-F_net - F_eng) / maxBrakeF : 0f));
        }
    }
}