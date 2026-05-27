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

        float Fxf = powertrain.CalculateEngineForce(throttleInput, vx);
        float Fbrake = brakeSystem.CalculateBrakeForce(brakeInput, vx);

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

        Fx_tot = Fxf - Fbrake - Faero - Rxf - Rxr - slopeForce - naturalDeceleration;

        if (AutoMode) {
            float engineForce = Fx_tot + Faero + Rxf + Rxr + slopeForce + naturalDeceleration;
            powertrain.UpdateRpmForAutonomousMode(engineForce, vx);
        }

        UpdateDynamics6D(Fx_tot, angles.average, dt);

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
    void UpdateDynamics6D(float Fxf_net, float delta, float dt) {
        float deltaRL = RateLimitDelta(delta, dt);
        float[] s    = GetState6DArray();
        float[] sNew = RK4Step(s, Fxf_net, deltaRL, dt);

        sNew[1] = Mathf.Clamp(sNew[1], 0f, vehicleParams.maxVelocity / 3.6f);
        sNew[3] = Mathf.Clamp(sNew[3], -2f, 2f);
        sNew[4] = Mathf.Clamp(sNew[4], -0.2618f, 0.2618f);
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
        yDot       = sNew[3];
        psi        = sNew[4];
        psiDot     = sNew[5];
        y         += yDot * dt;
    }

    float[] RK4Step(float[] s, float Fxf, float delta, float dt) {
        float[] k1 = Derivatives6D(s, Fxf, delta);
        float[] k2 = Derivatives6D(Add6(s, Scale6(k1, dt / 2f)), Fxf, delta);
        float[] k3 = Derivatives6D(Add6(s, Scale6(k2, dt / 2f)), Fxf, delta);
        float[] k4 = Derivatives6D(Add6(s, Scale6(k3, dt)),       Fxf, delta);
        float[] wsum = Add6(Add6(k1, Scale6(k2, 2f)), Add6(Scale6(k3, 2f), k4));
        return Add6(s, Scale6(wsum, dt / 6f));
    }

    // Rajamani Eq. 13.45 — parabolic pressure distribution, per wheel.
    // Returns force in N for one wheel; caller multiplies by 2 for the axle.
    static float TireForcePerWheel(float alpha, float C, float mu, float Fz_wheel) {
        float theta    = C / (3f * mu * Fz_wheel);
        float S        = Mathf.Tan(alpha);
        float invTheta = 1f / theta;
        if (Mathf.Abs(S) <= invTheta)
            return mu * Fz_wheel * (3f*theta*S - 3f*theta*theta*S*S + theta*theta*theta*S*S*S);
        return Mathf.Sign(S) * mu * Fz_wheel;   // saturated at μ·Fz
    }

    // dF_per_wheel/dα — slope of the Rajamani parabola at the current slip angle.
    // Used to schedule the Jacobian: returns C at α=0, falls to 0 at saturation.
    static float TireSlopeAtAlpha(float alpha, float C, float mu, float Fz_wheel) {
        float theta = C / (3f * mu * Fz_wheel);
        float S     = Mathf.Tan(alpha);
        if (Mathf.Abs(S) > 1f / theta) return 0f;   // saturated — zero restoring stiffness
        float dFdS = mu * Fz_wheel * (3f*theta - 6f*theta*theta*S + 3f*theta*theta*theta*S*S);
        return dFdS * (1f + S * S);   // chain rule: d/dα = dF/dS · sec²(α)
    }

    float[] Derivatives6D(float[] s, float Fxf_net, float delta) {
        float vxS    = s[1], vyS = s[3], psiS = s[4], psiDotS = s[5];
        float vxSafe = Mathf.Max(Mathf.Abs(vxS), 1.0f);

        float fv = ((float)System.Math.Tanh(10.0 * Mathf.Abs(vxS) - 8.0) + 1f) / 2f;

        float alphaF = fv * (delta - Mathf.Atan2(vyS + vehicleParams.lf * psiDotS, vxSafe));
        float alphaR = fv * Mathf.Atan2(vehicleParams.lr * psiDotS - vyS, vxSafe);

        // Per-wheel normal loads (static, 50/50 CG split)
        float mu      = vehicleParams.tireFrictionCoefficient;
        float axleLoadF = vehicleParams.mass * vehicleParams.gravity
                        * vehicleParams.lr / (vehicleParams.lf + vehicleParams.lr);
        float axleLoadR = vehicleParams.mass * vehicleParams.gravity
                        * vehicleParams.lf / (vehicleParams.lf + vehicleParams.lr);
        float Fz_f_wheel = 0.5f * axleLoadF;
        float Fz_r_wheel = 0.5f * axleLoadR;

        float Fyf = 2f * TireForcePerWheel(alphaF, vehicleParams.Caf, mu, Fz_f_wheel);
        float Fyr = 2f * TireForcePerWheel(alphaR, vehicleParams.Car, mu, Fz_r_wheel);
        float m   = vehicleParams.mass;

        float cosD = Mathf.Cos(delta), sinD = Mathf.Sin(delta);
        float cosP = Mathf.Cos(psiS),  sinP = Mathf.Sin(psiS);

        float zDot = vxS * cosP - vyS * sinP;
        float xDot = vxS * sinP + vyS * cosP;

        float vxDot = vyS * psiDotS + (Fxf_net * cosD - Fyf * sinD) / m;

        float k_vy  = vehicleParams.tireFrictionCoefficient * vehicleParams.gravity / 0.5f;
        float k_psi = vehicleParams.tireFrictionCoefficient * vehicleParams.mass
                    * vehicleParams.gravity * vehicleParams.wheelbase
                    / (4f * vehicleParams.Iz * 0.3f);

        float vyDot = (Fyf * cosD + Fyr + Fxf_net * sinD) / m - vxS * psiDotS
                    - (1f - fv) * k_vy * vyS;

        float psiDotDot = (vehicleParams.lf * Fyf * cosD
                         - vehicleParams.lr * Fyr
                         + vehicleParams.lf * Fxf_net * sinD) / vehicleParams.Iz
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
    public (float[,] Ac_lat, float[] Bc_lat) GetLatJacobian() {
        float vxS = Mathf.Max(Mathf.Abs(vx), 0.5f);
        float m   = vehicleParams.mass, Iz = vehicleParams.Iz;
        float lf  = vehicleParams.lf,   lr = vehicleParams.lr;

        // ── Normal loads per wheel (static 50/50 CG) ─────────────────────────
        float mu   = vehicleParams.tireFrictionCoefficient;
        float Fz_f = 0.5f * m * vehicleParams.gravity * lr / (lf + lr);
        float Fz_r = 0.5f * m * vehicleParams.gravity * lf / (lf + lr);

        // ── Equilibrium slip angles at the current operating point ────────────
        float delta  = angles.average;
        float fv     = ((float)System.Math.Tanh(10.0 * vxS - 8.0) + 1f) / 2f;
        float alphaF = fv * (delta - Mathf.Atan2(yDot + lf * psiDot, vxS));
        float alphaR = fv * Mathf.Atan2(lr * psiDot - yDot, vxS);

        // ── Scheduled effective axle stiffnesses (= 2·Caf at α=0, → 0 at saturation) ─
        float Caf_eff = 2f * TireSlopeAtAlpha(alphaF, vehicleParams.Caf, mu, Fz_f);
        float Car_eff = 2f * TireSlopeAtAlpha(alphaR, vehicleParams.Car, mu, Fz_r);

        // ── B: ∂(v̇y)/∂δ = (Caf_eff·cosδ − Fyf_eq·sinδ + Fxf·cosδ)/m ─────────
        float cosD   = Mathf.Cos(delta);
        float sinD   = Mathf.Sin(delta);
        float Fyf_eq = 2f * TireForcePerWheel(alphaF, vehicleParams.Caf, mu, Fz_f);
        float Fx     = Fx_tot;
        float bCoeff = (Caf_eff * cosD - Fyf_eq * sinD + Fx * cosD) / m;
        float[] Bc   = { 0f, bCoeff, 0f, lf * bCoeff * m / Iz };   // yaw = lf·(same)/Iz

        float[,] Ac = {
            { 0, 1, vxS * Mathf.Cos(psi), 0 },
            { 0, -(Caf_eff + Car_eff)/(m*vxS), 0, (lr*Car_eff - lf*Caf_eff)/(m*vxS) - vxS },
            { 0, 0, 0, 1 },
            { 0, (lr*Car_eff - lf*Caf_eff)/(Iz*vxS), 0, -(lf*lf*Caf_eff + lr*lr*Car_eff)/(Iz*vxS) }
        };
        return (Ac, Bc);
    }

    public (float[,] Ac_long, float[] Bc_long) GetLongJacobian() {
        // A[1,1]: linearised drag damping  ∂(v̇x)/∂vx = -(ρ·Cd·A·|vx|)/m
        // Bc: acceleration input [m/s²] — feedforward cancellation absorbs mass+resistances,
        //     matching Python vehicle_6d._update_jacobians where B_long[1] ≈ 1.
        float dragLin = 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea
                      * Mathf.Abs(vx) / vehicleParams.mass;
        float[,] Ac = { { 0, 1 }, { 0, -dragLin } };
        float[]  Bc = { 0f, 1f };   // u [m/s²], not force — consistent with Nash solver bounds
        return (Ac, Bc);
    }

    public float[] GetLongState()      => new[] { position.z, vx };
    public float[] GetLatStateVector() => new[] { y, yDot, psi, psiDot };
    public float   GetX()              => position.z;
    public float   GetLength()         => vehicleParams.length;

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
