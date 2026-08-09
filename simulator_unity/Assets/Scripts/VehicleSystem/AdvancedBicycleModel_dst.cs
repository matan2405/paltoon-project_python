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
    private float vyBody = 0f;  // body-frame lateral velocity [m/s] = sNew[3]
    private float _x0 = 0f;   // world-X at spawn; y = _x0 - position.x
    private float psi = 0f;
    private float psiDot = 0f;
    private float vx = 0;
    private float ax = 0;
    private Vector4 state;
    private float Fx_tot     = 0f;
    private float _fxContact = 0f;
    private float _fxResist  = 0f;
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
    private float ay      = 0f;  // body-frame lateral acceleration [m/s²], from Eq. 3.11b
    private float y_dot_dot = 0f; // world-frame lateral acceleration [m/s²], from Eq. 3.11b result
    private float Totaldistance = 0f;
    private float m_eff;  // Belousov Eq. 3.8: m + Iw/r²

    void Awake()
    {
        // Initialize early so PlatoonManager.Start() can call GetVehicleParameters/GetCoastDeceleration
        vehicleParams = new VehicleParameters();
        EngineAudio engineAudio = GetComponent<EngineAudio>();
        powertrain = new Powertrain(vehicleParams, engineAudio);
        brakeSystem = new BrakeSystem(vehicleParams);
        steeringSystem = new SteeringSystem(vehicleParams);

        // Belousov Eq. 3.8: m_eff = m + Iw/r²  (wheel inertia referred to vehicle body)
        float r = vehicleParams.wheelRadius;
        m_eff = vehicleParams.mass + vehicleParams.wheelInertia / (r * r);
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

        // Engine braking always acts when throttle is released and vehicle is moving.
        // In the coast zone (no brake) it is lumped with naturalDeceleration.
        // When the brake pedal is pressed it still acts — drivetrain compression does
        // not disappear just because the friction brakes are also engaged.
        float Fengbrk = 0f;
        float naturalDeceleration = 0f;
        if (throttleInput < 0.001f && vx > 0.1f)
        {
            Fengbrk = powertrain.GetEngineBrakingForce(vx);
            if (brakeInput < 0.001f)
            {
                naturalDeceleration = powertrain.CalculateNaturalDeceleration(vx, powertrain.EngineRpm);
                naturalDeceleration += vx * vx * 0.01f;
            }
        }

        // Fxf_contact / Fxr_contact: forces at each axle contact patch (Belousov Eq. 3.11b/c).
        // Audi TT is FWD: engine force acts only at front axle.
        // Body-level resistances (Faero, Rr, Rg) are NOT contact-patch forces — they enter
        // only Eq. 3.11a via Fx_tot, not the sinδ terms of 3.11b/c.
        float Fxf_contact = Fxf_engine - FbrakeFront;
        float Fxr_contact = -FbrakeRear;

        // Fx_contact: contact-patch forces only — these are coupled to steering angle (enter cosδ/sinδ)
        // Fx_resist:  body-level resistances — act along vehicle body axis, NOT coupled to wheel angle
        float Fx_contact = Fxf_contact + Fxr_contact;
        float Fx_resist  = Faero + Rxf + Rxr + slopeForce + naturalDeceleration + Fengbrk;
        Fx_tot    = Fx_contact - Fx_resist;  // kept for GetFx_tot() / AutoMode RPM update
        _fxContact = Fx_contact;
        _fxResist  = Fx_resist;

        if (AutoMode) {
            powertrain.UpdateRpmForAutonomousMode(Fx_contact, vx);
        }

        UpdateDynamics6D(Fx_contact, Fxf_contact, Fx_resist, angles.average, dt);

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
    public float GetFx_tot()     => Fx_tot;
    public float GetFx_contact() => _fxContact;
    public float GetFx_resist()  => _fxResist;
    public float GetVy()         => vyBody;  // body-frame lateral velocity (sNew[3])
    public float GetLambda()     => angles.average;
    public float GetThrottleRaw()  => throttleInput;
    public float GetBrakeRaw()     => brakeInput;
    public float GetSteeringRaw()  => steeringInput;
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
    // Fx_contact : contact-patch forces [N] — enter Eq. 3.11a via cosδ, and 3.11b/c via sinδ
    // Fxf_contact: front axle contact-patch force [N] — enters Eq. 3.11b/c sinδ terms only
    // Fx_resist  : body-level resistances [N] (Faero+Rr+Rg) — enter Eq. 3.11a directly, NOT via cosδ
    void UpdateDynamics6D(float Fx_contact, float Fxf_contact, float Fx_resist, float delta, float dt) 
    {
        float deltaRL = RateLimitDelta(delta, dt);
        float[] s    = GetState6DArray();
        float[] sNew = RK4Step(s, Fx_contact, Fxf_contact, Fx_resist, deltaRL, dt);

        sNew[1] = Mathf.Clamp(sNew[1], 0f, vehicleParams.maxVelocity / 3.6f);

        if (float.IsNaN(sNew[0]) || float.IsNaN(sNew[1]) || float.IsNaN(sNew[2]) ||
            float.IsNaN(sNew[3]) || float.IsNaN(sNew[4]) || float.IsNaN(sNew[5])) // check for NaN in 6D state
        {
            Debug.LogWarning("[Belousov] NaN in 6D state — resetting lateral velocities");
            sNew[3] = 0f; sNew[5] = 0f;
        }
        ax = (sNew[1] - s[1]) / dt; // longitudinal acceleration (Belousov v̇x)
        position.z = sNew[0]; // longitudinal position (Belousov z)
        vx         = sNew[1];// longitudinal velocity (Belousov vx)
        position.x = sNew[2];// lateral position (Belousov x)
        vyBody     = sNew[3];      // body-frame lateral velocity
        psi        = sNew[4];// yaw angle (Belousov ψ)
        psiDot     = sNew[5];// yaw rate (Belousov ψ̇)
        ay = (sNew[3] - s[3]) / dt; // lateral acceleration (Belousov v̇y)

        yDot = -(vx * Mathf.Sin(psi) + vyBody * Mathf.Cos(psi));
        y    = _x0 - position.x;  // exact world-frame lateral tracking
        // a_lat = v̇y - vx·ψ̇  (Eq. 3.11b result, world-frame lateral acceleration)
        y_dot_dot = ay - vx * psiDot; // world-frame lateral acceleration
    }

    float[] RK4Step(float[] s, float Fx_contact, float Fxf_contact, float Fx_resist, float delta, float dt) {
        float[] k1 = Derivatives6D(s, Fx_contact, Fxf_contact, Fx_resist, delta);
        float[] k2 = Derivatives6D(Add6(s, Scale6(k1, dt / 2f)), Fx_contact, Fxf_contact, Fx_resist, delta);
        float[] k3 = Derivatives6D(Add6(s, Scale6(k2, dt / 2f)), Fx_contact, Fxf_contact, Fx_resist, delta);
        float[] k4 = Derivatives6D(Add6(s, Scale6(k3, dt)),       Fx_contact, Fxf_contact, Fx_resist, delta);
        float[] wsum = Add6(Add6(k1, Scale6(k2, 2f)), Add6(Scale6(k3, 2f), k4));
        return Add6(s, Scale6(wsum, dt / 6f));
    }

    // Belousov Eq. 3.11a/b/c:
    //   Fx_contact  → 3.11a via cosδ; Fxf_contact → 3.11b/c sinδ terms
    //   Fx_resist   → 3.11a directly (no cosδ): body-level drag, rolling, grade
    float[] Derivatives6D(float[] s, float Fx_contact, float Fxf_contact, float Fx_resist, float delta) {
        float vxS    = s[1], vyS = s[3], psiS = s[4], psiDotS = s[5];
        float vxSafe = Mathf.Max(Mathf.Abs(vxS), 1.0f);

        float fv = ((float)System.Math.Tanh(10.0 * Mathf.Abs(vxS) - 8.0) + 1f) / 2f;

        float alphaF = fv * (delta - Mathf.Atan2(vyS + vehicleParams.lf * psiDotS, vxSafe));
        float alphaR = fv * Mathf.Atan2(vehicleParams.lr * psiDotS - vyS, vxSafe);

        // Rajamani Eq. 13.45 — parabolic tire model (replaces linear Fyf = C·α)
        // Caf/Car are per-wheel stiffness values → divide axle load by 2, then sum both wheels (×2)
        // Normal loads: 50/50 CG split → Fz_axle_f = m·g·lr/(lf+lr), per wheel = /2
        float mu         = vehicleParams.tireFrictionCoefficient;
        float Fz_f_wheel = vehicleParams.mass * vehicleParams.gravity * vehicleParams.lr
                         / (vehicleParams.lf + vehicleParams.lr) / 2f;
        float Fz_r_wheel = vehicleParams.mass * vehicleParams.gravity * vehicleParams.lf
                         / (vehicleParams.lf + vehicleParams.lr) / 2f;
        float Fyf = 2f * TireForce(alphaF, vehicleParams.Caf, mu, Fz_f_wheel);
        float Fyr = 2f * TireForce(alphaR, vehicleParams.Car, mu, Fz_r_wheel);
        float m   = vehicleParams.mass;

        float cosD = Mathf.Cos(delta), sinD = Mathf.Sin(delta);
        float cosP = Mathf.Cos(psiS),  sinP = Mathf.Sin(psiS);

        float zDot = vxS * cosP - vyS * sinP;
        float xDot = vxS * sinP + vyS * cosP;

        // Eq. 3.11a: contact forces enter via cosδ; body resistances subtract directly (no cosδ)
        // m_eff = m + Iw/r² (Belousov Eq. 3.8)
        // fv gates the centripetal coupling vyS·ψ̇: physically valid at speed but
        // undefined near vx=0 where vy/ψ̇ retain inertial values → explosion without gating.
        float vxDot = fv * vyS * psiDotS + (Fx_contact * cosD - Fyf * sinD - Fx_resist) / m_eff;

        float k_vy  = vehicleParams.tireFrictionCoefficient * vehicleParams.gravity / 0.5f;
        float k_psi = vehicleParams.tireFrictionCoefficient * vehicleParams.mass
                    * vehicleParams.gravity * vehicleParams.wheelbase
                    / (4f * vehicleParams.Iz * 0.3f);

        // Eq. 3.11b: only front contact-patch force contributes via sinδ
        // fv gates -vxS·ψ̇ for the same reason as vyS·ψ̇ in vxDot: near vx=0 this
        // coupling loses physical meaning and amplifies vy/ψ̇ residuals.
        float vyDot = (Fyf * cosD + Fyr + Fxf_contact * sinD) / m - fv * vxS * psiDotS
                    - (1f - fv) * k_vy * vyS;

        // Eq. 3.11c: only front contact-patch force generates yaw moment via sinδ
        float psiDotDot = (vehicleParams.lf * Fyf * cosD
                         - vehicleParams.lr * Fyr
                         + vehicleParams.lf * Fxf_contact * sinD) / vehicleParams.Iz
                         - (1f - fv) * k_psi * psiDotS;

        return new[] { zDot, vxDot, xDot, vyDot, psiDotS, psiDotDot };
    }

    float RateLimitDelta(float delta, float dt) {
        float maxRate = SimCfg.I != null ? SimCfg.I.Timing.MaxSteerRate : 0.087f;
        float limited = Mathf.Clamp(delta, _prevDelta - maxRate, _prevDelta + maxRate);
        _prevDelta = limited;
        return limited;
    }

    public float GetALat() => y_dot_dot;  // world-frame lateral acceleration (Belousov Eq. 3.11b result)

        float[] GetState6DArray() => new[] { position.z, vx, position.x, vyBody, psi, psiDot };

    // Rajamani Eq. 13.45 — parabolic pressure distribution tire model.
    // Linear (F ≈ C·α) for small slip; saturates at μ·Fz for full sliding.
    static float TireForce(float alpha, float C, float mu, float Fz)
    {
        float theta = C / (3f * mu * Fz);
        float S     = Mathf.Tan(alpha);
        if (Mathf.Abs(S) <= 1f / theta)
            return mu * Fz * (3f*theta*S - 3f*theta*theta*S*S + theta*theta*theta*S*S*S);
        return Mathf.Sign(S) * mu * Fz;
    }

    static float[] Add6(float[] a, float[] b) =>
        new[] { a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5] };

    static float[] Scale6(float[] v, float sc) =>
        new[] { v[0]*sc, v[1]*sc, v[2]*sc, v[3]*sc, v[4]*sc, v[5]*sc };

    // ── Nash linearization methods ────────────────────────────────────────
    // Jacobian of Belousov Eq. 3.11b/c, linearised around δ=0 (straight driving).
    // State: [y, vy, ψ, ψ̇].  Caf/Car are per-wheel → factor 2 for full axle.
    // k_vy / k_psi are the low-speed numerical damping terms from Derivatives6D —
    // included here so the Jacobian exactly matches the simulated equations.
    //
    // Derivation (δ=0 → cosδ=1, sinδ=0, αf=-(vy+lf·ψ̇)/vx, αr=-(vy-lr·ψ̇)/vx):
    //   A[1,1] = ∂v̇y/∂vy  = -2(Caf+Car)/(m·vx) - (1-fv)·k_vy
    //   A[1,3] = ∂v̇y/∂ψ̇  =  2(lr·Car - lf·Caf)/(m·vx) - vx
    //   A[3,1] = ∂ψ̈/∂vy   =  2(lr·Car - lf·Caf)/(Iz·vx)
    //   A[3,3] = ∂ψ̈/∂ψ̇   = -2(lf²·Caf + lr²·Car)/(Iz·vx) - (1-fv)·k_psi
    //   B[1]   = ∂v̇y/∂δ  =  2·Caf/m
    //   B[3]   = ∂ψ̈/∂δ   =  2·lf·Caf/Iz
    public (float[,] Ac_lat, float[] Bc_lat) GetLatJacobian() {
        float vxS = Mathf.Max(Mathf.Abs(vx), 0.5f);
        float m   = vehicleParams.mass, Iz = vehicleParams.Iz;
        float lf  = vehicleParams.lf,   lr = vehicleParams.lr;
        float Caf = vehicleParams.Caf,  Car = vehicleParams.Car;
        float fv  = ((float)System.Math.Tanh(10.0 * vxS - 8.0) + 1f) / 2f;

        float k_vy  = vehicleParams.tireFrictionCoefficient * vehicleParams.gravity / 0.5f;
        float k_psi = vehicleParams.tireFrictionCoefficient * vehicleParams.mass
                    * vehicleParams.gravity * vehicleParams.wheelbase
                    / (4f * vehicleParams.Iz * 0.3f);

        // Standard bicycle model convention: y = pos.x - _x0 (rightward = +).
        // ẏ = vy + vx·ψ  →  Ac[0,1]=+1, Ac[0,2]=+vx
        float[,] Ac = {
            { 0,  1,  vxS,  0 },
            { 0, -2f*(Caf+Car)/(m*vxS) - (1f-fv)*k_vy,      0,  2f*(lr*Car - lf*Caf)/(m*vxS) - vxS },
            { 0,  0,  0,  1 },
            { 0,  2f*(lr*Car - lf*Caf)/(Iz*vxS), 0, -2f*(lf*lf*Caf + lr*lr*Car)/(Iz*vxS) - (1f-fv)*k_psi }
        };
        float[] Bc = { 0f, 2f*Caf/m, 0f, 2f*lf*Caf/Iz };
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
                      * Mathf.Abs(vx) / m_eff;
        float[,] Ac = { { 0, 1 }, { 0, -dragLin } };
        float[]  Bc = { 0f, 1f };   // u = a_des [m/s²], not force [N]
        return (Ac, Bc);
    }

    public float[] GetLongState()      => new[] { position.z, vx };
    // Solver convention: y_solver = pos.x - _x0 (rightward = +), matching GetLatJacobian Ac[0,1]=+1.
    // GetY() returns _x0 - pos.x (Belousov), so y_solver = -GetY(). yDot likewise negated.
    public float[] GetLatStateVector() => new[] { -y, -yDot, psi, psiDot };
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

    // Physical acceleration bounds [m/s²] from the coupled Eq. 3.11a:
    //   m·(v̇x − vy·ψ̇) = Fxf·cosδ + Fxr − Fyf·sinδ − Ra − Rr
    // aMax: full-throttle net (Fxf_engine_max·cosδ − Faero − Rr + F_couple) / m
    // aMin: full-brake + engine braking (−FbrakeMax − F_engBrk − Faero − Rr + F_couple) / m
    public void GetPhysicalAccelBounds(out float aMin, out float aMax)
    {
        float m     = vehicleParams.mass;
        float delta = angles.average;
        float cosD  = Mathf.Cos(delta);
        float sinD  = Mathf.Sin(delta);

        float vxSafe = Mathf.Max(Mathf.Abs(vx), 1.0f);
        float fv     = ((float)System.Math.Tanh(10.0 * Mathf.Abs(vx) - 8.0) + 1f) / 2f;
        float alphaF = fv * (delta - Mathf.Atan2(vyBody + vehicleParams.lf * psiDot, vxSafe));
        float mu     = vehicleParams.tireFrictionCoefficient;
        float Fz_f   = m * vehicleParams.gravity * vehicleParams.lr
                     / (vehicleParams.lf + vehicleParams.lr) / 2f;
        float Fyf    = 2f * TireForce(alphaF, vehicleParams.Caf, mu, Fz_f);

        float Faero = 0.5f * 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;
        float Cr    = 0.012f * (1f + vx * 0.008f);
        float Frr   = m * vehicleParams.gravity * Cr;

        float F_couple = m * vyBody * psiDot - Fyf * sinD;

        float Fxf_max = powertrain.CalculateMaxForceAtCurrentSpeed(vx);
        aMax = (Fxf_max * cosD - Faero - Frr + F_couple) / m;
        aMax = Mathf.Max(aMax, 0f);

        float Fbrake_max  = brakeSystem.CalculateBrakeForce(1.0f, vx);
        float F_eng_coast = powertrain.GetEngineBrakingForce(vx);
        aMin = (-Fbrake_max - F_eng_coast - Faero - Frr + F_couple) / m;
        aMin = Mathf.Min(aMin, 0f);
    }

    // Control input u = a_des [m/s²] → (throttle, brake) ∈ [0,1]
    // Three zones: throttle | coast (engine braking sufficient) | brake
    // Inverts the coupled Eq. 3.11a: F_need = m·a_des − F_couple + Ra + Rr
    // In the throttle zone divides by cosδ to recover Fxf from its projected value.
    public (float throttle, float brake) AccelerationToInputs(float a_des)
    {
        float m     = vehicleParams.mass;
        float delta = angles.average;
        float cosD  = Mathf.Cos(delta);
        float sinD  = Mathf.Sin(delta);

        float vxSafe = Mathf.Max(Mathf.Abs(vx), 1.0f);
        float fv     = ((float)System.Math.Tanh(10.0 * Mathf.Abs(vx) - 8.0) + 1f) / 2f;
        float alphaF = fv * (delta - Mathf.Atan2(vyBody + vehicleParams.lf * psiDot, vxSafe));
        float mu     = vehicleParams.tireFrictionCoefficient;
        float Fz_f   = m * vehicleParams.gravity * vehicleParams.lr
                     / (vehicleParams.lf + vehicleParams.lr) / 2f;
        float Fyf    = 2f * TireForce(alphaF, vehicleParams.Caf, mu, Fz_f);

        float Faero = 0.5f * 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;
        float Cr    = 0.012f * (1f + vx * 0.008f);
        float Frr   = m * vehicleParams.gravity * Cr;

        float F_couple = m * vyBody * psiDot - Fyf * sinD;
        float F_need   = m * a_des - F_couple + Faero + Frr;
        float F_eng    = powertrain.GetEngineBrakingForce(vx) + Faero + Frr;

        if (F_need >= 0f) {
            float Fxf_need = cosD > 0.01f ? F_need / cosD : F_need;
            float maxF     = powertrain.CalculateMaxForceAtCurrentSpeed(vx);
            return (Mathf.Clamp01(maxF > 1f ? Fxf_need / maxF : 0f), 0f);
        } else if (F_need >= -F_eng)
            return (0f, 0f);
        else {
            float maxBrakeF = brakeSystem.CalculateBrakeForce(1.0f, vx);
            return (0f, Mathf.Clamp01(maxBrakeF > 1f ? (-F_need - F_eng) / maxBrakeF : 0f));
        }
    }
}