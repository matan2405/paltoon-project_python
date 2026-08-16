using System.Linq;
using UnityEngine;
using System.Collections;

public class Powertrain
{
    private VehicleParameters vehicleParams;
    private int engineState = 0;  // 0 = off, 1 = starting, 2 = running
    private EngineAudio engineAudio;
    private MonoBehaviour monoBehaviourInstance;

    // ✅ NEW: Use separate transmission system
    private Transmission transmission;

    [Header("Audi TT 2.0 TFSI - Engine Only")]
    private const float maxTorque = 370f;      // Peak torque: 370 Nm
    private const float maxPowerPS = 230f;     // Peak power: 230 PS
    private const float maxPowerKW = 169f;     // 230 PS = 169 kW
    private const float idleRPM = 800f;        // Engine idle RPM
    private const float redlineRPM = 6700f;    // Engine redline

    // RPM breakpoints: idle→1000 and 1000→1600 from dyno graph; plateau from official spec; power peak end from graph
    private const float torqueRiseStartRPM  = 1000f;  // graph: rapid rise begins ~1000 rpm (~240 Nm)
    private const float torqueRiseEndRPM    = 1600f;  // spec:  370 Nm plateau starts at 1600 rpm
    private const float torquePlateauEndRPM = 4300f;  // spec:  370 Nm plateau ends at 4300 rpm
    private const float powerPeakStartRPM   = 4500f;  // spec:  power-limited zone starts
    private const float powerPeakEndRPM     = 6400f;  // graph: power plateau ends at ~6400 rpm

    // Engine response characteristics
    private float engineRpmAcceleration = 2500f; // FASTER RPM increases
    private float engineRpmDeceleration = 2000f; // FASTER RPM decreases
    private float engineInertia = 0.1f;          // LESS inertia

    private float angularVelocity;

    // Rajamani Eq. 9.1.3 quadratic engine friction coefficients (Tf = a0·ωe² + a1·ωe + a2).
    // Calibrated so Fengbrk ≈ 50–70 N at highway RPM (~2900 rpm, 33 m/s in 5th gear),
    // giving total coast decel ≈ 0.65–0.75 m/s² (literature: 0.2–1.0 m/s²).
    private const float engBrkA0 = 0.0008f; // quadratic [N/(rpm²)] → ~6.8 N at 2900 rpm contribution
    private const float engBrkA1 = 0.008f;  // linear    [N/rpm]   → ~23 N at 2900 rpm contribution
    private const float engBrkA2 = 15f;     // constant  [N]       → 15 N base drag

    public float EngineRpm { get; private set; }
    public int EngineState
    {
        get { return engineState; }
        private set { engineState = value; }
    }

    public float GetidleRPM() => idleRPM;

    public Powertrain(VehicleParameters vehicleParams, EngineAudio audio = null)
    {
        this.vehicleParams = vehicleParams;
        this.engineAudio = audio;
        if (audio != null)
        {
            this.monoBehaviourInstance = audio;
        }

        // ✅ NEW: Initialize transmission system
        transmission = new Transmission(vehicleParams);

        engineState = 0;
        EngineRpm = 0;
    }

    public void StartEngine()
    {
        if (engineState == 0)
        {
            if (engineAudio != null && monoBehaviourInstance != null)
            {
                monoBehaviourInstance.StartCoroutine(engineAudio.StartEngine());
            }

            engineState = 1;

            if (engineAudio != null)
            {
                monoBehaviourInstance.StartCoroutine(HandleEngineStart());
            }
            else
            {
                engineState = 2;
                EngineRpm = idleRPM;
            }
        }
    }

    private IEnumerator HandleEngineStart()
    {
        yield return new WaitForSeconds(0.6f);
        engineState = 2;
        EngineRpm = idleRPM;
    }

    public void StopEngine()
    {
        engineState = 0;
        EngineRpm = 0;
        if (engineAudio != null)
        {
            engineAudio.StopEngine();
        }
    }

    // ✅ UPDATED: Main update method now updates transmission
    public void UpdatePowertrain(float deltaTime, float throttle, float vx)
    {
        if (engineState != 2) return;

        // Update transmission system
        transmission.Update(deltaTime, EngineRpm, vx, throttle);
    }

    // Accurate engine torque curve based on the graphs
    private float CalculateEngineTorque(float rpm)
    {
        if (rpm < idleRPM || rpm > redlineRPM)
            return 0f;

        if (rpm <= torqueRiseStartRPM)
        {
            // 800–1000 rpm: gentle rise from idle torque to ~240 Nm (graph)
            return Mathf.Lerp(180f, 240f, (rpm - idleRPM) / (torqueRiseStartRPM - idleRPM));
        }
        else if (rpm <= torqueRiseEndRPM)
        {
            // 1000–1600 rpm: rapid rise to peak 370 Nm (graph + spec)
            return Mathf.Lerp(240f, maxTorque, (rpm - torqueRiseStartRPM) / (torqueRiseEndRPM - torqueRiseStartRPM));
        }
        else if (rpm <= torquePlateauEndRPM)
        {
            return maxTorque;
        }
        else if (rpm <= powerPeakStartRPM)
        {
            return Mathf.Lerp(maxTorque, maxTorque * 0.95f, (rpm - torquePlateauEndRPM) / (powerPeakStartRPM - torquePlateauEndRPM));
        }
        else if (rpm <= powerPeakEndRPM)
        {
            angularVelocity = (2f * Mathf.PI * rpm) / 60f;
            return (maxPowerKW * 1000f) / angularVelocity;
        }
        else
        {
            float rpmRatio = (rpm - powerPeakEndRPM) / (redlineRPM - powerPeakEndRPM);
            float powerMultiplier = Mathf.Lerp(1.0f, 0.7f, rpmRatio);
            float adjustedPowerKW = maxPowerKW * powerMultiplier;

            angularVelocity = (2f * Mathf.PI * rpm) / 60f;
            return (adjustedPowerKW * 1000f) / angularVelocity;
        }
    }

    // Calculate actual power output
    private float CalculateEnginePower(float rpm)
    {
        float torque = CalculateEngineTorque(rpm);
        angularVelocity = (2f * Mathf.PI * rpm) / 60f;
        return (torque * angularVelocity) / 1000f; // Power in kW
    }

    // Rajamani Eq. 9.1.3: Tf = a0·ωe² + a1·ωe + a2, scaled by gear ratio.
    // Physical interpretation: viscous drag (quadratic) + dry friction (linear) + constant pumping loss.
    // Gear factor: higher gear = less engine braking (engine sees lower RPM per wheel rev).
    private float CalculateEngineBrakingForce(float currentRpm, float vx)
    {
        if (engineState != 2 || Mathf.Abs(vx) < 0.1f) return 0f;

        float N = currentRpm / 1000f; // normalise to kRPM for numerical stability
        float Tf = engBrkA0 * N * N + engBrkA1 * N + engBrkA2;

        // Scale by overall gear ratio (higher gear = lower engine torque at wheels).
        // GetCurrentGearReduction() returns overall ratio (≈3.2 in 5th); divide by top-gear ratio
        // so Fengbrk is roughly constant in top gear and larger in low gears.
        float gearFactor = transmission.GetCurrentGearReduction() / 3.2f;

        return Mathf.Max(0f, Tf * gearFactor);
    }

    public float CalculateEngineForce(float throttle, float vx)
    {
        if (engineState != 2) return 0f;

        // ✅ UPDATED: Update powertrain with current conditions
        UpdatePowertrain(Time.fixedDeltaTime, throttle, vx);

        if (throttle > 0.001f)
        {
            // ✅ REALISTIC: Normal automatic transmission behavior for research
            
            int currentGearNumber = transmission.GetCurrentGear();
            
            // ✅ REALISTIC RPM LIMITS: What a normal automatic transmission allows
            float[] normalMaxRpm = { 3000f, 3400f, 3800f, 4200f, 4600f, 5200f }; // Conservative/realistic
            
            float currentGearMaxRpm = normalMaxRpm[Mathf.Clamp(currentGearNumber - 1, 0, 5)];
            
            // Calculate current RPM needed for this speed
            float speedBasedRpm = transmission.CalculateRequiredEngineRPM(vx);
            
            // ✅ NORMAL THROTTLE RESPONSE: Moderate aggressiveness
            float throttleRpmRange = 1200f; // Standard throttle response
            
            float targetRpm = speedBasedRpm + (throttle * throttleRpmRange);
            
            // ✅ GEAR-SPECIFIC LIMITS: Each gear has realistic maximum
            targetRpm = Mathf.Min(targetRpm, currentGearMaxRpm);
            
            // ✅ SAFETY LIMIT: Normal automatic transmission protection
            targetRpm = Mathf.Min(targetRpm, 5500f); // Safe operating limit
            
            // ✅ MINIMUM: Don't go below idle
            targetRpm = Mathf.Max(targetRpm, idleRPM);
            
            // ✅ RESEARCH DEBUG: Log for analysis
            if (Time.fixedTime % 2f < Time.fixedDeltaTime) // Log every 2 seconds
            {
                Debug.Log($"Research Data - Gear {currentGearNumber}: Speed={vx*3.6f:F1}km/h, RPM={EngineRpm:F0}, Target={targetRpm:F0}, Max={currentGearMaxRpm:F0}");
            }
            
            // ✅ IMPROVED: During shifting, maintain current RPM
            if (!transmission.IsShifting())
            {
                EngineRpm = Mathf.MoveTowards(EngineRpm, targetRpm, engineRpmAcceleration * Time.fixedDeltaTime);
            }

            // Calculate engine torque at current RPM
            float engineTorque = CalculateEngineTorque(EngineRpm);

            // ✅ UPDATED: Use transmission for gear reduction
            float totalGearReduction = transmission.GetCurrentGearReduction();
            float torqueAtWheels = engineTorque * totalGearReduction;

            // Convert torque to force at wheel contact patch
            float engineForce = torqueAtWheels / vehicleParams.wheelRadius;

            // Clamp by power and traction limits
            float maxF = CalculateMaxForceAtCurrentSpeed(vx);
            return Mathf.Min(engineForce * throttle, maxF);
        }
        else
        {
            // ✅ FASTER: Engine RPM follows wheel speed immediately when coasting
            float requiredRpm = transmission.CalculateRequiredEngineRPM(vx);

            // ✅ INSTANT CONNECTION: RPM follows wheels almost instantly (like automatic transmission)
            EngineRpm = Mathf.Lerp(EngineRpm, requiredRpm, 20f * Time.fixedDeltaTime);

            // ✅ STRONG: Engine braking force
            float engineBrakingForce = CalculateEngineBrakingForce(EngineRpm, vx);

            return -engineBrakingForce; // Negative force = deceleration
        }
    }

    /// Update RPM in autonomous mode - similar to CalculateEngineForce function
    public void UpdateRpmForAutonomousMode(float targetForce, float vx)
    {
        if (engineState != 2) return;

        // Update transmission system like in regular mode
        UpdatePowertrain(Time.fixedDeltaTime, 0.5f, vx); // medium throttle

        if (targetForce > 50f) // Positive force - acceleration
        {
            // ✅ Use transmission to get current gear number
            // This is similar to CalculateEngineForce but for autonomous mode
            int currentGearNumber = transmission.GetCurrentGear();

            // ✅ Use realistic RPM limits for autonomous mode
            float[] normalMaxRpm = { 3000f, 3400f, 3800f, 4200f, 4600f, 5200f };
            float currentGearMaxRpm = normalMaxRpm[Mathf.Clamp(currentGearNumber - 1, 0, 5)];

            // Calculate basic RPM by speed
            float speedBasedRpm = transmission.CalculateRequiredEngineRPM(vx);

            // ✅ Calculate "virtual throttle" from required force
            float maxPossibleForce = CalculateMaxForceAtCurrentSpeed(vx);
            float virtualThrottle = Mathf.Clamp01(targetForce / maxPossibleForce);

            // ✅ Same throttle response as in regular mode
            float throttleRpmRange = 1200f;
            float targetRpm = speedBasedRpm + (virtualThrottle * throttleRpmRange);

            // ✅ Same limitations
            targetRpm = Mathf.Min(targetRpm, currentGearMaxRpm);
            targetRpm = Mathf.Min(targetRpm, 5500f); // Safety limit
            targetRpm = Mathf.Max(targetRpm, idleRPM);

            // ✅ Research log
            if (Time.fixedTime % 2f < Time.fixedDeltaTime)
            {
                Debug.Log($"Autonomous Mode - Gear {currentGearNumber}: Speed={vx * 3.6f:F1}km/h, RPM={EngineRpm:F0}, Target={targetRpm:F0}, Force={targetForce:F0}N");
            }

            // ✅ Update RPM only if not shifting
            if (!transmission.IsShifting())
            {
                EngineRpm = Mathf.MoveTowards(EngineRpm, targetRpm, engineRpmAcceleration * Time.fixedDeltaTime);
            }
        }
        else if (targetForce < -50f) // Negative force - braking
        {
            // ✅ Same logic as in no-throttle mode

            float requiredRpm = transmission.CalculateRequiredEngineRPM(vx);

            // ✅ Same immediate connection to wheels
            EngineRpm = Mathf.Lerp(EngineRpm, requiredRpm, 20f * Time.fixedDeltaTime);
        }
        else // Small force - constant driving
        {
            // ✅ RPM follows speed smoothly
            float requiredRpm = transmission.CalculateRequiredEngineRPM(vx);

            // Smoother update than in braking
            float rpmChangeRate = Mathf.Abs(requiredRpm - EngineRpm) > 200f ?
                                (requiredRpm > EngineRpm ? engineRpmAcceleration : engineRpmDeceleration) :
                                1500f; // Slow change for constant speed

            EngineRpm = Mathf.MoveTowards(EngineRpm, requiredRpm, rpmChangeRate * Time.fixedDeltaTime);
        }

        // ✅ Final limitation
        EngineRpm = Mathf.Clamp(EngineRpm, idleRPM, redlineRPM);
    }
    /// Calculate maximum possible force at current speed
    public float CalculateMaxForceAtCurrentSpeed(float vx)
    {
        // Torque-limited force: actual engine torque at current RPM × gear reduction → wheel contact patch
        float totalGearReduction = transmission.GetCurrentGearReduction();
        float fTorque = (CalculateEngineTorque(EngineRpm) * totalGearReduction) / vehicleParams.wheelRadius;

        // Power-limited force: F = P(rpm) / vx — uses actual power at current RPM, not peak.
        // Clamp denominator to avoid unrealistically large values at near-zero speed.
        float vRef   = Mathf.Max(Mathf.Abs(vx), 7f);
        float fPower = (CalculateEnginePower(EngineRpm) * 1000f) / vRef;

        // Traction limit (FWD): only front axle drives.
        // Front axle static load = mass × 0.62 (62/38 weight split, VehicleParameters).
        // μ = tireFrictionCoefficient (0.8, dry asphalt, VehicleParameters).
        // Under acceleration load transfers rearward → static split is an upper bound.
        float fTraction = vehicleParams.tireFrictionCoefficient
                        * vehicleParams.mass * 0.62f
                        * vehicleParams.gravity;

        return Mathf.Min(fTorque, Mathf.Min(fPower, fTraction));
    }
    // Engine braking force at a given speed, without side effects (read-only)
    public float GetEngineBrakingForce(float vx) {
        float rpm = transmission.CalculateRequiredEngineRPM(vx);
        return CalculateEngineBrakingForce(rpm, vx);
    }

    // ✅ MINIMAL: Natural deceleration since engine braking does most work
    public float CalculateNaturalDeceleration(float vx, float engineRpm)
    {
        if (vx < 0.1f) return 0f;

        // ✅ VERY MINIMAL: Only basic mechanical resistance
        float engineFrictionBase = 1f;  // Reduced even more
        float engineRpmFactor = Mathf.Clamp01(engineRpm / 6000f);
        float engineFriction = engineFrictionBase * (1f + engineRpmFactor * 0.05f);

        float transmissionFriction = 0.5f; // Minimal
        float differentialFriction = 0.5f; // Minimal
        float wheelBearingFriction = 0.5f; // Minimal

        return engineFriction + transmissionFriction + differentialFriction + wheelBearingFriction;
    }

    // ✅ UPDATED: Debug functions now use transmission system
    public float GetCurrentEnginePowerKW()
    {
        return CalculateEnginePower(EngineRpm);
    }

    public float GetCurrentEngineTorque()
    {
        return CalculateEngineTorque(EngineRpm);
    }

    // ✅ UPDATED: Transmission-related getters
    public int GetCurrentGear()
    {
        return transmission.GetCurrentGear();
    }

    public bool IsShifting()
    {
        return transmission.IsShifting();
    }

    public float GetTheoreticalSpeed()
    {
        return transmission.CalculateTheoreticalSpeed(EngineRpm);
    }

    public float GetWheelSlip(float actualVx)
    {
        float theoreticalSpeed = GetTheoreticalSpeed();
        if (theoreticalSpeed < 0.1f) return 0f;
        return ((theoreticalSpeed - actualVx) / theoreticalSpeed) * 100f;
    }

    public string GetGearSpeedRange(int gear)
    {
        return transmission.GetGearSpeedRange(gear);
    }

    // ✅ NEW: Additional transmission access methods
    public Transmission GetTransmission()
    {
        return transmission;
    }

    public string GetGearDebugInfo(float vx)
    {
        return transmission.GetGearDebugInfo(EngineRpm, vx);
    }
}