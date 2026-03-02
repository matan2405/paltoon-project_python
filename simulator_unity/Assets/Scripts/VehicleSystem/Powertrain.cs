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

    // Accurate RPM points from the graphs
    private const float torqueRiseStartRPM = 1200f;
    private const float torqueRiseEndRPM = 1600f;
    private const float torquePlateauEndRPM = 4300f;
    private const float powerPeakStartRPM = 4500f;
    private const float powerPeakEndRPM = 6200f;

    // Engine response characteristics
    private float engineRpmAcceleration = 2500f; // FASTER RPM increases
    private float engineRpmDeceleration = 2000f; // FASTER RPM decreases
    private float engineInertia = 0.1f;          // LESS inertia

    private float angularVelocity;

    // ✅ MUCH STRONGER: Engine braking for realistic deceleration
    private const float engineBrakingCoefficient = 0.8f; // Increased significantly from 0.5f
    private const float compressionBrakingForce = 400f;   // DOUBLED from 200f

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
            return Mathf.Lerp(180f, 240f, (rpm - idleRPM) / (torqueRiseStartRPM - idleRPM));
        }
        else if (rpm <= torqueRiseEndRPM)
        {
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

    // ✅ MUCH STRONGER: Engine braking that provides realistic deceleration
    private float CalculateEngineBrakingForce(float currentRpm, float vx)
    {
        if (engineState != 2 || Mathf.Abs(vx) < 0.1f) return 0f;

        float rpmFactor = Mathf.Clamp01((currentRpm - idleRPM) / (redlineRPM - idleRPM));
        float speedFactor = Mathf.Clamp01(Mathf.Abs(vx) / 35f);

        // ✅ MUCH STRONGER: Realistic engine braking forces
        float compressionBraking = compressionBrakingForce * rpmFactor * speedFactor;
        float internalFriction = engineBrakingCoefficient * currentRpm * 0.25f; // Doubled friction
        float turboEffect = 60f * rpmFactor * speedFactor; // Doubled turbo effect

        // ✅ STRONGER: Gear-based multiplier with more effect
        float gearMultiplier = transmission.GetCurrentGearReduction() / 6f; // More braking effect (was /8f)

        // ✅ SPEED-BASED: More braking at higher speeds
        float speedMultiplier = 1f + (speedFactor * 0.8f); // Increased from 0.5f

        // ✅ NEW: RPM-based multiplier - higher RPM = more braking
        float rpmMultiplier = 1f + (rpmFactor * 1.2f);

        float totalBraking = (compressionBraking + internalFriction + turboEffect) * gearMultiplier * speedMultiplier * rpmMultiplier;

        return totalBraking;
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

            // Apply throttle scaling
            return engineForce * throttle;
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
    private float CalculateMaxForceAtCurrentSpeed(float vx)
    {
        // Calculate maximum force in current gear
        float maxTorque = 370f; // Maximum engine torque
        float totalGearReduction = transmission.GetCurrentGearReduction();
        float maxTorqueAtWheels = maxTorque * totalGearReduction;
        float maxForce = maxTorqueAtWheels / vehicleParams.wheelRadius;
        
        return maxForce;
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