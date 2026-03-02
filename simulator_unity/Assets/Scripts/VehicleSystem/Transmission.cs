using UnityEngine;

/// <summary>
/// Advanced automatic transmission system for Audi TT 2.0 TFSI
/// Includes logic to prevent gear hunting and smooth transitions
/// </summary>
public class Transmission
{
    private VehicleParameters vehicleParams;

    [Header("Audi TT 2.0 TFSI - Automatic Transmission")]
    // Real gear ratios from Audi TT specifications  
    private float[] gearRatios = { 3.769f, 2.087f, 1.481f, 1.152f, 1.167f, 0.970f };
    private float[] finalDriveRatios = { 3.238f, 3.238f, 3.238f, 3.238f, 2.615f, 2.615f };

    // ✅ BALANCED: Reasonable shift points - not too late, not too early
    private float[] upshiftRPM = { 2800f, 3200f, 3600f, 4000f, 4400f }; // Balanced shift points
    private float[] downshiftRPM = { 1400f, 1600f, 1800f, 2000f, 2200f }; // Reasonable downshifts

    // ✅ REALISTIC: Speed-based shift points for proper gear usage
    private float[] upshiftSpeed = { 25f, 45f, 70f, 100f, 130f }; // km/h - realistic ranges
    private float[] downshiftSpeed = { 12f, 25f, 45f, 70f, 100f }; // km/h - reasonable downshift

    // Transmission state
    private int currentGear = 0; // Start in 1st gear (index 0)
    private bool isShifting = false;
    private float shiftTimer = 0f;
    private float shiftDuration = 0.2f; // 200ms shift time

    // ✅ BALANCED: Reasonable cooldown to prevent hunting
    private float shiftCooldown = 0f;
    private float minShiftInterval = 0.3f; // 300ms between shifts
    private int lastShiftDirection = 0; // 1 = up, -1 = down, 0 = none

    // ✅ MODERATE: Driving mode parameters
    public enum DriveMode { Eco, Comfort, Sport }
    private DriveMode currentDriveMode = DriveMode.Comfort;
    private float sportModeRpmBonus = 400f; // Sport mode holds gears longer
    private float ecoModeRpmReduction = 300f; // Eco mode shifts earlier

    public Transmission(VehicleParameters vehicleParams)
    {
        this.vehicleParams = vehicleParams;
        currentGear = 0; // Start in 1st gear
    }

    public void Update(float deltaTime, float engineRpm, float vx, float throttle)
    {
        // Update timers
        if (isShifting)
        {
            shiftTimer += deltaTime;
            if (shiftTimer >= shiftDuration)
            {
                isShifting = false;
                shiftTimer = 0f;
            }
        }

        if (shiftCooldown > 0f)
        {
            shiftCooldown -= deltaTime;
        }

        // Check for shifts when not shifting and cooldown expired
        if (!isShifting && shiftCooldown <= 0f)
        {
            CheckForGearChange(engineRpm, vx, throttle);
        }
    }

    private void CheckForGearChange(float engineRpm, float vx, float throttle)
    {
        float currentSpeedKmh = vx * 3.6f;
        bool shouldShift = false;
        int newGear = currentGear;

        // ✅ BALANCED UPSHIFT: Consider both RPM and speed, with some priority to RPM
        if (currentGear < gearRatios.Length - 1) // Not in top gear
        {
            float upshiftRpmThreshold = GetAdjustedUpshiftRpm(currentGear, throttle);
            float upshiftSpeedThreshold = upshiftSpeed[currentGear];

            // ✅ PRIMARY: RPM-based shifting with speed as secondary consideration
            bool rpmCondition = engineRpm > upshiftRpmThreshold;
            bool speedCondition = currentSpeedKmh > upshiftSpeedThreshold && throttle > 0.1f;
            bool emergencyCondition = engineRpm > (upshiftRpmThreshold + 500f); // Emergency at very high RPM

            if (rpmCondition || speedCondition || emergencyCondition)
            {
                newGear = currentGear + 1;
                shouldShift = true;
                lastShiftDirection = 1;
                string reason = emergencyCondition ? "EMERGENCY" : (rpmCondition ? "RPM" : "SPEED");
                Debug.Log($"🔺 UPSHIFT ({reason}): {currentGear + 1} → {newGear + 1} | RPM: {engineRpm:F0} | Speed: {currentSpeedKmh:F1} km/h");
            }
        }

        // ✅ SMART DOWNSHIFT: Prevent unnecessary downshifts
        if (currentGear > 0 && !shouldShift) // Not in 1st gear and not already shifting up
        {
            float downshiftRpmThreshold = GetAdjustedDownshiftRpm(currentGear, throttle);
            float downshiftSpeedThreshold = downshiftSpeed[currentGear - 1];

            // ✅ KICKDOWN: Aggressive downshift with high throttle
            bool kickdown = (throttle > 0.8f && engineRpm < 3000f);
            bool lowRpm = engineRpm < downshiftRpmThreshold;
            bool lowSpeed = currentSpeedKmh < downshiftSpeedThreshold;

            if (lowRpm || lowSpeed || kickdown)
            {
                newGear = currentGear - 1;
                shouldShift = true;
                lastShiftDirection = -1;
                string reason = kickdown ? "KICKDOWN" : (lowRpm ? "LOW_RPM" : "LOW_SPEED");
                Debug.Log($"🔻 DOWNSHIFT ({reason}): {currentGear + 1} → {newGear + 1} | RPM: {engineRpm:F0} | Speed: {currentSpeedKmh:F1} km/h");
            }
        }

        // Execute the shift
        if (shouldShift && newGear != currentGear)
        {
            currentGear = newGear;
            isShifting = true;
            shiftTimer = 0f;
            shiftCooldown = minShiftInterval;
        }
    }

    // ✅ BALANCED: Reasonable throttle influence on shift points
    private float GetAdjustedUpshiftRpm(int gear, float throttle)
    {
        float baseRpm = upshiftRPM[gear];

        switch (currentDriveMode)
        {
            case DriveMode.Sport:
                return baseRpm + sportModeRpmBonus;
            case DriveMode.Eco:
                return baseRpm - ecoModeRpmReduction;
            case DriveMode.Comfort:
            default:
                // ✅ MODERATE throttle influence - more throttle = later shift
                return baseRpm + (throttle * 300f);
        }
    }

    private float GetAdjustedDownshiftRpm(int gear, float throttle)
    {
        float baseRpm = downshiftRPM[gear - 1];

        switch (currentDriveMode)
        {
            case DriveMode.Sport:
                return baseRpm + 150f; // Hold gears longer in sport mode
            case DriveMode.Eco:
                return baseRpm - 100f; // Shift down earlier in eco mode
            case DriveMode.Comfort:
            default:
                return baseRpm;
        }
    }

    // Calculate required engine RPM for current speed and gear
    public float CalculateRequiredEngineRPM(float vx)
    {
        if (Mathf.Abs(vx) < 0.01f) return 800f; // Idle RPM

        float wheelRpm = (Mathf.Abs(vx) * 60f) / (2f * Mathf.PI * vehicleParams.wheelRadius);
        float totalGearReduction = gearRatios[currentGear] * finalDriveRatios[currentGear];
        float requiredRPM = wheelRpm * totalGearReduction;

        return Mathf.Max(requiredRPM, 800f); // Minimum idle RPM
    }

    // Calculate theoretical speed for current engine RPM and gear
    public float CalculateTheoreticalSpeed(float engineRpm)
    {
        if (engineRpm < 800f) return 0f;

        float totalGearReduction = gearRatios[currentGear] * finalDriveRatios[currentGear];
        float wheelRpm = engineRpm / totalGearReduction;
        float wheelAngularVelocity = (wheelRpm * 2f * Mathf.PI) / 60f;
        return wheelAngularVelocity * vehicleParams.wheelRadius;
    }

    // ✅ REALISTIC: Speed ranges with balanced shift points
    public string GetGearSpeedRange(int gear)
    {
        if (gear < 1 || gear > gearRatios.Length) return "Invalid";

        int gearIndex = gear - 1;

        // Use realistic RPM range (1000-5000 RPM for practical driving)
        float minRpm = 1000f; // Practical minimum
        float maxRpm = gearIndex < upshiftRPM.Length ? upshiftRPM[gearIndex] + 200f : 5000f; // Shift point + margin

        float totalGearReduction = gearRatios[gearIndex] * finalDriveRatios[gearIndex];

        float minWheelRpm = minRpm / totalGearReduction;
        float maxWheelRpm = maxRpm / totalGearReduction;

        float minSpeed = (minWheelRpm * 2f * Mathf.PI * vehicleParams.wheelRadius * 60f) / 60f * 3.6f;
        float maxSpeed = (maxWheelRpm * 2f * Mathf.PI * vehicleParams.wheelRadius * 60f) / 60f * 3.6f;

        return $"{minSpeed:F0}-{maxSpeed:F0} km/h";
    }

    // Get optimal speed for cruising in each gear (at 2500 RPM)
    public float GetOptimalCruisingSpeed(int gear)
    {
        if (gear < 1 || gear > gearRatios.Length) return 0f;

        int gearIndex = gear - 1;
        float cruisingRpm = 2500f; // Good cruising RPM

        float totalGearReduction = gearRatios[gearIndex] * finalDriveRatios[gearIndex];
        float wheelRpm = cruisingRpm / totalGearReduction;
        float wheelAngularVelocity = (wheelRpm * 2f * Mathf.PI) / 60f;

        return wheelAngularVelocity * vehicleParams.wheelRadius;
    }

    // Public getters
    public int GetCurrentGear() => currentGear + 1; // Return 1-based gear number
    public bool IsShifting() => isShifting;
    public float GetShiftProgress() => isShifting ? (shiftTimer / shiftDuration) : 0f;
    public float[] GetGearRatios() => gearRatios;
    public float[] GetFinalDriveRatios() => finalDriveRatios;
    public DriveMode GetDriveMode() => currentDriveMode;

    // Setters
    public void SetDriveMode(DriveMode mode) => currentDriveMode = mode;

    // Force manual gear change (for future manual mode)
    public bool ForceGearChange(int newGear)
    {
        newGear--; // Convert to 0-based index
        if (newGear >= 0 && newGear < gearRatios.Length && !isShifting)
        {
            currentGear = newGear;
            isShifting = true;
            shiftTimer = 0f;
            return true;
        }
        return false;
    }

    // Calculate current gear reduction
    public float GetCurrentGearReduction()
    {
        return gearRatios[currentGear] * finalDriveRatios[currentGear];
    }

    // Get gear information for debugging
    public string GetGearDebugInfo(float engineRpm, float vx)
    {
        float currentSpeedKmh = vx * 3.6f;
        float theoreticalSpeed = CalculateTheoreticalSpeed(engineRpm) * 3.6f;
        float wheelSlip = theoreticalSpeed > 0.1f ? ((theoreticalSpeed - currentSpeedKmh) / theoreticalSpeed) * 100f : 0f;

        return $"Gear {GetCurrentGear()}: {currentSpeedKmh:F1}/{theoreticalSpeed:F1} km/h, Slip: {wheelSlip:F1}%";
    }
}