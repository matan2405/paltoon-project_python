using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PlatoonManager : MonoBehaviour
{
    [Header("Platoon Vehicles")]
    // This array holds the vehicles that are part of the platoon.
    public AdvancedBicycleModel[] platoonVehicles;
    public AdvancedBicycleModel[] GetPlatoonVehicles() => platoonVehicles;

    [Header("Platoon Parameters")]
    // These parameters define the behavior of the platoon.
    public float platoonMaxVelocity    = 250 / 3.6f;
    public float platoonTargetVelocity = 50 / 3.6f; // Target speed 50 km/h converted to m/s
    public float MaxAcceleration       = 2.5f;
    public float PlatoonTimeHeadway    = 1.5f;  // [s] — single source of truth; pushed to Nash configs on Start
    public float PlatoonStandstillDist = 2.0f;  // [m] — single source of truth; pushed to Nash configs on Start

    // Both read directly from Inspector fields — single source of truth, no Nash dependency
    public float TimeGap        => PlatoonTimeHeadway;
    public float StandstillDist => PlatoonStandstillDist;
    
    [Header("Control Inputs")]
    private float[] ThrottleInput;
    private float[] SteeringInput;
    private float[] BrakeInput;

    [Header("Data Logging")]
    private List<float>[] gap;
    private List<float>[] des_gap;
    
    [Header("Auto Setup")]
    public bool autoFindVehicles = true; // Automatically find vehicles in scene
    public bool startFromZeroSpeed = true; // Start all vehicles from zero speed
    private float[] acc;
    private float[] vx;
    private float[] _prevAcc;
    private const float JerkMax = 2.0f; // ISO 15622 [m/s³]

    [Header("Ego Vehicle (Join Trigger)")]
    public AdvancedBicycleModel egoVehicle;   // Assign in Inspector — starts outside platoon
    public KeyCode joinKey  = KeyCode.J;      // Press to join platoon at current position
    public KeyCode leaveKey = KeyCode.L;      // Press to leave platoon (returns to manual)
    public int joinWheelButton  = 0;          // Steering wheel button index to join (G29: 0=X/Cross)
    public int leaveWheelButton = 2;          // Steering wheel button index to leave (G29: 2=Square)
    public bool _egoInPlatoon = false;        // public so NashCoordinator/HUD can read it

    [Header("Nash")]
    public NashCoordinator nashCoordinator;   // Assign in Inspector — activated on J

    void PushToNashConfigs()
    {
        if (nashCoordinator == null) return;
        nashCoordinator.Settings.SafetyField.PlatoonTimeGap = PlatoonTimeHeadway;
        nashCoordinator.Settings.SafetyField.StandstillDist = PlatoonStandstillDist;
        nashCoordinator.Settings.RefGen.RajamaniH           = PlatoonTimeHeadway;
    }

    void OnValidate() => PushToNashConfigs();

    void Start()
    {
        PushToNashConfigs();
        SetupPlatoon();
        InitializeDataLogging();
        InitializeVehicleStates();
    }

    void Update()
    {
        if (egoVehicle == null) return;
        bool joinPressed  = Input.GetKeyDown(joinKey)  || Input.GetKeyDown("joystick button " + joinWheelButton);
        bool leavePressed = Input.GetKeyDown(leaveKey) || Input.GetKeyDown("joystick button " + leaveWheelButton);
        if (!_egoInPlatoon && joinPressed)
            nashCoordinator?.EnableNash();
        if (_egoInPlatoon && leavePressed)
            LeavePlatoon();
    }
    
    void SetupPlatoon()
    {
        // Auto-find vehicles if array is empty, excluding the ego vehicle
        if (autoFindVehicles && (platoonVehicles == null || platoonVehicles.Length == 0))
        {
            AdvancedBicycleModel[] foundVehicles = FindObjectsByType<AdvancedBicycleModel>(FindObjectsSortMode.None)
                .Where(v => v != egoVehicle).ToArray();
            platoonVehicles = foundVehicles;
            Debug.Log($"Auto-found {foundVehicles.Length} vehicles in scene");
        }
        
        if (platoonVehicles == null || platoonVehicles.Length == 0)
        {
            Debug.LogError("No vehicles found! Please assign vehicles to the platoonVehicles array.");
            return;
        }
        
        // Sort vehicles by position (front to back) only when ego is NOT in the array.
        // When ego is already inserted (via JoinPlatoon), the order is set by the Nash
        // lock and must not be overridden by Z — ego may be slightly behind its locked
        // follower during the lane-change approach.
        bool egoPresent = egoVehicle != null &&
                          System.Array.IndexOf(platoonVehicles, egoVehicle) >= 0;
        if (!egoPresent)
            System.Array.Sort(platoonVehicles, (a, b) => b.GetPosition().z.CompareTo(a.GetPosition().z));
        
        // Initialize input arrays
        int numVehicles = platoonVehicles.Length;
        ThrottleInput = new float[numVehicles];
        SteeringInput = new float[numVehicles];
        BrakeInput = new float[numVehicles];
        acc = new float[numVehicles];
        vx = new float[numVehicles];
        _prevAcc = new float[numVehicles];

        // Configure vehicles for platoon control
        for (int i = 0; i < platoonVehicles.Length; i++)
        {
            var vehicle = platoonVehicles[i];
            vehicle.name = $"Vehicle_{i + 1}";
            vehicle.AutoMode = true;
            vehicle.SetIndexPlatoon(i);
            vx[i] = vehicle.GetVx();
            acc[i] = vehicle.GetAx();
            //Debug.Log($"Configured {vehicle.name} at position {vehicle.GetPosition()}");
        }
        
        //Debug.Log($"Platoon setup complete with {numVehicles} vehicles");
        //Debug.Log($"Target speed: {platoonTargetVelocity * 3.6f:F1} km/h");
    }
    
    void InitializeDataLogging()
    {
        int N = platoonVehicles.Length;
        gap = new List<float>[N - 1];
        des_gap = new List<float>[N - 1];
        for (int i = 0; i < N - 1; i++)
        {
            gap[i] = new List<float>();
            des_gap[i] = new List<float>();
        }
    }
    
    void InitializeVehicleStates()
    {
        if (!startFromZeroSpeed) return;
        
        // Set all vehicles to start from zero speed
        foreach (var vehicle in platoonVehicles)
        {
            vehicle.SetVx(0f);
            vehicle.SetAx(0f);
        }
        
        Debug.Log("All vehicles initialized to zero speed");
    }

    public void AddVehicleToPlatoon(AdvancedBicycleModel vehicle)
    {
        if (platoonVehicles == null)
        {
            platoonVehicles = new AdvancedBicycleModel[] { vehicle };
        }
        else
        {
            var tempList = new System.Collections.Generic.List<AdvancedBicycleModel>(platoonVehicles);
            tempList.Add(vehicle);
            platoonVehicles = tempList.ToArray();
        }
        
        // Reconfigure arrays
        SetupPlatoon();
        InitializeDataLogging();
    }
    
    public (float a_des, float s_des) CalculateRajamaniAcceleration(AdvancedBicycleModel Car_1, AdvancedBicycleModel Car_2)
    {
        float h = TimeGap;

        float k1 = -0.12f, k5 = 0.1f;
        float k2 = -k1 - h * k1 * k5;
        float k3 = 1f / h - k1 * k5;
        float k4 = k5 / h;

        // e defined so equilibrium at e=-(h*v) gives bumper-to-bumper = StandstillDist + h*v,
        // matching NashCoordinator exactly and supporting vehicles of different lengths.
        float halfLengths = Car_1.GetVehicleParameters().length / 2f + Car_2.GetVehicleParameters().length / 2f;
        float e     = Car_2.GetPosition().z - Car_1.GetPosition().z + halfLengths + StandstillDist;
        float e_dot = Car_2.GetVx() - Car_1.GetVx();

        float s_des = StandstillDist + h * Car_2.GetVx();
        float a_des = -k1 * Car_1.GetAx()
                      - k2 * Car_2.GetAx()
                      - k3 * e_dot
                      - k4 * e
                      - k5 * Car_2.GetVx();

        return (a_des, s_des);
    }
    
    public static float FreeRoadAcceleration(float v, float v_target, float a_max, float a_min)
    {
        float delta = 4f; // exponent

        float dv_dt;
        if (v_target >= v)
        {
            dv_dt = a_max * (1f - Mathf.Pow(v / v_target, delta));
        }
        else
        {
            dv_dt = -a_min * (1f - Mathf.Pow(v_target / v, delta));
        }
        return dv_dt;
    }
    
    float JerkLimit(int idx, float aDes) {
        float dt    = Time.fixedDeltaTime;
        float maxDA = JerkMax * dt;
        float limited = Mathf.Clamp(aDes, _prevAcc[idx] - maxDA, _prevAcc[idx] + maxDA);
        _prevAcc[idx] = limited;
        return limited;
    }

    void JoinPlatoon()
    {
        // Fallback: called from Update (manual J press without Nash lock).
        // Insert ego before the first vehicle it is ahead of or parallel to.
        int insertIdx = platoonVehicles.Length;
        for (int i = 0; i < platoonVehicles.Length; i++)
        {
            float dz      = egoVehicle.GetPosition().z - platoonVehicles[i].GetPosition().z;
            float halfSum = egoVehicle.GetVehicleParameters().length * 0.5f
                          + platoonVehicles[i].GetVehicleParameters().length * 0.5f;
            if (Mathf.Abs(dz) <= halfSum || dz > 0f) { insertIdx = i; break; }
        }
        JoinPlatoonAtIndex(insertIdx);
    }

    public void JoinPlatoon(AdvancedBicycleModel lockedLeader, AdvancedBicycleModel lockedFollower)
    {
        // Insert ego between lockedLeader and lockedFollower — exact position determined
        // by the Nash lock at GapSearch entry, not by current Z (which may have drifted).
        int insertIdx = platoonVehicles.Length;
        if (lockedFollower != null)
        {
            for (int i = 0; i < platoonVehicles.Length; i++)
            {
                if (platoonVehicles[i] == lockedFollower) { insertIdx = i; break; }
            }
        }
        else if (lockedLeader != null)
        {
            // No follower: insert immediately after the leader
            for (int i = 0; i < platoonVehicles.Length; i++)
            {
                if (platoonVehicles[i] == lockedLeader) { insertIdx = i + 1; break; }
            }
        }
        JoinPlatoonAtIndex(insertIdx);
    }

    void JoinPlatoonAtIndex(int insertIdx)
    {
        // Save current _prevAcc so existing vehicles don't get a jerk spike on re-init
        float[] savedPrevAcc = _prevAcc != null ? (float[])_prevAcc.Clone() : null;

        var list = new List<AdvancedBicycleModel>(platoonVehicles);
        list.Insert(insertIdx, egoVehicle);
        platoonVehicles = list.ToArray();

        // Re-init platoon state (sorts by Z, rebuilds arrays, sets AutoMode=true for all).
        // NOTE: SetupPlatoon re-sorts by Z — so insertIdx is only a hint; the sort
        // corrects any minor positional drift. The locked-follower path above guarantees
        // we pass the right index before the sort runs.
        SetupPlatoon();
        InitializeDataLogging();

        // Restore _prevAcc for vehicles that were already in the platoon
        if (savedPrevAcc != null)
        {
            for (int newIdx = 0; newIdx < platoonVehicles.Length; newIdx++)
            {
                if (newIdx < insertIdx)
                    _prevAcc[newIdx] = savedPrevAcc[newIdx];
                else if (newIdx == insertIdx)
                    _prevAcc[newIdx] = egoVehicle.GetAx();
                else
                    _prevAcc[newIdx] = savedPrevAcc[newIdx - 1];
            }
        }

        _egoInPlatoon = true;
        Debug.Log($"Ego joined platoon at index {insertIdx} " +
                  $"(Z={egoVehicle.GetPosition().z:F1}m), platoon size={platoonVehicles.Length}");
    }

    void LeavePlatoon()
    {
        nashCoordinator?.DisableNash();

        // Find ego's current index BEFORE removing (needed for _prevAcc mapping)
        int egoIdx = System.Array.IndexOf(platoonVehicles, egoVehicle);
        float[] savedPrevAcc = _prevAcc != null ? (float[])_prevAcc.Clone() : null;

        var list = new List<AdvancedBicycleModel>(platoonVehicles);
        list.Remove(egoVehicle);
        platoonVehicles = list.ToArray();

        SetupPlatoon();
        InitializeDataLogging();

        // Restore _prevAcc: skip the ego's old slot
        if (savedPrevAcc != null && egoIdx >= 0)
        {
            for (int newIdx = 0; newIdx < platoonVehicles.Length; newIdx++)
            {
                int oldIdx = newIdx < egoIdx ? newIdx : newIdx + 1;
                if (oldIdx < savedPrevAcc.Length)
                    _prevAcc[newIdx] = savedPrevAcc[oldIdx];
            }
        }

        // Return ego to manual control
        egoVehicle.AutoMode       = false;
        egoVehicle.NashModeActive = false;

        _egoInPlatoon = false;
        Debug.Log($"Ego left platoon. Remaining platoon size={platoonVehicles.Length}");
    }

    void FixedUpdate()
    {
        if (platoonVehicles == null || platoonVehicles.Length == 0) return;

        // Leader: free-road acceleration
        AdvancedBicycleModel LeaderVehicle = platoonVehicles[0];
        LeaderVehicle.GetPhysicalAccelBounds(out float leaderAMin, out float leaderAMax);
        float acc_leader = FreeRoadAcceleration(LeaderVehicle.GetVx(), platoonTargetVelocity, leaderAMax, -leaderAMin);
        acc_leader = JerkLimit(0, Mathf.Clamp(acc_leader, leaderAMin, leaderAMax));
        acc[0] = acc_leader;
        vx[0] = Mathf.Clamp(LeaderVehicle.GetVx() + acc_leader * Time.fixedDeltaTime, 0f, platoonMaxVelocity);
        if (!LeaderVehicle.NashModeActive) {
            var (lt, lb) = LeaderVehicle.AccelerationToInputs(acc_leader);
            LeaderVehicle.NashThrottle = lt;
            LeaderVehicle.NashBrake    = lb;
        }

        // Followers: Rajamani
        for (int car_num = 1; car_num < platoonVehicles.Length; car_num++)
        {
            AdvancedBicycleModel Car_ahead = platoonVehicles[car_num - 1];
            AdvancedBicycleModel Car       = platoonVehicles[car_num];

            float actual_gap = Car_ahead.GetPosition().z - Car.GetPosition().z;
            gap[car_num - 1].Add(actual_gap);

            (float a_des, float s_des) = CalculateRajamaniAcceleration(Car_ahead, Car);
            des_gap[car_num - 1].Add(s_des);

            Car.GetPhysicalAccelBounds(out float carAMin, out float carAMax);
            float acc_ = JerkLimit(car_num, Mathf.Clamp(a_des, carAMin, carAMax));
            acc[car_num] = acc_;
            vx[car_num]  = Mathf.Clamp(Car.GetVx() + acc_ * Time.fixedDeltaTime, 0f, platoonMaxVelocity);

            // Inject into NashThrottle/NashBrake only if NashCoordinator isn't controlling this vehicle
            if (!Car.NashModeActive) {
                var (t, b) = Car.AccelerationToInputs(acc_);
                Car.NashThrottle = t;
                Car.NashBrake    = b;
            }
        }
        if (Time.fixedTime % 2f < Time.fixedDeltaTime) LogPlatoonStatus();
        //Debug.Log($"des_gap1: {des_gap[1 - 1].Last()} , actual_gap1: {gap[1 - 1].Last()} recorded.");
        //Debug.Log($"des_gap1_Get: {GetDesiredGap(1)} , actual_gap2: {GetActualGap(1)} recorded.");
        //Debug.Log($"des_gap2: {des_gap[2-1].Last()} , actual_gap2: {gap[2-1].Last()} recorded.");
        //Debug.Log($"des_gap2_Get: {GetDesiredGap(2)} , actual_gap2: {GetActualGap(2)} recorded.");

    }
    
    public float GetThrottleInput_Pl(int car_num) => car_num < ThrottleInput.Length ? ThrottleInput[car_num] : 0f;
    public float GetBrakeInput_Pl(int car_num) => car_num < BrakeInput.Length ? BrakeInput[car_num] : 0f;
    public float GetSteeringInput_Pl(int car_num) => car_num < SteeringInput.Length ? SteeringInput[car_num] : 0f;

    public float GetActualGap(int car_num)
    {
        if (car_num < 1 || car_num > gap.Length || gap[car_num - 1].Count == 0) return 0f;
        return gap[car_num - 1].Count > 0 ? gap[car_num - 1].Last() : 0f;
    }
    public float GetDesiredGap(int car_num)
    {
        if (car_num < 1 || car_num > des_gap.Length || des_gap[car_num - 1].Count == 0 ) return 0f;
        //Debug.Log($"GetDesiredGap called for car_num: {car_num}, des_gap value: {des_gap[car_num - 1][des_gap[car_num - 1].Count - 1]}");
        // Return the last recorded desired gap for the specified vehicle
        return des_gap[car_num - 1].Count > 0 ? des_gap[car_num - 1].Last() : 0f;
    }
    public float GetTargetVelocity()
    {
         return platoonTargetVelocity;
    }

    public float GetMaxAcc()
    {
         return MaxAcceleration;
    }
    public float GetVelocityVehicle(int car_num)
    {
         return vx[car_num];
    }
    public float GetAccelerationVehicle(int car_num)
    {
         return acc[car_num];
    }

    // Debug methods
    [ContextMenu("Log Platoon Status")]
    public void LogPlatoonStatus()
    {
        if (platoonVehicles == null || platoonVehicles.Length == 0) return;
        
        Debug.Log("=== Platoon Status ===");
        Debug.Log($"Target Speed: {platoonTargetVelocity * 3.6f:F1} km/h");
        Debug.Log($"Lead Vehicle Speed: {platoonVehicles[0].GetVx() * 3.6f:F1} km/h");
        
        for (int i = 1; i < platoonVehicles.Length; i++)
        {
            float actualGap = platoonVehicles[i - 1].GetPosition().z - platoonVehicles[i].GetPosition().z;
            float desiredGap = des_gap[i - 1].Count > 0 ? des_gap[i - 1][des_gap[i - 1].Count - 1] : 0f;
            float gapError = actualGap - desiredGap;
            float speedKmh = platoonVehicles[i].GetVx() * 3.6f;
            
            Debug.Log($"Vehicle {i + 1}: Speed={speedKmh:F1}km/h, Gap={actualGap:F1}m, Desired={desiredGap:F1}m, Error={gapError:F1}m");
        }
    }
    
}