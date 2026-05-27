using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PlatoonManager : MonoBehaviour
{
    [Header("Platoon Vehicles")]
    public AdvancedBicycleModel[] platoonVehicles;
    public AdvancedBicycleModel[] GetPlatoonVehicles() => platoonVehicles;

    [Header("Platoon Parameters")]
    public float platoonMaxVelocity = 250 / 3.6f;
    public float platoonTargetVelocity = 50 / 3.6f; // Target speed 50 km/h converted to m/s
    public float MaxAcceleration = 2.5f; // Maximum acceleration for the platoon vehicles in m/s²
    public float headway = 1.5f; // Desired time headway in seconds

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
    public bool _egoInPlatoon = false;        // public so NashCoordinator/HUD can read it

    void Start()
    {
        SetupPlatoon();
        InitializeDataLogging();
        InitializeVehicleStates();
    }

    void Update()
    {
        if (egoVehicle == null) return;
        if (!_egoInPlatoon && Input.GetKeyDown(joinKey))
            JoinPlatoon();
        else if (_egoInPlatoon && Input.GetKeyDown(leaveKey))
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

        // Sort vehicles by position (front to back) - assuming they're aligned on Z axis
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
        }
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

        SetupPlatoon();
        InitializeDataLogging();
    }

    public (float a_des, float s_des) CalculateRajamaniAcceleration(AdvancedBicycleModel Car_1, AdvancedBicycleModel Car_2)
    {
        float h = headway;

        float k1 = -0.12f, k5 = 0.1f; // k1 < -tau/h, k5 > 0
        float k2 = -k1 - h * k1 * k5;
        float k3 = 1f / h - k1 * k5;
        float k4 = k5 / h;

        float e = Car_2.GetPosition().z - Car_1.GetPosition().z + Car_1.GetVehicleParameters().length + 2f; // [m] actual gap
        float e_dot = Car_2.GetVx() - Car_1.GetVx(); // [m/s] relative velocity

        float s_des = Car_1.GetVehicleParameters().length + 2f + h * Car_2.GetVx(); // [m] desired gap
        float a_des = -k1 * Car_1.GetAx()
                      - k2 * Car_2.GetAx()
                      - k3 * e_dot
                      - k4 * e
                      - k5 * Car_2.GetVx();

        return (a_des, s_des);
    }

    public static float FreeRoadAcceleration(float v, float v_target, float a_max)
    {
        float delta = 4f;

        float dv_dt;
        if (v_target >= v)
        {
            dv_dt = a_max * (1f - Mathf.Pow(v / v_target, delta));
        }
        else
        {
            dv_dt = -a_max * (1f - Mathf.Pow(v_target / v, delta));
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
        float[] savedPrevAcc = _prevAcc != null ? (float[])_prevAcc.Clone() : null;

        // Insert ego at the first position where ego is ahead of the existing vehicle (descending Z order)
        int insertIdx = platoonVehicles.Length;
        for (int i = 0; i < platoonVehicles.Length; i++)
        {
            if (egoVehicle.GetPosition().z > platoonVehicles[i].GetPosition().z)
            {
                insertIdx = i;
                break;
            }
        }

        var list = new List<AdvancedBicycleModel>(platoonVehicles);
        list.Insert(insertIdx, egoVehicle);
        platoonVehicles = list.ToArray();

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
        float acc_leader = FreeRoadAcceleration(LeaderVehicle.GetVx(), platoonTargetVelocity, MaxAcceleration);
        acc_leader = JerkLimit(0, Mathf.Clamp(acc_leader, -MaxAcceleration, MaxAcceleration));
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

            float acc_ = JerkLimit(car_num, Mathf.Clamp(a_des, -MaxAcceleration, MaxAcceleration));
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
    }

    public float GetThrottleInput_Pl(int car_num) => car_num < ThrottleInput.Length ? ThrottleInput[car_num] : 0f;
    public float GetBrakeInput_Pl(int car_num) => car_num < BrakeInput.Length ? BrakeInput[car_num] : 0f;
    public float GetSteeringInput_Pl(int car_num) => car_num < SteeringInput.Length ? SteeringInput[car_num] : 0f;

    public float GetActualGap(int car_num)
    {
        if (car_num < 1 || car_num > gap.Length || gap[car_num - 1].Count == 0) return 0f;
        return gap[car_num - 1].Last();
    }
    public float GetDesiredGap(int car_num)
    {
        if (car_num < 1 || car_num > des_gap.Length || des_gap[car_num - 1].Count == 0) return 0f;
        return des_gap[car_num - 1].Last();
    }
    public float GetTargetVelocity() => platoonTargetVelocity;
    public float GetMaxAcc() => MaxAcceleration;
    public float GetVelocityVehicle(int car_num) => vx[car_num];
    public float GetAccelerationVehicle(int car_num) => acc[car_num];

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
