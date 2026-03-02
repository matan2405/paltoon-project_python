// using System.Collections.Generic;
// using System.Linq;
// using UnityEngine;

// public class AccMerge : MonoBehaviour
// {
//     [Header("Platoon Vehicles")]
//     // This array holds the vehicles that are part of the platoon.
//     public AdvancedBicycleModel[] platoonVehicles;
//     private PlatoonManager platoonManager;

//     [Header("Platoon Parameters")]
//     // These parameters define the behavior of the platoon.
//     public float platoonMaxVelocity = 250 / 3.6f;
//     public float platoonTargetVelocity = 50 / 3.6f; // Target speed 50 km/h converted to m/s
//     public float MaxAcceleration = 2.5f; // Maximum acceleration for the platoon vehicles in m/s²
//     public float headway = 1.5f; // Desired time headway in seconds
    
//     [Header("Control Inputs")]
//     private float[] ThrottleInput;
//     private float[] SteeringInput;
//     private float[] BrakeInput;

//     [Header("Data Logging")]
//     private List<float>[] gap;
//     private List<float>[] des_gap;
    
//     [Header("Auto Setup")]
//     public bool autoFindVehicles = true; // Automatically find vehicles in scene
//     public bool startFromZeroSpeed = true; // Start all vehicles from zero speed
//     private float[] acc;
//     private float[] vx;
//     void Start()
//     {
//         platoonManager = GetComponent<PlatoonManager>();
//         if (platoonManager == null)
//         {
//             Debug.LogError("PlatoonManager component not found! Please attach this script to a GameObject with a PlatoonManager.");
//             return;
//         }
//         platoonVehicles = platoonManager.platoonVehicles;

//     }
    
//     public float CalculateTTE(AdvancedBicycleModel mergingVehicle, AdvancedBicycleModel egoVehicle)
//     {
//         // Calculate Time to Enter (TTE) for merging vehicle and ego vehicle
//         Vector3 mergingPos = PredictTrajectory(mergingVehicle);
//         Vector3 egoPos = PredictTrajectory(egoVehicle);
        
//         // בדיקה אם המסלולים מתחככים
//         return CalculateCollisionTime(mergingPos, egoPos);
//     }
    
//     void InitializeDataLogging()
//     {
//         int N = platoonVehicles.Length;
//         gap = new List<float>[N - 1];
//         des_gap = new List<float>[N - 1];
//         for (int i = 0; i < N - 1; i++)
//         {
//             gap[i] = new List<float>();
//             des_gap[i] = new List<float>();
//         }
//     }
    
//     void InitializeVehicleStates()
//     {
//         if (!startFromZeroSpeed) return;
        
//         // Set all vehicles to start from zero speed
//         foreach (var vehicle in platoonVehicles)
//         {
//             vehicle.SetVx(0f);
//             vehicle.SetAx(0f);
//         }
        
//         Debug.Log("All vehicles initialized to zero speed");
//     }

//     public void AddVehicleToPlatoon(AdvancedBicycleModel vehicle)
//     {
//         if (platoonVehicles == null)
//         {
//             platoonVehicles = new AdvancedBicycleModel[] { vehicle };
//         }
//         else
//         {
//             var tempList = new System.Collections.Generic.List<AdvancedBicycleModel>(platoonVehicles);
//             tempList.Add(vehicle);
//             platoonVehicles = tempList.ToArray();
//         }
        
//         // Reconfigure arrays
//         SetupPlatoon();
//         InitializeDataLogging();
//     }
    
//     public (float a_des, float s_des) CalculateRajamaniAcceleration(AdvancedBicycleModel Car_1, AdvancedBicycleModel Car_2)
//     {
//         float h = headway;   // Use the inspector value for time headway
        
//         float k1 = -0.12f, k5 = 0.1f; // k1 < -tau/h, k5 > 0
//         float k2 = -k1 - h * k1 * k5;
//         float k3 = 1f / h - k1 * k5;
//         float k4 = k5 / h;

//         // Calculate actual gap and relative velocity
//         float e = Car_2.GetPosition().z - Car_1.GetPosition().z + Car_1.GetVehicleParameters().length + 2f; // [m] actual gap
//         float e_dot = Car_2.GetVx() - Car_1.GetVx(); // [m/s] relative velocity

//         float s_des = Car_1.GetVehicleParameters().length + 2f + h * Car_2.GetVx(); // [m] desired gap
//         float a_des = -k1 * Car_1.GetAx()
//                       - k2 * Car_2.GetAx()
//                       - k3 * e_dot
//                       - k4 * e
//                       - k5 * Car_2.GetVx();

//         return (a_des, s_des);
//     }
    
//     public static float FreeRoadAcceleration(float v, float v_target, float a_max)
//     {
//         float delta = 4f; // exponent

//         float dv_dt;
//         if (v_target >= v)
//         {
//             dv_dt = a_max * (1f - Mathf.Pow(v / v_target, delta));
//         }
//         else
//         {
//             dv_dt = -a_max * (1f - Mathf.Pow(v_target / v, delta));
//         }
//         return dv_dt;
//     }
    
//     void ConvertAccelerationToInputs(float acceleration, int vehicleIndex)
//     {
//         // Reset inputs
//         ThrottleInput[vehicleIndex] = 0f;
//         BrakeInput[vehicleIndex] = 0f;
//         SteeringInput[vehicleIndex] = 0f; // Straight line driving
        
//         if (acceleration > 0.1f)
//         {
//             // Accelerate
//             float throttleGain = 0.5f;
//             ThrottleInput[vehicleIndex] = Mathf.Clamp01(acceleration / MaxAcceleration * throttleGain);
//         }
//         else if (acceleration < -0.1f)
//         {
//             // Decelerate
//             float brakeGain = 0.4f;
//             BrakeInput[vehicleIndex] = Mathf.Clamp01(-acceleration / MaxAcceleration * brakeGain);
//         }
//         else
//         {
//             // Maintain speed - gentle throttle if moving slowly
//             if (platoonVehicles[vehicleIndex].GetVx() < 2f)
//             {
//                 ThrottleInput[vehicleIndex] = 0.1f;
//             }
//         }
//     }

//     void FixedUpdate()
//     {
//         if (platoonVehicles == null || platoonVehicles.Length == 0) return;

//         // Lead vehicle control using Free Road Acceleration
//         AdvancedBicycleModel LeaderVehicle = platoonVehicles[0];
//         float acc_leader = FreeRoadAcceleration(LeaderVehicle.GetVx(), platoonTargetVelocity, MaxAcceleration);
//         float vx_leader = Mathf.Clamp(LeaderVehicle.GetVx() + acc_leader * Time.fixedDeltaTime, 0f, platoonMaxVelocity);

//         //Debug.Log($"Car {1} - Actual Velocity: {vx_leader * 3.6f:F1} km/h, Target Velocity: {platoonTargetVelocity * 3.6f:F1} km/h, Acc Leader: {acc_leader:F2} m/s²");
//         // Update leader vehicle state
//         acc[0] = acc_leader;
//         vx[0] = vx_leader;
//         //LeaderVehicle.SetVx(vx_leader);
//         //LeaderVehicle.SetAx(acc_leader);

//         // Convert leader acceleration to inputs
//         //ConvertAccelerationToInputs(acc_leader, 0);

//         // Following vehicles control using Rajamani
//         for (int car_num = 1; car_num < platoonVehicles.Length; car_num++)
//         {
//             // For each vehicle in the platoon, calculate the desired acceleration and spacing
//             AdvancedBicycleModel Car_ahead = platoonVehicles[car_num - 1]; // Get the preceding vehicle in the platoon
//             AdvancedBicycleModel Car = platoonVehicles[car_num]; // Get the current vehicle in the platoon

//             // Calculate the gap between the two vehicles
//             Debug.Log($"Car {car_num + 1} - Position: {Car.GetPosition().z:F4}m, Ahead Position: {Car_ahead.GetPosition().z:F4}m");
//             float actual_gap = Car_ahead.GetPosition().z - Car.GetPosition().z;
//             gap[car_num - 1].Add(actual_gap);

//             // Calculate desired acceleration and spacing using Rajamani model
//             (float a_des, float s_des) = CalculateRajamaniAcceleration(Car_ahead, Car);
//             des_gap[car_num - 1].Add(s_des);

//             //Debug.Log($"Car {car_num + 1} - Actual Gap: {actual_gap:F1}m, Desired Gap: {s_des:F1}m, Desired Acceleration: {a_des:F2} m/s²");
//             //Debug.Log($"Car {car_num + 1} - Actual Velocity: {Car.GetVx() * 3.6f:F1} km/h, Target Velocity: {platoonTargetVelocity * 3.6f:F1} km/h");

//             // Calculate the actual acceleration based on the desired acceleration
//             float acc_ = Mathf.Clamp(a_des, -MaxAcceleration, MaxAcceleration);
//             float vx_ = Mathf.Clamp(Car.GetVx() + acc_ * Time.fixedDeltaTime, 0f, platoonMaxVelocity);

//             // Convert acceleration to inputs
//             //ConvertAccelerationToInputs(acc, car_num);

//             // Update vehicle state
//             acc[car_num] = acc_;
//             vx[car_num] = vx_;
//             //Car.SetVx(vx_);
//             //Car.SetAx(acc_);
//         }
//         LogPlatoonStatus();
//         //Debug.Log($"des_gap1: {des_gap[1 - 1].Last()} , actual_gap1: {gap[1 - 1].Last()} recorded.");
//         //Debug.Log($"des_gap1_Get: {GetDesiredGap(1)} , actual_gap2: {GetActualGap(1)} recorded.");
//         //Debug.Log($"des_gap2: {des_gap[2-1].Last()} , actual_gap2: {gap[2-1].Last()} recorded.");
//         //Debug.Log($"des_gap2_Get: {GetDesiredGap(2)} , actual_gap2: {GetActualGap(2)} recorded.");

//     }
    
//     public float GetThrottleInput_Pl(int car_num) => car_num < ThrottleInput.Length ? ThrottleInput[car_num] : 0f;
//     public float GetBrakeInput_Pl(int car_num) => car_num < BrakeInput.Length ? BrakeInput[car_num] : 0f;
//     public float GetSteeringInput_Pl(int car_num) => car_num < SteeringInput.Length ? SteeringInput[car_num] : 0f;

//     public float GetActualGap(int car_num)
//     {
//         if (car_num < 1 || car_num > gap.Length || gap[car_num - 1].Count == 0) return 0f;
//         return gap[car_num - 1].Count > 0 ? gap[car_num - 1].Last() : 0f;
//     }
//     public float GetDesiredGap(int car_num)
//     {
//         if (car_num < 1 || car_num > des_gap.Length || des_gap[car_num - 1].Count == 0 ) return 0f;
//         //Debug.Log($"GetDesiredGap called for car_num: {car_num}, des_gap value: {des_gap[car_num - 1][des_gap[car_num - 1].Count - 1]}");
//         // Return the last recorded desired gap for the specified vehicle
//         return des_gap[car_num - 1].Count > 0 ? des_gap[car_num - 1].Last() : 0f;
//     }
//     public float GetTargetVelocity()
//     {
//          return platoonTargetVelocity;
//     }

//     public float GetMaxAcc()
//     {
//          return MaxAcceleration;
//     }
//     public float GetVelocityVehicle(int car_num)
//     {
//          return vx[car_num];
//     }
//     public float GetAccelerationVehicle(int car_num)
//     {
//          return acc[car_num];
//     }

//     // Debug methods
//     [ContextMenu("Log Platoon Status")]
//     public void LogPlatoonStatus()
//     {
//         if (platoonVehicles == null || platoonVehicles.Length == 0) return;
        
//         Debug.Log("=== Platoon Status ===");
//         Debug.Log($"Target Speed: {platoonTargetVelocity * 3.6f:F1} km/h");
//         Debug.Log($"Lead Vehicle Speed: {platoonVehicles[0].GetVx() * 3.6f:F1} km/h");
        
//         for (int i = 1; i < platoonVehicles.Length; i++)
//         {
//             float actualGap = platoonVehicles[i - 1].GetPosition().z - platoonVehicles[i].GetPosition().z;
//             float desiredGap = des_gap[i - 1].Count > 0 ? des_gap[i - 1][des_gap[i - 1].Count - 1] : 0f;
//             float gapError = actualGap - desiredGap;
//             float speedKmh = platoonVehicles[i].GetVx() * 3.6f;
            
//             Debug.Log($"Vehicle {i + 1}: Speed={speedKmh:F1}km/h, Gap={actualGap:F1}m, Desired={desiredGap:F1}m, Error={gapError:F1}m");
//         }
//     }
    
// }