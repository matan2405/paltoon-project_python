using UnityEngine;

public class AutonomousOrManual : MonoBehaviour
{
    // This script is used to determine if the vehicle is autonomous or manual.
    public AdvancedBicycleModel vehicle;
    public enum VehicleType
    {
        Autonomous,
        Manual
    }
    public VehicleType vehicleType = VehicleType.Autonomous;
    public bool isAutonomous => vehicleType == VehicleType.Autonomous;
    public bool isManual => vehicleType == VehicleType.Manual;
    
    // Awake is called when the script instance is being loaded
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        
    }
}
