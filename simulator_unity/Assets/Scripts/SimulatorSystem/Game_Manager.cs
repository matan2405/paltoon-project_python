using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class Game_Manager : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public GameObject needle_vel;
    public GameObject needle_rpm;
    public TMP_Text velocity;
    public TMP_Text rpm;
    public TMP_Text gear;
    public TMP_Text totalDistance;
    private float startPos_vel = 220, endPos_vel = -49;
    private float startPos_rpm = 49, endPos_rpm = -220;
    private float desiredPos_rpm;
    private float desiredPos_vel;
    public AdvancedBicycleModel bicycleModel;
    private VehicleParameters vehicleParams;
    public float vehicleVelocity;
    public float vehicleRPM;
    public float vehicleGear;
    void Start()
    {
        vehicleVelocity = 0f;
        vehicleRPM = 0;
        UpdateNeedle_vel();
        UpdateNeedle_rpm();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        vehicleVelocity = (bicycleModel.GetVx()) * 3.6f;
        vehicleRPM = bicycleModel.GetEngineRpm();
        UpdateNeedle_vel();
        UpdateNeedle_rpm();
        UpdateGear();
    }

    public void UpdateNeedle_vel()
    {
        desiredPos_vel = startPos_vel - endPos_vel;
        float temp = vehicleVelocity / 180;
        needle_vel.transform.eulerAngles = new Vector3(0, 0, (startPos_vel - temp * desiredPos_vel));
        velocity.text = Mathf.Round(vehicleVelocity).ToString();
        totalDistance.text = Mathf.Round(bicycleModel.GetTotalDistance()).ToString() + " m";
    }
    public void UpdateNeedle_rpm()
    {
        desiredPos_rpm = startPos_rpm - endPos_rpm;
        float temp = vehicleRPM / 7000;
        needle_rpm.transform.eulerAngles = new Vector3(0, 0, (startPos_rpm - temp * desiredPos_rpm));
        rpm.text = "RPM: " + Mathf.Round(vehicleRPM).ToString();
    }
    public void UpdateGear()
    {
        // Update the gear display based on the current gear of the bicycle model
        vehicleGear = bicycleModel.GetCurrentGear();
        // Assuming you have a UI Text element to display the gear
        gear.text = "Gear: " +  vehicleGear.ToString();
}
}
