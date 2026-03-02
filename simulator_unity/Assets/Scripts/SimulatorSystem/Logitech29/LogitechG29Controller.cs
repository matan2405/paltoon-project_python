using UnityEngine;
using UnityEngine.InputSystem;

public class LogitechG29Controller : MonoBehaviour
{

    // Input values
    private float steering;
    private float throttle;
    private float brake;
    //private float clutch;

    // Sensitivity settings
    [SerializeField] private float steeringDeadzone = 0.001f;
    [SerializeField] private float pedalDeadzone = 0.01f;
    [SerializeField] private float steeringSensitivity = 1.0f;
    public LogitechG29Connection connection;

    private void Start()
    {
        connection = GetComponent<LogitechG29Connection>();

    }
    void FixedUpdate()
    {
        if (connection.IsWheelConnected())
        {
            // Get steering input (-1 to 1 range)
            steering = ProcessInput(Input.GetAxis("Steer"), steeringDeadzone) * steeringSensitivity;
            // Get pedal inputs (0 to 1 range)
            // Note: Pedals are typically -1 to 1 in raw input, so we convert to 0 to 1
            throttle = ProcessInput((-Input.GetAxis("G29Throttle")+1)/2 , pedalDeadzone);
            brake = ProcessInput((-Input.GetAxis("G29Brake") + 1f) / 2f, pedalDeadzone);
            //clutch = ProcessInput((-Input.GetAxis("G29Clutch") + 1f) / 2f, pedalDeadzone);

            // Example of using the inputs
            Debug.Log($"Steering: {steering:F2}, Throttle: {throttle:F2}, Brake: {brake:F2}");
        }
    }
    private float ProcessInput(float input, float deadzone)
    {
        // Apply deadzone
        if (Mathf.Abs(input) < deadzone)
            return 0f;

        // Normalize the input accounting for deadzone
        return Mathf.Sign(input) * (Mathf.Abs(input) - deadzone) / (1 - deadzone);
    }
    
    // Public getters for other scripts to access the inputs
    public float GetSteering() => steering;
    public float GetThrottle() => throttle;
    public float GetBrake() => brake;
    //public float GetClutch() => clutch;

}