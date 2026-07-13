using UnityEngine;

public class VehicleInputs : MonoBehaviour
{
    public static VehicleInputs Instance { get; private set; } // Singleton instance

    public LogitechG29Controller controller;
    private LogitechG29Connection connection;

    public float SteeringInput { get; private set; }
    public float ThrottleInput { get; private set; }
    public float BrakeInput { get; private set; }
    private bool PressPedal = false;

    public Transform SteeringWheel;      // SteeringW
    public Transform SteeringWheelPivot; // SteerW_cube
    public Transform PlayerCar;          // PlayerCar — provides forward axis
    public float maxSteeringWheelAngle = 450f;
    private float appliedWheelAngle;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
            return;
        }
    }

    void Start()
    {
        controller = GetComponent<LogitechG29Controller>();
        connection = controller.connection;
        SteeringInput = 0;
        ThrottleInput = 0;
        BrakeInput = 0;
        appliedWheelAngle = 0;
    }

    void FixedUpdate()
    {
        float rawThrottleInput = 0f;
        float rawSteeringInput = 0f;
        float rawBrakeInput = 0f;
        if (connection.IsWheelConnected())
        {
            ThrottleInput = controller.GetThrottle();
            SteeringInput = controller.GetSteering();
            BrakeInput = controller.GetBrake();
        }
        else
        {
            ThrottleInput = Mathf.Clamp(Input.GetAxis("Vertical"), 0, 1);
            SteeringInput = Input.GetAxis("Horizontal");
            BrakeInput = Input.GetAxis("Jump");
        }

        RotateSteeringWheel(SteeringInput);

        // if (!PressPedal && (rawThrottleInput < 0.01f || rawBrakeInput < 0.01f))
        // {
        //     PressPedal = true;
        // }
        // if (PressPedal)
        // {
        //     ThrottleInput = rawThrottleInput;
        //     SteeringInput = rawSteeringInput;
        //     BrakeInput = rawBrakeInput;
        // }
        // else
        // {
        //     ThrottleInput = 0f;
        //     SteeringInput = 0f;
        //     BrakeInput = 0f;
        // }

        if (ThrottleInput !=0f && BrakeInput != 0f)
        {
            ThrottleInput = 0f; // Reset throttle if not pressed
            BrakeInput = 0f; // Reset brake if not pressed
        }
        // Debugging logs for inputs
        Debug.Log($"Steering1: {SteeringInput:F2}, Throttle1: {ThrottleInput:F2}, Brake1: {BrakeInput:F2}");
    }

    void RotateSteeringWheel(float steerInput)
    {
        if (SteeringWheel == null || SteeringWheelPivot == null || PlayerCar == null) return;

        float targetAngle = steerInput * maxSteeringWheelAngle;
        float deltaAngle  = targetAngle - appliedWheelAngle;
        SteeringWheel.RotateAround(SteeringWheelPivot.position, PlayerCar.forward, deltaAngle);
        appliedWheelAngle = targetAngle;
    }
}
