using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(LogitechG29Controller))]
public class KinematicBicycleModel : MonoBehaviour
{
    public LogitechG29Controller controller;
    public LogitechG29Connection connection;
    [Header("Vehicle Parameters")]
    [SerializeField] private float LengthCar = 4.177f; //Length of the car (meters)
    [SerializeField] private float b = 1.832f; //Width of the car (meters)
    [SerializeField] private float h = 1.353f; //height of the car (meters)
    [SerializeField] private float wheelbase = 2.505f;          // Distance between front and rear axles (meters)
    [SerializeField] private float lw_f = 1.578f;    // front wheels axle (meters)
    [SerializeField] private float lw_r = 1.564f;    // rear wheels axle (meters)
    [SerializeField] private float rearAxleToCoG = 1.4f;    // Distance from rear axle to Center of Gravity (meters)
    [SerializeField] private float maxSteeringAngle = 30.0f;  // Maximum steering angle in degrees
    [SerializeField] private float maxSpeed = 100.0f;          // Maximum speed in meters per second
    [SerializeField] private float maxAcceleration = (float)(100.0/3.6/3.7);   // Maximum acceleration in meters per second^2
    [SerializeField] private float maxBrakeDeceleration = (float)(1.25* Physics.gravity.magnitude); // max decelartion ->1.25g in meters per second^2
    [SerializeField] private float TorqueEngine = 370f; // max engine torque - N/m
    [SerializeField] private float differential_ratio = 4.18f; // max engine torque - N/m
    //[SerializeField] float[] gear_ratio; // max engine torque - N/m
    [SerializeField] private float wheel_R = 0.4826f / 2; // radius of the wheel (meters)
    [SerializeField] private float wheel_mass = 10.0f; // wheel mass (kg)
    

    [Header("Current State")]
    [SerializeField] public float currentVelocity = 0.0f;       // Current speed in meters per second
    [SerializeField] public float currentSteeringAngle = 0.0f; // Current steering angle in degrees
    [SerializeField] public float currentAcceleration = 0.0f;// Current acceleration in meters per second^2
    [SerializeField] public float currentBeta = 0.0f;
    [SerializeField] public float TorqueWheel = 0.0f;

    //[SerializeField] private LogitechG29Controller

    private int currentPoint = 0;

    // Internal variables
    private Rigidbody rb;
    
    public Vector3 position;
    public float heading;  // Vehicle heading in radians
    private float beta;     // Slip angle at center of gravity
    float wheel_inertia;
    
    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.isKinematic = true;  // We'll handle the physics ourselves
        wheel_inertia = 0.5f * wheel_mass * Mathf.Pow(wheel_R,2);
        connection = GetComponent<LogitechG29Connection>();
        controller = GetComponent<LogitechG29Controller>();


        // Initialize position and heading
        position = transform.position;
        heading = transform.eulerAngles.y * Mathf.Deg2Rad; //convert the angle to degrees to radians
        //TorqueWheel = TorqueEngine * differential_ratio * gear_ratio[0];
        Debug.Log(currentPoint);
        Debug.Log("Time: " + Time.time);
        Debug.Log("Pos_x: " + position.x);
        Debug.Log("Pos_z: " + position.z);
        Debug.Log("Heading angle: " + heading);
        Debug.Log("Velocity: " + currentVelocity);
        Debug.Log("Acceleration: " +currentAcceleration);
        currentPoint++;
    }

     private void FixedUpdate()
    {
        float accelerationInput;
        float steeringInput;
        float brakeInput;
        if (connection.IsWheelConnected())
        {
            accelerationInput = controller.GetThrottle();
            steeringInput = controller.GetSteering();
            brakeInput = controller.GetBrake();
        }
        else
        {
            // Get input (you can modify this based on your input system)
            accelerationInput = Input.GetAxis("Vertical");
            steeringInput = Input.GetAxis("Horizontal");
            brakeInput = Input.GetAxis("Jump");
            Debug.Log($"Steering: {steeringInput:F2}, Throttle: {accelerationInput:F2}, Brake: {brakeInput:F2}");

        }
        // Update speed and steering
        if (accelerationInput != 0.0f)
        {
            UpdateSpeed(accelerationInput);
        }
   
        UpdateSteering(steeringInput);

        //ApplyFriction();
        ApplyBraking(brakeInput);
        
        // Calculate vehicle dynamics
        CalculateKinematicBicycleModel();

        // Update vehicle position and rotation
        UpdateVehicleTransform();
    }

    private void UpdateSpeed(float accelerationInput)
    {
        // Simple speed control - you can make this more sophisticated
        currentAcceleration = accelerationInput * maxAcceleration; // Max acceleration
        currentVelocity += currentAcceleration * Time.deltaTime; // changing according current acceleration
        currentVelocity = Mathf.Clamp(currentVelocity, 0, maxSpeed); // saving the limits of the velocity (0<currentVelocity<maxSpeed)
    }

    private void UpdateSteering(float steeringInput)
    {
        // Update steering angle based on input
        float targetSteeringAngle = steeringInput * maxSteeringAngle;
        currentSteeringAngle = Mathf.Lerp(currentSteeringAngle, targetSteeringAngle, Time.deltaTime * 1f);
    }

    private void CalculateKinematicBicycleModel()
    {
        if (Mathf.Approximately(currentVelocity, 0f)) // varified that the calculation don't happend when the velocity close to zero
            return;

        // Convert steering angle to radians
        float steeringRad = currentSteeringAngle * Mathf.Deg2Rad;

        // Calculate slip angle at center of gravity
        beta = Mathf.Atan2(rearAxleToCoG * Mathf.Tan(steeringRad), wheelbase);

        // Calculate rate of change of heading
        float headingRate = (currentVelocity * Mathf.Cos(beta) * Mathf.Tan(steeringRad)) / wheelbase;

        // Update heading
        heading += headingRate * Time.deltaTime;

        // Update position
        float Longitudinal_velocity= currentVelocity * Mathf.Cos(heading + beta);
        float Lateral_velocity = currentVelocity * Mathf.Sin(heading + beta);

        position.z += Longitudinal_velocity * Time.deltaTime; // in this case, Z is the longitudinal axile
        position.x += Lateral_velocity * Time.deltaTime; // in this case, X is the lateral axile

        currentBeta = beta;
    }
    private void ApplyBraking(float brakeInput)
    {
        if (brakeInput > 0)
        {
            // חישוב התאוטה ביחס לאחוז הלחיצה על דוושת הבלימה
            currentAcceleration = brakeInput * maxBrakeDeceleration * (-1);
            currentVelocity += currentAcceleration * Time.deltaTime; // changing according current acceleration
            // עדכון מהירות הרכב על פי התאוטה
            currentVelocity = Mathf.Clamp(currentVelocity, 0, maxSpeed);
        }
    }

    private void ApplyFriction()
    {
        float mass = rb.mass;  
        float gravity = Physics.gravity.magnitude; 
        float frictionCoefficient = 0.015f; // לדוגמה עבור אספלט
        float frictionForce = frictionCoefficient * mass * gravity;

        currentAcceleration = (frictionForce / mass)*(-1);
        currentVelocity += currentAcceleration * Time.deltaTime; // changing according current acceleration
        currentVelocity = Mathf.Clamp(currentVelocity, 0, maxSpeed); // saving the limits of the velocity (0<currentVelocity<maxSpeed)

        // Calculated affection of the friction coefficient on the velocity
        currentVelocity = Mathf.Clamp(currentVelocity, 0, maxSpeed);
    }
    private void UpdateVehicleTransform()
    {
        // Update position
        transform.position = position;

        // Update rotation (convert heading from radians to degrees)
        transform.rotation = Quaternion.Euler(0f, heading * Mathf.Rad2Deg, 0f);

        //Debug.Log(currentPoint);
        //Debug.Log("Time: " + Time.time);
        //Debug.Log("Pos_x: " + position.x);
        //Debug.Log("Pos_z: " + position.z);
        //Debug.Log("Heading angle: " + heading);
        //Debug.Log("Velocity: " + currentVelocity);
        //Debug.Log("Acceleration: " + currentAcceleration);
        currentPoint++;
    }

    // Helper method to visualize the model in the Unity Editor
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying)
            return;

        // Draw velocity vector
        Gizmos.color = Color.green;
        Vector3 velocityDir = transform.TransformDirection(new Vector3(Mathf.Cos(beta), 0f, Mathf.Sin(beta)));
        Gizmos.DrawRay(transform.position, velocityDir * currentVelocity);

        // Draw steering direction
        Gizmos.color = Color.red;
        Vector3 steeringDir = Quaternion.Euler(0f, currentSteeringAngle, 0f) * transform.forward;
        Gizmos.DrawRay(transform.position + transform.forward * wheelbase, steeringDir * 2f);
    }
}