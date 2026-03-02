using UnityEngine;

public class LogitechG29Connection : MonoBehaviour
{
    private bool isWheelConnected = false;

    void Start()
    {
        CheckWheelConnection();
    }

    void Update()
    {
        // Check connection every 5 seconds
        if (Time.frameCount % 300 == 0)
        {
            CheckWheelConnection();
        }
    }

    private void CheckWheelConnection()
    {
        string[] joystickNames = Input.GetJoystickNames();
        isWheelConnected = false;

        foreach (string joystick in joystickNames)
        {
            if (!string.IsNullOrEmpty(joystick) &&
                (joystick.ToLower().Contains("g29") || joystick.ToLower().Contains("logitech")))
            {
                isWheelConnected = true;
                Debug.Log("G29 wheel is connected");
                break;
            }
        }

        if (!isWheelConnected)
        {
            Debug.LogWarning("G29 wheel is not connected!");
        }
    }

    public bool IsWheelConnected()
    {
        return isWheelConnected;
    }
}