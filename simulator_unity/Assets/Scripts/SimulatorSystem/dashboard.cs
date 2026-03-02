using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// לוח בקרה מלא תואם למערכת החלפת מסכים
/// </summary>
public class VehicleDashboard : MonoBehaviour
{
    [Header("Dashboard Settings")]
    public bool showOnMainDisplay = true;
    public KeyCode toggleDashboardKey = KeyCode.F1;
    
    [Header("Display Integration")]
    public int targetDisplay = 0; // Display 0 - מסך ראשי עם Third Person
    
    private AdvancedBicycleModel vehicle;
    private VehicleParameters vehicleParams;
    private Powertrain powertrain;
    private Transmission transmission;
    private BrakeSystem brakeSystem;
    private SteeringSystem steeringSystem;
    
    private bool dashboardVisible = true;
    
    // GUI Styling
    private GUIStyle titleStyle;
    private GUIStyle labelStyle;
    private GUIStyle boxStyle;
    private GUIStyle headerStyle;
    private GUIStyle bigTextStyle;
    private bool stylesInitialized = false;
    
    // Layout constants
    private const float SCALE = 1.0f;
    private const int COLUMN_WIDTH = 320;
    private const int COLUMN_SPACING = 30;
    private const int BASE_HEIGHT = 20;
    private const int FONT_SIZE = 12;
    private const int BIG_FONT_SIZE = 16;

    void Start()
    {
        // מצא את רכיב הרכב
        vehicle = FindObjectOfType<AdvancedBicycleModel>();
        if (vehicle == null)
        {
            Debug.LogError("AdvancedBicycleModel not found! Dashboard will not work.");
            enabled = false;
            return;
        }
        
        vehicleParams = vehicle.GetVehicleParameters();
        powertrain = vehicle.powertrain;
        transmission = vehicle.GetTransmission();
        brakeSystem = vehicle.GetBrakeSystem();
        steeringSystem = vehicle.GetSteeringSystem();
        
        Debug.Log($"📊 Complete Dashboard initialized. Press {toggleDashboardKey} to toggle.");
    }
    
    void Update()
    {
        if (Input.GetKeyDown(toggleDashboardKey))
        {
            dashboardVisible = !dashboardVisible;
            Debug.Log($"📊 Dashboard visibility: {dashboardVisible}");
        }
        
        // עדכן את targetDisplay לפי CameraSwitcher אם קיים
        UpdateDisplayFromCameraSwitcher();
        
        // בדוק אם עדיין יש חיבור לרכב
        if (vehicle == null)
        {
            vehicle = FindObjectOfType<AdvancedBicycleModel>();
            if (vehicle != null)
            {
                powertrain = vehicle.powertrain;
                transmission = vehicle.GetTransmission();
                brakeSystem = vehicle.GetBrakeSystem();
                steeringSystem = vehicle.GetSteeringSystem();
                vehicleParams = vehicle.GetVehicleParameters();
            }
        }
    }
    
    void UpdateDisplayFromCameraSwitcher()
    {
        // אם יש CameraSwitcher, הדשבורד תמיד עוקב אחרי Third Person Camera
        if (CameraSwitcher.Instance != null)
        {
            int thirdPersonDisplay = CameraSwitcher.Instance.GetDashboardCameraDisplay();
            if (targetDisplay != thirdPersonDisplay)
            {
                targetDisplay = thirdPersonDisplay;
                Debug.Log($"📊 Dashboard follows Third Person camera to display {targetDisplay}");
            }
        }
    }
    
    void InitializeStyles()
    {
        if (stylesInitialized) return;
        
        // כותרת ראשית
        titleStyle = new GUIStyle(GUI.skin.label);
        titleStyle.fontSize = 18;
        titleStyle.fontStyle = FontStyle.Bold;
        titleStyle.normal.textColor = Color.cyan;
        titleStyle.alignment = TextAnchor.MiddleCenter;
        
        // טקסט רגיל
        labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.fontSize = FONT_SIZE;
        labelStyle.normal.textColor = Color.white;
        labelStyle.alignment = TextAnchor.MiddleLeft;
        
        // טקסט גדול
        bigTextStyle = new GUIStyle(GUI.skin.label);
        bigTextStyle.fontSize = BIG_FONT_SIZE;
        bigTextStyle.fontStyle = FontStyle.Bold;
        bigTextStyle.normal.textColor = Color.white;
        bigTextStyle.alignment = TextAnchor.MiddleLeft;
        
        // קופסאות
        boxStyle = new GUIStyle(GUI.skin.box);
        boxStyle.fontSize = FONT_SIZE;
        
        // כותרות קטגוריות
        headerStyle = new GUIStyle(GUI.skin.label);
        headerStyle.fontSize = FONT_SIZE + 2;
        headerStyle.fontStyle = FontStyle.Bold;
        headerStyle.normal.textColor = Color.yellow;
        headerStyle.alignment = TextAnchor.MiddleLeft;
        
        stylesInitialized = true;
    }
    
    void OnGUI()
    {
        if (!dashboardVisible || vehicle == null) return;
        
        // הצג דשבורד רק כשמצלמת השלישי פעילה
        if (CameraSwitcher.Instance != null && !CameraSwitcher.Instance.IsThirdPersonActive())
        {
            return; // אל תציג דשבורד כשמצלמת הנהג פעילה
        }
        
        InitializeStyles();
        DrawCleanDashboard();
    }
    
    bool ShouldShowOnCurrentDisplay()
    {
        // הדשבורד מוצג תמיד כשמצלמת השלישי פעילה
        // והוא תמיד יהיה על אותו מסך כמו מצלמת השלישי
        return true; // פשוט הצג תמיד - ה-targetDisplay מטפל בזה
    }
    
    void DrawCleanDashboard()
    {
        // רקע כהה
        GUI.color = new Color(0, 0, 0, 0.85f);
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), Texture2D.whiteTexture);
        GUI.color = Color.white;
        
        // כותרת ראשית
        Rect titleRect = new Rect(20, 10, Screen.width - 40, 40);
        GUI.Label(titleRect, "🚗 AUDI TT 2.0 TFSI - ADVANCED VEHICLE DASHBOARD 🏎️", titleStyle);
        
        // חשב מיקומי עמודות
        int leftX = 20;
        int centerX = leftX + COLUMN_WIDTH + COLUMN_SPACING;
        int rightX = centerX + COLUMN_WIDTH + COLUMN_SPACING;
        int startY = 70;
        
        // שלוש עמודות נפרדות - כמו במקור!
        DrawVehicleColumn(leftX, startY);
        DrawEngineColumn(centerX, startY);
        DrawSystemColumn(rightX, startY);
        
        // סטטוס בר תחתון
        DrawStatusBar();
    }
    
    void DrawVehicleColumn(int x, int y)
    {
        // קופסה 1: מידע רכב בסיסי
        int box1Height = 200;
        GUI.Box(new Rect(x, y, COLUMN_WIDTH, box1Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "🚗 Vehicle Status", headerStyle);
        
        float vx = vehicle.GetVx();
        float ax = vehicle.GetAx();
        float totalForce = vehicle.GetFx_tot();
        
        // מהירות בגדול
        GUI.color = vx > 50 ? Color.red : (vx > 25 ? Color.yellow : Color.green);
        GUI.Label(new Rect(x + 10, y + 40, COLUMN_WIDTH - 20, 30), 
                  $"🏁 Speed: {vx * 3.6f:F1} km/h", bigTextStyle);
        GUI.color = Color.white;
        
        GUI.Label(new Rect(x + 10, y + 80, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"⚡ Acceleration: {ax:F2} m/s²", labelStyle);
        GUI.Label(new Rect(x + 10, y + 100, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"💪 Total Force: {totalForce:F0} N", labelStyle);
        
        // הילוך נוכחי
        GUI.color = transmission.IsShifting() ? Color.red : Color.cyan;
        string gearText = transmission.IsShifting() ? 
            $"Gear: {transmission.GetCurrentGear()} (Shifting...)" : 
            $"Gear: {transmission.GetCurrentGear()}";
        GUI.Label(new Rect(x + 10, y + 130, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"⚙️ {gearText}", labelStyle);
        GUI.color = Color.white;
        
        // מצב מנוע
        float throttle = VehicleInputs.Instance.ThrottleInput;
        string engineMode = GetEngineMode(throttle, vx);
        GUI.color = GetEngineColor(throttle, vx);
        GUI.Label(new Rect(x + 10, y + 150, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🔧 Engine: {engineMode}", labelStyle);
        GUI.color = Color.white;
        
        // קופסה 2: בקרות - כמו במקור!
        int box2Y = y + box1Height + 20;
        int box2Height = 180;
        GUI.Box(new Rect(x, box2Y, COLUMN_WIDTH, box2Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, box2Y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "🎮 Vehicle Controls", headerStyle);
        
        DrawControlsOnly(x + 10, box2Y + 40, COLUMN_WIDTH - 20);
        
        // קופסה 3: דינמיקה רוחבית
        int box3Y = box2Y + box2Height + 20;
        int box3Height = 100;
        GUI.Box(new Rect(x, box3Y, COLUMN_WIDTH, box3Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, box3Y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "↔️ Lateral Dynamics", headerStyle);
        
        GUI.Label(new Rect(x + 10, box3Y + 40, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"Lateral Speed: {vehicle.GetYDot():F2} m/s", labelStyle);
        GUI.Label(new Rect(x + 10, box3Y + 60, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"Yaw Rate: {vehicle.GetPsiDot():F2} rad/s", labelStyle);
    }
    
    void DrawEngineColumn(int x, int y)
    {
        // קופסה 1: מידע מנוע - כמו במקור!
        int box1Height = 300;
        GUI.Box(new Rect(x, y, COLUMN_WIDTH, box1Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "🏎️ Engine Analysis", headerStyle);
        
        if (powertrain == null) return;
        
        float rpm = powertrain.EngineRpm;
        
        // RPM בגדול
        GUI.color = rpm > 6000 ? Color.red : (rpm > 4500 ? Color.yellow : Color.green);
        GUI.Label(new Rect(x + 10, y + 40, COLUMN_WIDTH - 20, 30), 
                  $"🔢 RPM: {rpm:F0}", bigTextStyle);
        GUI.color = Color.white;
        
        GUI.Label(new Rect(x + 10, y + 80, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🎯 Zone: {GetEngineZoneDescription(rpm)}", labelStyle);
        
        // כוח ומומנט
        float currentTorque = powertrain.GetCurrentEngineTorque();
        float currentPowerKW = powertrain.GetCurrentEnginePowerKW();
        float currentPowerPS = currentPowerKW * 1.36f;
        
        GUI.Label(new Rect(x + 10, y + 110, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"💪 Torque: {currentTorque:F0} Nm ({GetTorquePercentage(rpm):F0}%)", labelStyle);
        GUI.Label(new Rect(x + 10, y + 130, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"⚡ Power: {currentPowerPS:F0} PS ({GetPowerPercentage(rpm):F0}%)", labelStyle);
        
        // השוואת מהירויות
        float vx = vehicle.GetVx();
        float theoreticalSpeed = powertrain.GetTheoreticalSpeed();
        float wheelSlip = powertrain.GetWheelSlip(vx);
        
        GUI.Label(new Rect(x + 10, y + 160, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "--- Speed Comparison ---", headerStyle);
        GUI.Label(new Rect(x + 10, y + 180, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"📊 Actual: {vx * 3.6f:F1} km/h", labelStyle);
        GUI.Label(new Rect(x + 10, y + 200, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"📈 Theoretical: {theoreticalSpeed * 3.6f:F1} km/h", labelStyle);
        
        // wheel slip עם צבע
        Color originalColor = GUI.color;
        if (Mathf.Abs(wheelSlip) > 10f)
            GUI.color = Color.red;
        else if (Mathf.Abs(wheelSlip) > 5f)
            GUI.color = Color.yellow;
        else
            GUI.color = Color.green;
            
        GUI.Label(new Rect(x + 10, y + 220, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🎯 Wheel Slip: {wheelSlip:F1}%", labelStyle);
        GUI.color = originalColor;
        
        // אזור עבודה
        DrawWorkingZone(x + 10, y + 250, COLUMN_WIDTH - 20, rpm);
        
        // קופסה 2: תיבת הילוכים - כמו במקור!
        int box2Y = y + box1Height + 20;
        int box2Height = 200;
        GUI.Box(new Rect(x, box2Y, COLUMN_WIDTH, box2Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, box2Y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "⚙️ Transmission System", headerStyle);
        
        // הילוכים בקומפקטי
        for (int i = 1; i <= 6; i++)
        {
            Color color = GUI.color;
            if (i == transmission.GetCurrentGear())
            {
                GUI.color = transmission.IsShifting() ? Color.red : Color.green;
                GUI.Label(new Rect(x + 10, box2Y + 30 + (i * 20), COLUMN_WIDTH - 20, BASE_HEIGHT), 
                          $"► {i}: {transmission.GetGearSpeedRange(i)} ◄", labelStyle);
            }
            else
            {
                GUI.color = Color.gray;
                GUI.Label(new Rect(x + 10, box2Y + 30 + (i * 20), COLUMN_WIDTH - 20, BASE_HEIGHT), 
                          $"   {i}: {transmission.GetGearSpeedRange(i)}", labelStyle);
            }
            GUI.color = color;
        }
        
        GUI.Label(new Rect(x + 10, box2Y + 170, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"Drive Mode: {transmission.GetDriveMode()}", labelStyle);
    }
    
    void DrawSystemColumn(int x, int y)
    {
        // קופסה 1: פירוט כוחות - כמו במקור!
        int box1Height = 280;
        GUI.Box(new Rect(x, y, COLUMN_WIDTH, box1Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "🔧 Force Analysis (N)", headerStyle);
        
        float throttle = VehicleInputs.Instance.ThrottleInput;
        float brake = VehicleInputs.Instance.BrakeInput;
        float vx = vehicle.GetVx();
        
        // חשב כוחות
        float engineForce = powertrain.CalculateEngineForce(throttle, vx);
        float brakeForce = brakeSystem.CalculateBrakeForce(brake, vx);
        float aeroForce = 0.5f * 1.225f * vehicleParams.dragCoefficient * vehicleParams.frontalArea * vx * vx;
        float rollingForce = vx > 0.01f ? vehicleParams.mass * vehicleParams.gravity * 0.012f : 0;
        float naturalDecel = (throttle < 0.001f && brake < 0.001f && vx > 0.1f) ? 
            powertrain.CalculateNaturalDeceleration(vx, powertrain.EngineRpm) : 0;
        float totalForce = vehicle.GetFx_tot();
        
        int yOffset = 40;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🚀 Engine Force: +{engineForce:F0}", labelStyle);
        yOffset += 25;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🛑 Brake Force: -{brakeForce:F0}", labelStyle);
        yOffset += 25;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"💨 Air Resistance: -{aeroForce:F0}", labelStyle);
        yOffset += 25;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🛞 Rolling Resistance: -{rollingForce:F0}", labelStyle);
        yOffset += 25;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🔧 Engine Braking: -{naturalDecel:F0}", labelStyle);
        
        yOffset += 40;
        Color originalColor = GUI.color;
        GUI.color = (totalForce > 0) ? Color.green : Color.red;
        GUI.Label(new Rect(x + 10, y + yOffset, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"📊 NET TOTAL: {totalForce:F0} N", bigTextStyle);
        GUI.color = originalColor;
        
        // מצב חיבור מנוע
        yOffset += 40;
        DrawEngineConnectionStatus(x + 10, y + yOffset, COLUMN_WIDTH - 20, throttle, vx);
        
        // קופסה 2: מידע מערכת
        int box2Y = y + box1Height + 20;
        int box2Height = 120;
        GUI.Box(new Rect(x, box2Y, COLUMN_WIDTH, box2Height), "", boxStyle);
        GUI.Label(new Rect(x + 10, box2Y + 10, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  "💻 System Information", headerStyle);
        
        GUI.Label(new Rect(x + 10, box2Y + 40, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"🖥️ Current Display: {targetDisplay}", labelStyle);
        GUI.Label(new Rect(x + 10, box2Y + 60, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"📺 Total Displays: {Display.displays.Length}", labelStyle);
        GUI.Label(new Rect(x + 10, box2Y + 80, COLUMN_WIDTH - 20, BASE_HEIGHT), 
                  $"⚡ Current FPS: {(1.0f / Time.deltaTime):F0}", labelStyle);
    }

    void DrawControlsOnly(int x, int y, int width)
    {
        // רק נתוני הבקרות - כמו במקור!
        float throttle = VehicleInputs.Instance.ThrottleInput;
        float brake = VehicleInputs.Instance.BrakeInput;
        float steering = VehicleInputs.Instance.SteeringInput;

        // דוושת גז
        Color originalColor = GUI.color;
        GUI.color = throttle > 0.8f ? Color.red : (throttle > 0.5f ? Color.yellow : Color.green);
        GUI.Label(new Rect(x, y, 140, BASE_HEIGHT),
                  $"🚀 Throttle: {throttle * 100:F0}%", labelStyle);
        GUI.color = originalColor;
        DrawInputBar(x + 150, y + 3, 120, 12, throttle, Color.green);

        // דוושת בלמים
        GUI.color = brake > 0.7f ? Color.red : Color.white;
        GUI.Label(new Rect(x, y + 25, 140, BASE_HEIGHT),
                  $"🛑 Brake: {brake * 100:F0}%", labelStyle);
        GUI.color = originalColor;
        DrawInputBar(x + 150, y + 28, 120, 12, brake, Color.red);

        // הגה
        var angles = steeringSystem.CalculateAckermanSteering(steering);
        string steeringDirection = steering > 0.05f ? "RIGHT ➡️" : (steering < -0.05f ? "LEFT ⬅️" : "STRAIGHT ⬆️");


        GUI.Label(new Rect(x, y + 55, width, BASE_HEIGHT),
                  $"🚗 Steering: {steeringDirection}", labelStyle);
        GUI.Label(new Rect(x, y + 75, width, BASE_HEIGHT),
                  $"🎯 Wheel Angle: {angles.average * Mathf.Rad2Deg:F1}°", labelStyle);

        // בר הגה
        DrawSteeringBar(x, y + 100, width, 15, steering);
        
        
        GUI.Label(new Rect(x, y+120, 1450, BASE_HEIGHT),
                  $"🚀 Wheel Steeing Angle: {vehicle.GetSteeringWheelAngle()}%", labelStyle);
        GUI.color = originalColor;
    }
    
    void DrawStatusBar()
    {
        int barHeight = 35;
        int startY = Screen.height - barHeight - 5;
        
        GUI.color = new Color(0, 0, 0, 0.8f);
        GUI.DrawTexture(new Rect(0, startY, Screen.width, barHeight), Texture2D.whiteTexture);
        GUI.color = Color.white;
        
        GUIStyle statusStyle = new GUIStyle(labelStyle);
        statusStyle.alignment = TextAnchor.MiddleCenter;
        statusStyle.normal.textColor = Color.cyan;
        
        string controls = "F1: Toggle Dashboard | C: Switch Camera | F9: Swap Displays | WASD: Drive | Space: Brake";
        GUI.Label(new Rect(0, startY + 5, Screen.width, 25), controls, statusStyle);
    }
    
    // Helper methods - כמו במקור!
    void DrawInputBar(int x, int y, int width, int height, float value, Color color)
    {
        GUI.color = Color.gray;
        GUI.DrawTexture(new Rect(x, y, width, height), Texture2D.whiteTexture);
        
        GUI.color = color;
        int fillWidth = (int)(width * Mathf.Clamp01(value));
        if (fillWidth > 0)
        {
            GUI.DrawTexture(new Rect(x, y, fillWidth, height), Texture2D.whiteTexture);
        }
        
        GUI.color = Color.black;
        DrawRect(new Rect(x, y, width, height), 1);
        GUI.color = Color.white;
    }
    
    void DrawSteeringBar(int x, int y, int width, int height, float steeringValue)
    {
        GUI.color = Color.gray;
        GUI.DrawTexture(new Rect(x, y, width, height), Texture2D.whiteTexture);
        
        int centerX = x + width / 2;
        GUI.color = Color.white;
        GUI.DrawTexture(new Rect(centerX - 1, y, 2, height), Texture2D.whiteTexture);
        
        int steeringX = centerX + (int)(steeringValue * width / 2);
        GUI.color = steeringValue > 0 ? Color.blue : (steeringValue < 0 ? Color.red : Color.green);
        GUI.DrawTexture(new Rect(steeringX - 3, y - 2, 6, height + 4), Texture2D.whiteTexture);
        
        GUI.color = Color.black;
        DrawRect(new Rect(x, y, width, height), 1);
        GUI.color = Color.white;
    }
    
    void DrawRect(Rect rect, int thickness)
    {
        GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, thickness), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.x, rect.y + rect.height - thickness, rect.width, thickness), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.x, rect.y, thickness, rect.height), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.x + rect.width - thickness, rect.y, thickness, rect.height), Texture2D.whiteTexture);
    }
    
    string GetEngineZoneDescription(float rpm)
    {
        if (rpm < 800) return "Engine Off";
        else if (rpm < 1200) return "Idle/Low";
        else if (rpm < 1600) return "Torque Rise";
        else if (rpm < 4300) return "Torque Plateau";
        else if (rpm < 4500) return "Power Transition";
        else if (rpm < 6200) return "Power Plateau";
        else if (rpm < 6700) return "Above Max Power";
        else return "Redline!";
    }
    
    float GetTorquePercentage(float rpm)
    {
        if (powertrain == null) return 0f;
        float currentTorque = powertrain.GetCurrentEngineTorque();
        return (currentTorque / 370f) * 100f;
    }
    
    float GetPowerPercentage(float rpm)
    {
        if (powertrain == null) return 0f;
        float currentPowerKW = powertrain.GetCurrentEnginePowerKW();
        return (currentPowerKW / 169f) * 100f;
    }
    
    string GetEngineMode(float throttle, float vx)
    {
        if (throttle > 0.001f)
            return "🚀 Driving";
        else if (vx > 0.1f)
            return "🛑 Engine Braking";
        else
            return "⭕ Idle";
    }
    
    Color GetEngineColor(float throttle, float vx)
    {
        if (throttle > 0.001f)
            return Color.green;
        else if (vx > 0.1f)
            return Color.yellow;
        else
            return Color.gray;
    }
    
    void DrawWorkingZone(int x, int y, int width, float rpm)
    {
        Color originalColor = GUI.color;
        if (rpm >= 1600 && rpm <= 4300)
        {
            GUI.color = Color.green;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "✅ Max Torque Zone", labelStyle);
        }
        else if (rpm >= 4500 && rpm <= 6200)
        {
            GUI.color = Color.blue;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "⚡ Max Power Zone", labelStyle);
        }
        else if (rpm >= 1200 && rpm < 1600)
        {
            GUI.color = Color.yellow;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "📈 Rising Zone", labelStyle);
        }
        else if (rpm > 6200)
        {
            GUI.color = Color.red;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "⚠️ High RPM", labelStyle);
        }
        else
        {
            GUI.color = Color.gray;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "⭕ Low Zone", labelStyle);
        }
        GUI.color = originalColor;
    }
    
    void DrawEngineConnectionStatus(int x, int y, int width, float throttle, float vx)
    {
        Color originalColor = GUI.color;
        if (throttle < 0.001f && vx > 0.1f)
        {
            GUI.color = Color.yellow;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "🔗 Engine Connected", labelStyle);
            GUI.Label(new Rect(x, y + 20, width, BASE_HEIGHT), "Engine Braking Active", labelStyle);
        }
        else if (throttle > 0.001f)
        {
            GUI.color = Color.green;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "🚀 Engine Driving", labelStyle);
        }
        else
        {
            GUI.color = Color.gray;
            GUI.Label(new Rect(x, y, width, BASE_HEIGHT), "⭕ Engine at Rest", labelStyle);
        }
        GUI.color = originalColor;
    }
}