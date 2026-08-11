using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera firstPersonCamera;  // מצלמת הנהג (driver)
    public Camera thirdPersonCamera;  // מצלמת גוף שלישי (dashboard view)

    private bool isThirdPerson = true;
    
    // שיתוף מידע עם מערכות אחרות
    public static CameraSwitcher Instance { get; private set; }
    
    void Awake()
    {
        // Singleton pattern
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
            return;
        }
    }

    void Start()
    {
        // רק הפעל מסכים - אל תשנה את המצלמות!
        SetupDisplaysAsConfigured();
        
        // הפעל את שתי המצלמות כמו שהן מוגדרות
        firstPersonCamera.enabled = true;   
        thirdPersonCamera.enabled = true;   
        isThirdPerson = true; 
        
        // עדכן את הדשבורד
        UpdateDashboardDisplay();
        
        Debug.Log($"🚗 Cameras left as configured: Driver → Display {firstPersonCamera.targetDisplay}, Third Person → Display {thirdPersonCamera.targetDisplay}");
    }
    
    void DisableOtherDisplayManagers()
    {
        DisplayManager[] otherManagers = FindObjectsOfType<DisplayManager>();
        foreach (var manager in otherManagers)
        {
            Debug.Log("🚫 Disabling DisplayManager to prevent conflicts");
            manager.enabled = false;
        }
    }
    
    void SetupDisplaysAsConfigured()
    {
        // רק 2 מסכים - מסך נייד + מסך חיצוני
        if (Display.displays.Length > 1 && !Display.displays[1].active)
        {
            Display.displays[1].Activate();
            Debug.Log("📺 Activated second display");
        }
        
        // תקן את המצלמות ל-2 מסכים בלבד
        thirdPersonCamera.targetDisplay = 0;    // Third Person + Dashboard → Display 0 (מסך ראשי)
        firstPersonCamera.targetDisplay = 1;    // Driver + שעונים → Display 1 (מסך שני)

        Debug.Log("📺 Fixed for 2 displays: Third Person + Dashboard → Display 0, Driver + Gauges → Display 1");
    }

    void Update()
    {
        // אין צורך בשום בקרות - הכל קבוע
    }

    void SwitchCamera()
    {
        if (isThirdPerson)
        {
            firstPersonCamera.enabled = false;
            thirdPersonCamera.enabled = true;
            Debug.Log("📷 Switched to Third Person (with Dashboard)");
        }
        else
        {
            firstPersonCamera.enabled = true;
            thirdPersonCamera.enabled = false;
            Debug.Log("📷 Switched to Driver view");
        }
        
        // עדכן את הדשבורד
        UpdateDashboardDisplay();
    }
    
    void UpdateDashboardDisplay()
    {
        // VehicleDashboard uses OnGUI which renders on the same display as the active camera
    }
    
    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 12;
        style.normal.textColor = Color.yellow;
        style.fontStyle = FontStyle.Bold;

        string driverDisplay = $"Display {firstPersonCamera.targetDisplay}";
        string thirdPersonDisplay = $"Display {thirdPersonCamera.targetDisplay}";
        string info = $"Fixed Setup: Driver Camera → {driverDisplay} | Third Person + Dashboard → {thirdPersonDisplay}";

        GUI.color = new Color(0, 0, 0, 0.8f);
        GUI.DrawTexture(new Rect(5, Screen.height - 105, Screen.width - 10, 35), Texture2D.whiteTexture);
        GUI.color = Color.white;
        GUI.Label(new Rect(10, Screen.height - 100, Screen.width - 20, 30), info, style);
    }

    // פונקציות ציבוריות למערכות אחרות
    public int GetDriverCameraDisplay() => firstPersonCamera.targetDisplay;
    public int GetDashboardCameraDisplay() => thirdPersonCamera.targetDisplay;
    public bool IsThirdPersonActive() => isThirdPerson;
    public Camera GetActiveCamera() => isThirdPerson ? thirdPersonCamera : firstPersonCamera;
    
}