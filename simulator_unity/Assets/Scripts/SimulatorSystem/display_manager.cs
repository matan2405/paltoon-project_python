using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// מנהל תצוגות פשוט ללא שגיאות
/// </summary>
public class DisplayManager : MonoBehaviour
{
    // [Header("Display Settings")]
    // public bool autoSetupDisplays = true;
    // public bool moveGameToSecondDisplay = true;
    // public bool showDashboardOnMainDisplay = true;
    
    // [Header("Display Assignment")]
    // public int gameDisplayTarget = 1;      // המשחק למסך השני
    // public int dashboardDisplayTarget = 0;  // הדשבורד למסך הראשי
    
    // private static DisplayManager instance;
    // public static DisplayManager Instance
    // {
    //     get
    //     {
    //         if (instance == null)
    //         {
    //             instance = FindObjectOfType<DisplayManager>();
    //             if (instance == null)
    //             {
    //                 GameObject go = new GameObject("DisplayManager");
    //                 instance = go.AddComponent<DisplayManager>();
    //             }
    //         }
    //         return instance;
    //     }
    // }
    
    // private List<Display> availableDisplays = new List<Display>();
    // private Camera mainGameCamera;
    // private Camera dashboardCamera;
    // private bool isInitialized = false;
    
    // void Awake()
    // {
    //     if (instance == null)
    //     {
    //         instance = this;
    //         DontDestroyOnLoad(gameObject);
    //     }
    //     else if (instance != this)
    //     {
    //         Destroy(gameObject);
    //         return;
    //     }
        
    //     Initialize();
    // }
    
    // void Initialize()
    // {
    //     if (isInitialized) return;
        
    //     DetectDisplays();
    //     if (autoSetupDisplays)
    //     {
    //         SetupDisplays();
    //     }
        
    //     isInitialized = true;
    // }
    
    // void DetectDisplays()
    // {
    //     availableDisplays.Clear();
        
    //     Debug.Log($"🖥️ Detecting displays... Found {Display.displays.Length} display(s)");
        
    //     for (int i = 0; i < Display.displays.Length; i++)
    //     {
    //         Display display = Display.displays[i];
    //         availableDisplays.Add(display);
            
    //         Debug.Log($"🖥️ Display {i}: {display.systemWidth}x{display.systemHeight}");
            
    //         // הפעל תצוגות נוספות
    //         if (i > 0 && !display.active)
    //         {
    //             display.Activate();
    //             Debug.Log($"✅ Activated display {i}");
    //         }
    //     }
    // }
    
    // void SetupDisplays()
    // {
    //     // מצא את המצלמה הראשית של המשחק
    //     mainGameCamera = Camera.main;
    //     if (mainGameCamera == null)
    //     {
    //         mainGameCamera = FindObjectOfType<Camera>();
    //     }
        
    //     if (availableDisplays.Count > 1)
    //     {
    //         // הגדר מסכים מרובים
    //         SetupMultipleDisplays();
    //     }
    //     else
    //     {
    //         // מסך יחיד - הכל על המסך הראשי
    //         SetupSingleDisplay();
    //     }
    // }
    
    // void SetupMultipleDisplays()
    // {
    //     Debug.Log("🎮 Setting up REVERSED dual display configuration:");
    //     Debug.Log($"🎮 Game Camera → Display {gameDisplayTarget} (Second Monitor)");
    //     Debug.Log($"📊 Dashboard → Display {dashboardDisplayTarget} (Main Monitor)");
        
    //     // הזז את מצלמת המשחק למסך השני
    //     if (mainGameCamera != null && moveGameToSecondDisplay)
    //     {
    //         mainGameCamera.targetDisplay = gameDisplayTarget;
    //         Debug.Log($"🎮 Main game camera moved to display {gameDisplayTarget}");
    //     }
        
    //     // צור דשבורד במסך הראשי
    //     if (showDashboardOnMainDisplay)
    //     {
    //         SetupDashboardCamera();
    //     }
    // }
    
    // void SetupSingleDisplay()
    // {
    //     Debug.Log("🖥️ Single display detected - keeping everything on main display");
        
    //     if (mainGameCamera != null)
    //     {
    //         mainGameCamera.targetDisplay = 1;
    //     }
        
    //     // צור דשבורד גם במסך היחיד
    //     SetupDashboardCamera();
    // }
    
    // void SetupDashboardCamera()
    // {
    //     // חפש או צור מצלמת דשבורד
    //     dashboardCamera = GameObject.FindWithTag("DashboardCamera")?.GetComponent<Camera>();
        
    //     if (dashboardCamera == null)
    //     {
    //         GameObject dashboardCameraGO = new GameObject("Dashboard Camera");
    //         dashboardCameraGO.tag = "DashboardCamera";
    //         dashboardCamera = dashboardCameraGO.AddComponent<Camera>();
    //     }
        
    //     // הגדר את דשבורד למסך הראשי (או למסך המתאים)
    //     dashboardCamera.targetDisplay = dashboardDisplayTarget;
    //     dashboardCamera.clearFlags = CameraClearFlags.SolidColor;
    //     dashboardCamera.backgroundColor = new Color(0.05f, 0.05f, 0.1f, 1f); // רקע כהה כחלחל
    //     dashboardCamera.cullingMask = LayerMask.GetMask("Dashboard"); // רק שכבת דשבורד
    //     dashboardCamera.orthographic = true;
    //     dashboardCamera.orthographicSize = 10;
    //     dashboardCamera.depth = 10; // מעל כל המצלמות האחרות
        
    //     // מקם את המצלמה
    //     dashboardCamera.transform.position = new Vector3(0, 0, -10);
    //     dashboardCamera.transform.rotation = Quaternion.identity;
        
    //     Debug.Log($"📊 Dashboard camera setup complete for display {dashboardDisplayTarget}");
        
    //     // הוסף רכיב VehicleDashboard - פשוט!
    //     VehicleDashboard dashboard = dashboardCamera.GetComponent<VehicleDashboard>();
    //     if (dashboard == null)
    //     {
    //         dashboard = dashboardCamera.gameObject.AddComponent<VehicleDashboard>();
    //         dashboard.targetDisplay = dashboardDisplayTarget;
    //     }
    // }
    
    // // פונקציות ציבוריות פשוטות
    // public int GetDisplayCount()
    // {
    //     return availableDisplays.Count;
    // }
    
    // public bool HasMultipleDisplays()
    // {
    //     return availableDisplays.Count > 1;
    // }
    
    // public void SwapDisplays()
    // {
    //     if (!HasMultipleDisplays()) return;
        
    //     // החלף בין המסכים
    //     int tempGameDisplay = gameDisplayTarget;
    //     gameDisplayTarget = dashboardDisplayTarget;
    //     dashboardDisplayTarget = tempGameDisplay;
        
    //     // עדכן את המצלמות
    //     if (mainGameCamera != null)
    //     {
    //         mainGameCamera.targetDisplay = gameDisplayTarget;
    //     }
        
    //     if (dashboardCamera != null)
    //     {
    //         dashboardCamera.targetDisplay = dashboardDisplayTarget;
    //     }
        
    //     Debug.Log($"🔄 Displays swapped! Game→{gameDisplayTarget}, Dashboard→{dashboardDisplayTarget}");
    // }
    
    // public void ToggleDashboardDisplay()
    // {
    //     if (dashboardCamera != null)
    //     {
    //         dashboardCamera.enabled = !dashboardCamera.enabled;
    //         Debug.Log($"📊 Dashboard camera {(dashboardCamera.enabled ? "enabled" : "disabled")}");
    //     }
    // }
    
    // public string GetDisplayInfo()
    // {
    //     string info = $"🖥️ Total Displays: {availableDisplays.Count}\n";
    //     for (int i = 0; i < availableDisplays.Count; i++)
    //     {
    //         Display display = availableDisplays[i];
    //         string usage = "";
    //         if (i == gameDisplayTarget) usage += "[GAME] ";
    //         if (i == dashboardDisplayTarget) usage += "[DASHBOARD] ";
            
    //         info += $"Display {i}: {display.systemWidth}x{display.systemHeight} {usage}{(display.active ? "(Active)" : "(Inactive)")}\n";
    //     }
    //     return info;
    // }
    
    // void Update()
    // {
    //     // בדיקת מקשי קיצור
    //     if (Input.GetKeyDown(KeyCode.F2))
    //     {
    //         ToggleDashboardDisplay();
    //     }
        
    //     if (Input.GetKeyDown(KeyCode.F3) && HasMultipleDisplays())
    //     {
    //         SwapDisplays();
    //     }
        
    //     if (Input.GetKeyDown(KeyCode.F4))
    //     {
    //         Debug.Log(GetDisplayInfo());
    //     }
        
    //     if (Input.GetKeyDown(KeyCode.F8))
    //     {
    //         // איפוס למצב ברירת מחדל
    //         gameDisplayTarget = 1;
    //         dashboardDisplayTarget = 0;
    //         SetupDisplays();
    //         Debug.Log("🔄 Reset to default: Game→Display1, Dashboard→Display0");
    //     }
    // }
    
    // void OnGUI()
    // {
    //     // הצג מידע על המסכים והקצאות
    //     if (Event.current.type == EventType.Repaint)
    //     {
    //         GUI.color = Color.white;
    //         string displayInfo = $"Displays: {availableDisplays.Count}";
    //         if (HasMultipleDisplays())
    //         {
    //             displayInfo += $" | Game→{gameDisplayTarget} | Dashboard→{dashboardDisplayTarget}";
    //         }
            
    //         GUIStyle style = new GUIStyle(GUI.skin.label);
    //         style.fontSize = 10;
    //         GUI.Label(new Rect(10, Screen.height - 30, 500, 20), displayInfo, style);
            
    //         // הוראות
    //         string instructions = "F2: Toggle Dashboard | F3: Swap Displays | F4: Display Info | F8: Reset Default";
    //         GUI.Label(new Rect(10, Screen.height - 50, 600, 20), instructions, style);
    //     }
    // }
}