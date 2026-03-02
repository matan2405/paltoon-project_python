using UnityEngine;

/// <summary>
/// סקריפט לבדיקה וחיקוי של מסכים מרובים
/// </summary>
public class DisplayDebugger : MonoBehaviour
{
    void Start()
    {
        DebugDisplays();
        ForceActivateDisplays();
    }
    
    void DebugDisplays()
    {
        Debug.Log("=== Display Debug Info ===");
        Debug.Log($"Total Displays Found: {Display.displays.Length}");
        
        for (int i = 0; i < Display.displays.Length; i++)
        {
            Display display = Display.displays[i];
            Debug.Log($"Display {i}:");
            Debug.Log($"  - Resolution: {display.systemWidth}x{display.systemHeight}");
            Debug.Log($"  - Active: {display.active}");
            Debug.Log($"  - Rendering Width: {display.renderingWidth}");
            Debug.Log($"  - Rendering Height: {display.renderingHeight}");
        }
        
        Debug.Log("=== End Display Debug ===");
    }
    
    void ForceActivateDisplays()
    {
        // הפעל בכוח את כל המסכים
        for (int i = 0; i < Display.displays.Length; i++)
        {
            Display.displays[i].Activate();
            Debug.Log($"Forced activation of Display {i}");
        }
        
        // חכה ואז בדוק שוב
        Invoke("CheckAfterActivation", 1f);
    }
    
    void CheckAfterActivation()
    {
        Debug.Log("=== After Forced Activation ===");
        for (int i = 0; i < Display.displays.Length; i++)
        {
            Display display = Display.displays[i];
            Debug.Log($"Display {i} - Active: {display.active}");
        }
    }
    
    void Update()
    {
        // מקשי בדיקה
        if (Input.GetKeyDown(KeyCode.F5))
        {
            DebugDisplays();
        }
        
        if (Input.GetKeyDown(KeyCode.F6))
        {
            ForceActivateDisplays();
        }
        
        if (Input.GetKeyDown(KeyCode.F7))
        {
            TestCreateSecondDisplayCamera();
        }
    }
    
    void TestCreateSecondDisplayCamera()
    {
        if (Display.displays.Length < 2)
        {
            Debug.LogWarning("Only one display detected - cannot create second display camera");
            return;
        }
        
        // צור מצלמה למסך השני
        GameObject testCameraGO = new GameObject("Test Second Display Camera");
        Camera testCamera = testCameraGO.AddComponent<Camera>();
        
        testCamera.targetDisplay = 1;
        testCamera.clearFlags = CameraClearFlags.SolidColor;
        testCamera.backgroundColor = Color.red; // רקע אדום לזיהוי
        testCamera.cullingMask = 0; // לא מציג אובייקטים
        
        Debug.Log("Test camera created for display 1 with RED background");
        Debug.Log("If you see red screen on second monitor - displays are working!");
    }
    
    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 12;
        style.normal.textColor = Color.yellow;
        
        GUI.Label(new Rect(10, 100, 300, 20), $"Displays Detected: {Display.displays.Length}", style);
        GUI.Label(new Rect(10, 120, 300, 20), "F5: Debug Displays", style);
        GUI.Label(new Rect(10, 140, 300, 20), "F6: Force Activate", style);
        GUI.Label(new Rect(10, 160, 300, 20), "F7: Test Red Screen", style);
        
        for (int i = 0; i < Display.displays.Length; i++)
        {
            string status = Display.displays[i].active ? "ACTIVE" : "INACTIVE";
            Color color = Display.displays[i].active ? Color.green : Color.red;
            
            GUI.color = color;
            GUI.Label(new Rect(10, 180 + (i * 20), 300, 20), 
                     $"Display {i}: {Display.displays[i].systemWidth}x{Display.displays[i].systemHeight} - {status}", style);
        }
        GUI.color = Color.white;
    }
}