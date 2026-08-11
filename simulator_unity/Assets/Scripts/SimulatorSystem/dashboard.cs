using UnityEngine;

public class VehicleDashboard : MonoBehaviour
{
    [Header("Wiring")]
    public NashCoordinator coordinator;
    public PlatoonManager  platoonManager;
    public KeyCode toggleDashboardKey = KeyCode.F1;

    [Header("Trial Info")]
    public string controlMode   = "Nash-Shared";
    public string trialGeometry = "Join-After";
    public int    trialNumber   = 1;
    public int    totalTrials   = 3;

    [Header("Layout (pixels, adjust in Inspector)")]
    public int dashboardY    = 100;
    public int columnHeight  = 190;
    public int headerHeight  = 28;
    public int margin        = 6;
    public int columnGap     = 6;

    private AdvancedBicycleModel vehicle;
    private bool dashboardVisible = true;

    private GUIStyle titleStyle, labelStyle, headerStyle, bigStyle;
    private bool stylesInitialized = false;

    private const int BH  = 16;
    private const int FS  = 10;
    private const int BFS = 13;

    void Start()
    {
        if (coordinator    == null) coordinator    = FindObjectOfType<NashCoordinator>();
        if (platoonManager == null) platoonManager = FindObjectOfType<PlatoonManager>();

        if (platoonManager != null && platoonManager.egoVehicle != null)
            vehicle = platoonManager.egoVehicle;
        else
            vehicle = FindObjectOfType<AdvancedBicycleModel>();

        if (vehicle == null) { enabled = false; return; }
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleDashboardKey))
            dashboardVisible = !dashboardVisible;
    }

    void OnGUI()
    {
        if (!dashboardVisible || vehicle == null) return;
        if (CameraSwitcher.Instance != null && !CameraSwitcher.Instance.IsThirdPersonActive()) return;
        InitializeStyles();
        DrawDashboard();
    }

    void InitializeStyles()
    {
        if (stylesInitialized) return;
        titleStyle  = new GUIStyle(GUI.skin.label) { fontSize = 14, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
        titleStyle.normal.textColor  = Color.white;
        headerStyle = new GUIStyle(GUI.skin.label) { fontSize = FS, fontStyle = FontStyle.Bold };
        headerStyle.normal.textColor = Color.yellow;
        labelStyle  = new GUIStyle(GUI.skin.label) { fontSize = FS };
        labelStyle.normal.textColor  = Color.white;
        bigStyle    = new GUIStyle(GUI.skin.label) { fontSize = BFS, fontStyle = FontStyle.Bold };
        bigStyle.normal.textColor    = Color.white;
        stylesInitialized = true;
    }

    void DrawDashboard()
    {
        string modeLabel = controlMode switch {
            "Manual"        => "MANUAL",
            "Full-Autonomy" => "FULL-AUTO",
            _               => "NASH-SHARED"
        };

        int sw = Screen.width;
        int cw = (sw - margin * 2 - columnGap * 3) / 4;

        // Header bar
        GUI.Box(new Rect(0, dashboardY, sw, headerHeight), "");
        GUI.Label(new Rect(20, dashboardY + 4, sw - 40, headerHeight - 8),
            $"SUPERVISOR  —  Trial {trialNumber}/{totalTrials}  |  {modeLabel}  |  {trialGeometry}",
            titleStyle);

        int y0 = dashboardY + headerHeight + 2;

        int x1 = margin;
        int x2 = x1 + cw + columnGap;
        int x3 = x2 + cw + columnGap;
        int x4 = x3 + cw + columnGap;

        DrawVehicleColumn(x1, y0, cw, columnHeight);
        DrawNashColumn   (x2, y0, cw, columnHeight);
        DrawSafetyColumn (x3, y0, cw, columnHeight);
        DrawPlatoonColumn(x4, y0, cw, columnHeight);

        DrawStatusBar(y0 + columnHeight + 2);
    }

    // ── Column 1: Vehicle / Phase ─────────────────────────────────────────────
    void DrawVehicleColumn(int x, int y, int cw, int ch)
    {
        GUI.Box(new Rect(x, y, cw, ch), "");
        GUI.Label(new Rect(x+6, y+3, cw-12, BH), "VEHICLE / PHASE", headerStyle);

        float vx_kph = vehicle.GetVx() * 3.6f;
        float ax     = vehicle.GetAx();

        Color spdC = vx_kph > 95f ? Color.red : vx_kph > 85f ? Color.yellow : Color.white;
        L(x, y+20, cw, "Speed:", spdC);
        B(x, y+32, cw, $"{vx_kph:F1} km/h", spdC);
        L(x, y+52, cw, $"Accel: {ax:F2} m/s²");

        string phase  = "—";
        Color  phaseC = new Color(0.5f, 0.5f, 0.5f);
        if (coordinator != null && coordinator.NashActive)
        {
            phase  = coordinator.Phase.ToString().ToUpper();
            phaseC = coordinator.Phase switch {
                MergePhase.Approach  => new Color(0.6f,  0.6f,  0.6f),
                MergePhase.GapSearch => new Color(1f,    0.75f, 0.15f),
                MergePhase.Merge     => new Color(0.27f, 0.6f,  1f),
                MergePhase.Following => new Color(0.27f, 0.85f, 0.4f),
                _                    => Color.white
            };
        }
        L(x, y+68, cw, "Phase:");
        B(x, y+80, cw, phase, phaseC);

        float throttle = 0f, brake = 0f, steer = 0f;
        if (VehicleInputs.Instance != null)
        {
            throttle = VehicleInputs.Instance.ThrottleInput;
            brake    = VehicleInputs.Instance.BrakeInput;
            steer    = VehicleInputs.Instance.SteeringInput;
        }

        int bw = (cw - 16 - 8) / 3;
        L(x, y+100, cw, "Inputs:");
        L(x+6,          y+114, bw, $"Gas {throttle*100:F0}%", new Color(0.4f,0.9f,0.4f));
        L(x+6+bw+4,     y+114, bw, $"Brk {brake*100:F0}%",   new Color(0.9f,0.4f,0.4f));
        L(x+6+2*(bw+4), y+114, bw, $"Str {steer:F2}",        new Color(0.5f,0.8f,1f));
        DrawBar(x+6,           y+128, bw, 6, throttle, new Color(0.3f,0.85f,0.3f));
        DrawBar(x+6+bw+4,      y+128, bw, 6, brake,    new Color(0.85f,0.3f,0.3f));
        DrawSteeringBar(x+6+2*(bw+4), y+128, bw, 6, steer);
    }

    // ── Column 2: Nash / Authority ────────────────────────────────────────────
    void DrawNashColumn(int x, int y, int cw, int ch)
    {
        GUI.Box(new Rect(x, y, cw, ch), "");
        GUI.Label(new Rect(x+6, y+3, cw-12, BH), "NASH / AUTHORITY", headerStyle);

        if (coordinator == null || !coordinator.NashActive)
        {
            L(x, y+22, cw, "Nash  INACTIVE", new Color(0.5f, 0.5f, 0.5f));
            return;
        }

        float lLong  = coordinator.LongLambda;
        float lLat   = coordinator.LatLambda;
        float aLong  = lLong / (1f + lLong);
        float aLat   = lLat  / (1f + lLat);
        float aMax   = Mathf.Max(aLong, aLat);
        float drvPct = (1f - aMax) * 100f;
        float sysPct = aMax * 100f;

        Color authC = sysPct > 70f ? new Color(0.27f,0.6f,1f)
                    : sysPct > 40f ? new Color(1f,0.82f,0.2f)
                                   : new Color(0.27f,0.85f,0.4f);

        L(x, y+20, cw, "Your control:");
        B(x, y+32, cw, $"{drvPct:F0}%", Color.white);
        L(x, y+52, cw, $"System: {sysPct:F0}%", authC);
        DrawFilledBar(x+6, y+66, cw-12, 7, aMax,
            new Color(0.27f,0.6f,1f), new Color(0.27f,0.85f,0.4f));

        L(x, y+78,  cw, $"λ_long {lLong:F2}  α {aLong:F2}");
        L(x, y+92,  cw, $"λ_lat  {lLat:F2}  α {aLat:F2}");

        float conflict = coordinator.Data.ConflictMav;
        Color cflC = conflict > 0.5f ? Color.red : conflict > 0.2f ? Color.yellow : new Color(0.27f,0.85f,0.4f);
        L(x, y+108, cw, $"Conflict MA: {conflict:F3}", cflC);

        float lnSync = coordinator.Data.LnSync;
        L(x, y+122, cw, $"Merge progress: {lnSync*100f:F0}%", new Color(0.27f,0.85f,0.4f));
        DrawFilledBar(x+6, y+136, cw-12, 6, Mathf.Clamp01(lnSync),
            new Color(0.27f,0.85f,0.4f), new Color(1f,0.75f,0.15f));
    }

    // ── Column 3: Safety ─────────────────────────────────────────────────────
    void DrawSafetyColumn(int x, int y, int cw, int ch)
    {
        GUI.Box(new Rect(x, y, cw, ch), "");
        GUI.Label(new Rect(x+6, y+3, cw-12, BH), "SAFETY", headerStyle);

        float ttc    = coordinator != null ? coordinator.Data.TTC    : 999f;
        float thw    = coordinator != null ? coordinator.Data.THW    : 999f;
        float velErr = coordinator != null ? coordinator.Data.VelErr : 0f;
        float yErr   = coordinator != null ? coordinator.Data.YErr   : 0f;

        string ttcStr;
        Color  ttcC;
        if      (ttc > 90f) { ttcStr = "—";               ttcC = new Color(0.5f,0.5f,0.5f); }
        else if (ttc < 2f)  { ttcStr = $"{ttc:F1} s  !!"; ttcC = Color.red; }
        else if (ttc < 4f)  { ttcStr = $"{ttc:F1} s";     ttcC = Color.yellow; }
        else                { ttcStr = $"{ttc:F1} s";     ttcC = new Color(0.27f,0.85f,0.4f); }

        Color thwC = thw < 0.8f ? Color.red : thw < 1.5f ? Color.yellow : new Color(0.27f,0.85f,0.4f);
        Color rvC  = Mathf.Abs(velErr) > 5f ? Color.red : Mathf.Abs(velErr) > 2f ? Color.yellow : Color.white;
        Color yC   = Mathf.Abs(yErr)   > 1f ? Color.red : Mathf.Abs(yErr)   > 0.3f ? Color.yellow : Color.white;

        L(x, y+20, cw, "TTC:");
        B(x, y+32, cw, ttcStr, ttcC);
        L(x, y+52, cw, "THW:");
        L(x, y+64, cw, thw < 990f ? $"{thw:F2} s" : "—", thwC);
        L(x, y+80, cw, "Vel error:");
        L(x, y+92, cw, $"{velErr:+0.00;-0.00} m/s", rvC);
        L(x, y+108, cw, "Lat error:");
        L(x, y+120, cw, $"{yErr:+0.00;-0.00} m", yC);

        if (coordinator != null && coordinator.Data.SafetyOverride)
            L(x, y+138, cw, "SAFETY OVERRIDE", Color.red);
        if (ttc < 2f && ttc > 0f)
            L(x, y+154, cw, "DANGER  TTC < 2s", Color.red);
    }

    // ── Column 4: Platoon ─────────────────────────────────────────────────────
    void DrawPlatoonColumn(int x, int y, int cw, int ch)
    {
        GUI.Box(new Rect(x, y, cw, ch), "");
        GUI.Label(new Rect(x+6, y+3, cw-12, BH), "PLATOON", headerStyle);

        if (platoonManager == null)
        {
            L(x, y+22, cw, "PlatoonManager not wired", new Color(0.5f,0.5f,0.5f));
            return;
        }

        float targetV = platoonManager.GetTargetVelocity();
        L(x, y+20, cw, $"Target: {targetV*3.6f:F1} km/h");

        var vehicles = platoonManager.GetPlatoonVehicles();
        if (vehicles == null || vehicles.Length == 0)
        {
            L(x, y+36, cw, "No platoon vehicles", new Color(0.5f,0.5f,0.5f));
            return;
        }

        int ly = y + 36;
        for (int i = 0; i < vehicles.Length; i++)
        {
            if (vehicles[i] == null || ly > y + ch - BH - 4) break;
            float spd  = vehicles[i].GetVx() * 3.6f;
            string lbl = i == 0 ? $"Leader  {spd:F1} km/h" : $"Car {i+1}    {spd:F1} km/h";
            L(x, ly, cw, lbl);                                            ly += BH;
            if (i > 0 && ly <= y + ch - BH - 4)
            {
                float ag = platoonManager.GetActualGap(i);
                float dg = platoonManager.GetDesiredGap(i);
                Color gc = ag < 5f ? Color.red : ag < 10f ? Color.yellow : Color.white;
                L(x, ly, cw, $"  gap {ag:F1} / des {dg:F1} m", gc);      ly += BH;
            }
        }

        if (platoonManager.egoVehicle != null)
        {
            bool  inPlatoon = platoonManager._egoInPlatoon;
            Color egoC      = inPlatoon ? new Color(0.27f,0.85f,0.4f) : Color.yellow;
            L(x, y + ch - BH - 2, cw, $"Ego: [{(inPlatoon ? "IN PLATOON" : "OUT")}]", egoC);
        }
    }

    // ── Status bar ────────────────────────────────────────────────────────────
    void DrawStatusBar(int by)
    {
        GUI.Box(new Rect(0, by, Screen.width, 22), "");
        GUIStyle s = new GUIStyle(labelStyle) { alignment = TextAnchor.MiddleCenter };
        s.normal.textColor = new Color(0.6f, 0.6f, 0.6f);
        GUI.Label(new Rect(0, by + 3, Screen.width, 16),
            $"F1 Toggle  |  FPS {1f / Time.deltaTime:F0}", s);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    void L(int x, int y, int cw, string text) =>
        GUI.Label(new Rect(x + 6, y, cw - 12, BH), text, labelStyle);

    void L(int x, int y, int cw, string text, Color c)
    {
        GUIStyle s = new GUIStyle(labelStyle);
        s.normal.textColor = c;
        GUI.Label(new Rect(x + 6, y, cw - 12, BH), text, s);
    }

    void B(int x, int y, int cw, string text, Color c)
    {
        GUIStyle s = new GUIStyle(bigStyle);
        s.normal.textColor = c;
        GUI.Label(new Rect(x + 6, y, cw - 12, BH + 6), text, s);
    }

    void DrawFilledBar(int x, int y, int w, int h, float fill, Color fullC, Color emptyC)
    {
        GUI.color = new Color(0.15f, 0.15f, 0.15f);
        GUI.DrawTexture(new Rect(x, y, w, h), Texture2D.whiteTexture);
        GUI.color = Color.Lerp(emptyC, fullC, fill);
        if (fill > 0f)
            GUI.DrawTexture(new Rect(x, y, w * fill, h), Texture2D.whiteTexture);
        GUI.color = Color.white;
    }

    void DrawBar(int x, int y, int w, int h, float value, Color color)
    {
        GUI.color = new Color(0.15f, 0.15f, 0.15f);
        GUI.DrawTexture(new Rect(x, y, w, h), Texture2D.whiteTexture);
        GUI.color = color;
        if (value > 0f)
            GUI.DrawTexture(new Rect(x, y, w * Mathf.Clamp01(value), h), Texture2D.whiteTexture);
        GUI.color = Color.white;
    }

    void DrawSteeringBar(int x, int y, int w, int h, float steer)
    {
        GUI.color = new Color(0.15f, 0.15f, 0.15f);
        GUI.DrawTexture(new Rect(x, y, w, h), Texture2D.whiteTexture);
        int cx = x + w / 2;
        GUI.color = new Color(0.4f, 0.4f, 0.4f);
        GUI.DrawTexture(new Rect(cx - 1, y, 2, h), Texture2D.whiteTexture);
        int sx = Mathf.Clamp(cx + (int)(steer * w / 2), x + 2, x + w - 2);
        GUI.color = steer > 0.05f ? Color.cyan : steer < -0.05f ? Color.magenta : Color.green;
        GUI.DrawTexture(new Rect(sx - 2, y - 1, 5, h + 2), Texture2D.whiteTexture);
        GUI.color = Color.white;
    }
}
