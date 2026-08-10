// DriverNashHUD
// Attach to any GameObject in the scene (e.g. NashSystem or driver_camera).
// Finds the existing Canvas in the scene and adds the HUD panel as a child,
// so it renders on whatever display the Canvas is already targeting.

using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class DriverNashHUD : MonoBehaviour
{
    [Header("Wiring")]
    public NashCoordinator coordinator;
    public PlatoonManager  platoonManager;

    // The canvas to attach to. If left null, the script finds it automatically
    // by looking for a Canvas whose camera targets Display 1.
    [Header("Canvas target (leave null to auto-detect)")]
    public Canvas targetCanvas;

    Image    _accentLine;
    Image    _authorityFill;
    TMP_Text _modeLabel;
    TMP_Text _gapLabel;
    TMP_Text _authorityPct;
    float    _smoothAlpha;

    // Panel size in canvas pixels (reference 1920x1080)
    const float PW = 340f;
    const float PH = 118f;

    // -------------------------------------------------------------------------
    void Start()
    {
        Canvas canvas = targetCanvas;

        // Auto-find: look for Canvas whose worldCamera targets Display 1
        if (canvas == null)
        {
            foreach (var c in FindObjectsOfType<Canvas>())
            {
                if (c.worldCamera != null && c.worldCamera.targetDisplay == 1)
                { canvas = c; break; }
                if (c.renderMode == RenderMode.ScreenSpaceOverlay)
                {
                    // Screen Space Overlay renders on the main display —
                    // but here the existing Canvas IS on Display 1 because
                    // it's rendered by the driver camera setup.
                    // Just take the first canvas we find as fallback.
                    if (canvas == null) canvas = c;
                }
            }
        }

        if (canvas == null)
        {
            Debug.LogError("[DriverNashHUD] No Canvas found. Assign targetCanvas in Inspector.");
            enabled = false;
            return;
        }

        Debug.Log($"[DriverNashHUD] Attaching to Canvas '{canvas.name}' renderMode={canvas.renderMode}");
        BuildPanel(canvas.transform);
    }

    // -------------------------------------------------------------------------
    void Update()
    {
        if (_modeLabel == null) return;

        float target = 0f;
        if (coordinator != null && coordinator.NashActive)
        {
            float aLong = coordinator.LongLambda / (1f + coordinator.LongLambda);
            float aLat  = coordinator.LatLambda  / (1f + coordinator.LatLambda);
            // Use max: the participant feels whichever axis the system dominates most.
            target = Mathf.Max(aLong, aLat);
        }
        _smoothAlpha = Mathf.Lerp(_smoothAlpha, target, Time.deltaTime * 4f);
        Refresh();
    }

    // -------------------------------------------------------------------------
    void Refresh()
    {
        Color accent = Accent();
        _accentLine.color = accent;
        _modeLabel.text   = ModeText();
        _modeLabel.color  = accent;

        // Bar fills left→right = driver share (1 − system α).
        // Full bar = driver in full control. Empty bar = system in full control.
        float driverShare = 1f - _smoothAlpha;
        float barMaxW = PW - 184f;
        _authorityFill.rectTransform.sizeDelta = new Vector2(barMaxW * driverShare, 14f);
        _authorityFill.color = Color.Lerp(
            new Color(0.25f, 0.55f, 1f),   // blue  — system dominant (low driver share)
            new Color(0.35f, 0.9f, 0.45f), // green — driver dominant (high driver share)
            driverShare);
        int pct = Mathf.RoundToInt(driverShare * 100f);
        _authorityPct.text  = $"{pct}%";
        _authorityPct.color = _authorityFill.color;

        float gap = Gap();
        if (gap < 990f)
        {
            string icon = gap > 15f ? "✓" : gap > 8f ? "▲" : "✕";
            _gapLabel.text  = $"Gap   {gap:F1} m   {icon}";
            _gapLabel.color = gap > 15f ? new Color(0.35f, 0.9f, 0.45f)
                            : gap > 8f  ? new Color(1f, 0.82f, 0.2f)
                                        : new Color(1f, 0.3f, 0.3f);
        }
        else
        {
            _gapLabel.text  = "Gap   —";
            _gapLabel.color = new Color(0.55f, 0.55f, 0.55f);
        }
    }

    // -------------------------------------------------------------------------
    void BuildPanel(Transform canvasTransform)
    {
        // Root panel — anchored to top-left corner of the canvas
        var panel = new GameObject("NashHUD_Panel");
        panel.transform.SetParent(canvasTransform, false);
        var prt = panel.AddComponent<RectTransform>();
        prt.anchorMin        = new Vector2(1, 0);
        prt.anchorMax        = new Vector2(1, 0);
        prt.pivot            = new Vector2(1, 0);
        prt.anchoredPosition = new Vector2(-370, 280); // tweak X/Y in Inspector to align with gauges
        prt.sizeDelta        = new Vector2(PW, PH);

        // background
        var bgImg = panel.AddComponent<Image>();
        bgImg.color = new Color(0, 0, 0, 0.60f);

        // accent line (top 3px)
        var accent = Child<Image>(panel, "Accent");
        _accentLine = accent;
        accent.color = Color.white;
        var art = accent.rectTransform;
        art.anchorMin        = new Vector2(0, 1);
        art.anchorMax        = new Vector2(1, 1);
        art.pivot            = new Vector2(0.5f, 1f);
        art.anchoredPosition = Vector2.zero;
        art.sizeDelta        = new Vector2(0, 3);

        // mode label
        _modeLabel = Lbl(panel, "Mode", "●  MANUAL", 22, true,
            new Vector2(14, -10), new Vector2(PW - 14, 30));

        // "Your control" label
        var sl = Lbl(panel, "SysCtrl", "Your control", 15, false,
            new Vector2(14, -46), new Vector2(110, 18));
        sl.color = new Color(0.6f, 0.6f, 0.6f);

        // authority bar bg
        var bbg = Child<Image>(panel, "BarBg");
        bbg.color = new Color(0.22f, 0.22f, 0.22f);
        TL(bbg.rectTransform, new Vector2(130, -46), new Vector2(PW - 184, 14));

        // authority bar fill — plain Image, width scaled each frame in Refresh()
        _authorityFill       = Child<Image>(panel, "BarFill");
        _authorityFill.color = new Color(0.25f, 0.55f, 1f);
        TL(_authorityFill.rectTransform, new Vector2(130, -46), new Vector2(0, 14));

        // percentage label — right of the bar
        _authorityPct = Lbl(panel, "AuthPct", "0%", 15, true,
            new Vector2(PW - 48, -46), new Vector2(44, 18));
        _authorityPct.alignment = TextAlignmentOptions.MidlineRight;
        _authorityPct.color     = new Color(0.6f, 0.6f, 0.6f);

        // gap label
        _gapLabel = Lbl(panel, "Gap", "Gap   —", 20, false,
            new Vector2(14, -74), new Vector2(PW - 14, 26));
        _gapLabel.color = new Color(0.55f, 0.55f, 0.55f);

        Debug.Log("[DriverNashHUD] Panel built successfully.");
    }

    // ── factory helpers ───────────────────────────────────────────────────────

    T Child<T>(GameObject parent, string name) where T : Component
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent.transform, false);
        return go.AddComponent<T>();
    }

    TMP_Text Lbl(GameObject parent, string name, string text,
                 int size, bool bold, Vector2 pos, Vector2 sz)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent.transform, false);
        var t = go.AddComponent<TextMeshProUGUI>();
        t.text         = text;
        t.fontSize     = size;
        t.fontStyle    = bold ? FontStyles.Bold : FontStyles.Normal;
        t.color        = Color.white;
        t.alignment    = TextAlignmentOptions.MidlineLeft;
        t.overflowMode = TextOverflowModes.Overflow;
        TL(t.rectTransform, pos, sz);
        return t;
    }

    static void TL(RectTransform rt, Vector2 pos, Vector2 sz)
    {
        rt.anchorMin        = new Vector2(0, 1);
        rt.anchorMax        = new Vector2(0, 1);
        rt.pivot            = new Vector2(0, 1);
        rt.anchoredPosition = pos;
        rt.sizeDelta        = sz;
    }

    // ── data helpers ──────────────────────────────────────────────────────────

    string ModeText()
    {
        if (coordinator == null || !coordinator.NashActive) return "●  MANUAL";
        return coordinator.Phase switch
        {
            MergePhase.Approach  => "●  NASH — APPROACH",
            MergePhase.GapSearch => "●  NASH — ALIGNING",
            MergePhase.Merge     => "●  NASH — MERGING",
            MergePhase.Following => "●  NASH — FOLLOWING",
            _                    => "●  NASH ACTIVE"
        };
    }

    Color Accent()
    {
        if (coordinator == null || !coordinator.NashActive)
            return new Color(0.5f, 0.5f, 0.5f);
        return coordinator.Phase switch
        {
            MergePhase.Approach  => new Color(0.5f,  0.5f,  0.5f),
            MergePhase.GapSearch => new Color(1f,    0.75f, 0.15f),
            MergePhase.Merge     => new Color(0.25f, 0.55f, 1f),
            MergePhase.Following => new Color(0.35f, 0.9f,  0.45f),
            _                    => new Color(0.25f, 0.55f, 1f)
        };
    }

    float Gap()
    {
        if (coordinator != null && coordinator.Data.GapToLeader < 990f)
            return coordinator.Data.GapToLeader;
        if (platoonManager != null)
            return platoonManager.GetActualGap(1);
        return 999f;
    }
}
