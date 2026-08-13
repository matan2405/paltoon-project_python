// DriverNashHUD — Driver instrument cluster for platoon assist.
//
// MANUAL mode:  shows only gauges + "MANUAL — press J" message, no platoon view.
// NASH mode:    shows gauges + proximity row (3 vehicle shapes + gaps).
//
// Layout (1080p reference, bottom-centre, 920×210 px):
//
//  ┌────────────┬──────────────────────────────────────┬────────────┐
//  │ RPM gauge  │   [LEADER]  ──  [YOU]  ──  [FOLLWR]  │ SPD gauge  │
//  │  (ring)    │    48 m                   52 m        │  (ring)    │
//  │  [GEAR]    │  — or —                               │  km/h      │
//  │  [rpm num] │  MANUAL / AUTONOMOUS message          │            │
//  ├────────────┴──────────────────────────────────────┴────────────┤
//  │  STATUS TEXT (phase)                 [wheel dial]  label       │
//  └────────────────────────────────────────────────────────────────┘

using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class DriverNashHUD : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────────────────
    [Header("Data sources")]
    public NashCoordinator coordinator;
    public PlatoonManager  platoonManager;

    [Header("World-Space HUD anchor (drag HUD_Anchor GO here)")]
    public GameObject hudAnchorGO;   // HUD_Anchor child of PlayerCar — Canvas is added to this GO
    public float      hudScale = 0.001f; // metres per pixel

    [Header("Panel position (overlay fallback only)")]
    public Vector2 panelOffset = new Vector2(440f, 190f);

    [Header("HMI Sprites — Right/")]
    public Sprite spriteGaugeRing;   // 8PartCircle.tga
    public Sprite spriteFalloff;     // FalloffCircle.tga

    [Header("Speed Line Texture")]
    public Texture speedLineTexture; // SpeedLine.tga

    [Header("Audio")]
    public AudioClip beepClip;
    [Range(1f,30f)]  public float gapWarnThreshold = 10f;
    [Range(0.5f,10f)] public float beepCooldown    = 2f;

    // ── Layout ───────────────────────────────────────────────────────────────
    // Layout: [  CENTRE (proximity/manual)  |  RPM  ]
    //         [                             |  SPD  ]
    //         [       STATUS BAR                    ]
    const float PH      =  310f;
    const float StatusH =   50f;
    const float ProxH   = PH - StatusH;        // 260 px
    const float GaugeW  =  240f;               // right column (gauges stacked)
    const float GaugeH  = ProxH / 2f;          // each gauge cell = 130 px
    const float GaugeR  =  56f;               // must be < GaugeH/2-4 = 61 to avoid overlap
    const float CentreW =  340f;
    const float PW      = CentreW + GaugeW;    // 580 px total
    const int   NSeg    =  14;
    const int   BarSegs =   8;

    // ── Runtime refs ─────────────────────────────────────────────────────────
    Image[]  _rpmSegs;
    TMP_Text _rpmNum;
    TMP_Text _gearLbl;

    Image[]  _spdSegs;
    TMP_Text _spdNum;

    // Proximity (Nash active)
    GameObject _proxRoot;
    Image      _leaderShape;
    Image      _selfShape;
    Image      _followerShape;
    TMP_Text   _leaderGapLbl;
    TMP_Text   _followerGapLbl;
    Image      _dangerLineL;   // line between leader and YOU
    Image      _dangerLineF;   // line between YOU and follower

    // Manual-mode message (Nash inactive)
    GameObject _manualRoot;

    // Speed lines
    RawImage _speedLineL;
    RawImage _speedLineR;
    float    _speedLineOff;
    Material _speedLineMat;

    // Status bar
    Image    _statusBg;
    TMP_Text _statusLbl;

    // Authority bar
    Image[]  _barSegs;
    Image    _barThumb;
    TMP_Text _authLbl;

    RectTransform _panelRt;

    // Smoothing / audio
    AudioSource _audio;
    float _lastBeep  = -99f;
    float _smAuth;
    float _smSpeed;
    float _smRpm;

    // =========================================================================
    void Start()
    {
        // Guard: only one instance may build the panel
        var all = FindObjectsOfType<DriverNashHUD>();
        foreach (var other in all)
        {
            if (other != this && other._panelRt != null)
            {
                Debug.LogWarning("[DriverNashHUD] Duplicate instance on " + gameObject.name + " — disabling.");
                enabled = false;
                return;
            }
        }

        if (hudAnchorGO != null)
        {
            // Add Canvas directly to the HUD_Anchor GO so it moves with the car
            var canvas = hudAnchorGO.AddComponent<Canvas>();
            canvas.renderMode   = RenderMode.WorldSpace;
            canvas.sortingOrder = 10;

            hudAnchorGO.AddComponent<GraphicRaycaster>();

            // Scale the RectTransform so 1px = hudScale metres, compensating for parent scale
            var rt = hudAnchorGO.GetComponent<RectTransform>();
            if (rt == null) rt = hudAnchorGO.AddComponent<RectTransform>();
            rt.sizeDelta = new Vector2(PW, PH);
            rt.pivot     = new Vector2(0.5f, 0.5f);

            // Neutralise parent scale so hudScale is absolute
            var ps = hudAnchorGO.transform.lossyScale;
            hudAnchorGO.transform.localScale = new Vector3(
                hudScale / ps.x,
                hudScale / ps.y,
                hudScale / ps.z);

            BuildPanel(hudAnchorGO.transform);

            // Panel fills the canvas rect exactly
            _panelRt.anchorMin = Vector2.zero;
            _panelRt.anchorMax = Vector2.one;
            _panelRt.offsetMin = Vector2.zero;
            _panelRt.offsetMax = Vector2.zero;
        }
        else
        {
            // Fallback: overlay on the main canvas
            Canvas ovCanvas = null;
            foreach (var c in FindObjectsOfType<Canvas>())
                if (c.renderMode == RenderMode.ScreenSpaceOverlay && c.transform.parent == null)
                { ovCanvas = c; break; }
            if (ovCanvas == null) { enabled = false; return; }

            var existing = ovCanvas.transform.Find("NashHUD_Root");
            if (existing != null) Destroy(existing.gameObject);

            var scaler = ovCanvas.GetComponent<CanvasScaler>();
            if (scaler != null) scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;

            BuildPanel(ovCanvas.transform);
        }

        _audio = GetComponent<AudioSource>() ?? gameObject.AddComponent<AudioSource>();
        _audio.playOnAwake = false; _audio.spatialBlend = 0f;
    }

    // =========================================================================
    void Update()
    {
        if (_statusLbl == null) return;

        if (_panelRt != null)
            _panelRt.anchoredPosition = panelOffset;

        var ego = platoonManager != null ? platoonManager.egoVehicle : null;

        float vxMs  = ego != null ? ego.GetVx()        : 0f;
        float vxKmh = vxMs * 3.6f;
        float rpm   = ego != null ? ego.GetEngineRpm() : 0f;
        int   gear  = ego != null ? ego.GetCurrentGear() : 1;

        _smSpeed = Mathf.Lerp(_smSpeed, vxKmh, Time.deltaTime * 6f);
        _smRpm   = Mathf.Lerp(_smRpm,   rpm,   Time.deltaTime * 6f);

        float tAuth = 0f;
        if (coordinator != null && coordinator.NashActive)
        {
            float aL  = coordinator.LongLambda / (1f + coordinator.LongLambda);
            float aLa = coordinator.LatLambda  / (1f + coordinator.LatLambda);
            tAuth = Mathf.Max(aL, aLa);
        }
        _smAuth = Mathf.Lerp(_smAuth, tAuth, Time.deltaTime * 4f);

        RefreshGauges(_smSpeed, _smRpm, gear);
        RefreshCentre(vxMs);
        RefreshStatus();
        RefreshWheelDial();
        RefreshSpeedLines(vxMs);
        MaybeBeep();
    }

    // =========================================================================
    // Gauge refresh
    // =========================================================================

    void RefreshGauges(float kmh, float rpm, int gear)
    {
        // RPM  0–7000
        float tRpm = Mathf.Clamp01(rpm / 7000f);
        Color rCol = Color.Lerp(new Color(0.2f, 0.5f, 1f), new Color(1f, 0.3f, 0.1f), tRpm);
        LitSegs(_rpmSegs, tRpm, rCol);
        _rpmNum.text  = $"{Mathf.RoundToInt(rpm)}";
        _rpmNum.color = rCol;
        _gearLbl.text = gear.ToString();

        // Speed  0–200 km/h
        float tSpd = Mathf.Clamp01(kmh / 200f);
        Color sCol = Color.Lerp(new Color(0.15f, 0.85f, 0.4f), new Color(1f, 0.25f, 0.15f), tSpd);
        LitSegs(_spdSegs, tSpd, sCol);
        _spdNum.text  = $"{Mathf.RoundToInt(kmh)}";
        _spdNum.color = sCol;
    }

    void LitSegs(Image[] segs, float t, Color litCol)
    {
        if (segs == null) return;
        float lit = t * segs.Length;
        var dim = new Color(0.10f, 0.12f, 0.16f);
        for (int i = 0; i < segs.Length; i++)
            segs[i].color = Color.Lerp(dim, litCol, Mathf.Clamp01(lit - i));
    }

    // =========================================================================
    // Centre column
    // =========================================================================

    void RefreshCentre(float vxMs)
    {
        bool nash = coordinator != null && coordinator.NashActive;

        _proxRoot.SetActive(nash);
        _manualRoot.SetActive(!nash);

        if (!nash) return;

        float lgap = LeaderGap();
        float fgap = FollowerGap();
        bool  hasL = lgap < 990f;
        bool  hasF = fgap < 990f;

        _leaderShape.enabled    = hasL;
        _leaderGapLbl.enabled   = hasL;
        _followerShape.enabled  = hasF;
        _followerGapLbl.enabled = hasF;

        // Vertical slide: gap label tracks above/below the car icon
        float cx  = CentreW * 0.5f;
        float cyT = -ProxH * 0.20f;
        float cyM = -ProxH * 0.50f;
        float cyB = -ProxH * 0.80f;

        if (hasL)
        {
            float slide = Mathf.Lerp(0f, ProxH * 0.18f, 1f - Mathf.Clamp01(lgap / 35f));
            float y = Mathf.Clamp(cyT - slide, cyM + 36f, cyT);
            SetCA(_leaderShape.rectTransform,  new Vector2(cx, y));
            SetCA(_leaderGapLbl.rectTransform, new Vector2(cx, y + 32f));
            _leaderShape.color  = GapColor(lgap);
            _leaderGapLbl.text  = $"{lgap:F0} m";
            _leaderGapLbl.color = GapColor(lgap);
        }
        if (hasF)
        {
            float slide = Mathf.Lerp(0f, ProxH * 0.18f, 1f - Mathf.Clamp01(fgap / 35f));
            float y = Mathf.Clamp(cyB + slide, cyB, cyM - 36f);
            SetCA(_followerShape.rectTransform,  new Vector2(cx, y));
            SetCA(_followerGapLbl.rectTransform, new Vector2(cx, y - 30f));
            _followerShape.color  = GapColor(fgap);
            _followerGapLbl.text  = $"{fgap:F0} m";
            _followerGapLbl.color = GapColor(fgap);
        }

        // Self icon pulse
        bool overr = coordinator.Phase != MergePhase.Approach;
        float pulse = overr ? 0.85f + 0.15f * Mathf.Sin(Time.time * 4f) : 1f;
        _selfShape.color = new Color(1f, 1f, 1f, pulse);

        // Danger lines — colour by TTC from coordinator
        float ttc = coordinator != null ? coordinator.Data.TTC : 999f;
        RefreshDangerLine(_dangerLineL, ttc, hasL);
        RefreshDangerLine(_dangerLineF, 999f, hasF);  // follower TTC not tracked, use safe
    }

    void RefreshDangerLine(Image line, float ttc, bool active)
    {
        if (line == null) return;
        if (!active) { line.color = new Color(0.15f, 0.15f, 0.15f, 0.3f); return; }

        Color col;
        float flash = 1f;
        if (ttc > 5f)
        {
            col = new Color(0.2f, 0.85f, 0.35f, 0.7f);          // green — safe
        }
        else if (ttc > 2f)
        {
            float t = 1f - (ttc - 2f) / 3f;
            col = Color.Lerp(new Color(0.2f,0.85f,0.35f,0.7f),
                             new Color(1f,0.75f,0.1f,0.85f), t); // yellow — caution
        }
        else
        {
            col = new Color(1f, 0.2f, 0.2f, 0.9f);              // red — danger
            flash = 0.6f + 0.4f * Mathf.Abs(Mathf.Sin(Time.time * 6f)); // flash
        }
        col.a *= flash;
        line.color = col;
        // Thicken line as danger increases
        var rt = line.rectTransform;
        float w = ttc > 5f ? 3f : ttc > 2f ? 5f : 7f;
        rt.sizeDelta = new Vector2(w, rt.sizeDelta.y);
    }

    // =========================================================================
    // Status + wheel dial
    // =========================================================================

    void RefreshStatus()
    {
        Color accent = AccentColor();
        _statusBg.color = new Color(accent.r * 0.15f, accent.g * 0.15f, accent.b * 0.15f, 0.95f);
        _statusLbl.text  = StatusText();
        _statusLbl.color = accent;
    }

    void RefreshWheelDial()
    {
        if (_barSegs == null) return;

        // ds = driver share [0=system, 1=driver]
        float ds  = 1f - _smAuth;
        // filled segments from RIGHT (driver side)
        float lit = ds * BarSegs;
        Color sysCol    = new Color(0.25f, 0.55f, 1f);   // blue  = system
        Color drvCol    = new Color(0.3f,  0.9f,  0.4f); // green = driver
        Color dimCol    = new Color(0.12f, 0.14f, 0.18f);
        for (int i = 0; i < BarSegs; i++)
        {
            // i=0 leftmost (system), i=BarSegs-1 rightmost (driver)
            float t = (float)i / (BarSegs - 1);
            Color fullCol = Color.Lerp(sysCol, drvCol, t);
            float fill = Mathf.Clamp01(lit - (BarSegs - 1 - i));
            _barSegs[i].color = Color.Lerp(dimCol, fullCol, fill);
        }
        // thumb position
        if (_barThumb != null)
        {
            float thumbX = Mathf.Lerp(0f, 1f, ds);
            _barThumb.rectTransform.anchorMin = new Vector2(thumbX, 0f);
            _barThumb.rectTransform.anchorMax = new Vector2(thumbX, 1f);
            _barThumb.color = Color.Lerp(sysCol, drvCol, ds);
        }

        bool nash = coordinator != null && coordinator.NashActive;
        if (!nash)
        { _authLbl.text = "Manual";        _authLbl.color = new Color(0.45f, 0.45f, 0.45f); }
        else if (ds > 0.75f)
        { _authLbl.text = "You're driving"; _authLbl.color = drvCol; }
        else if (ds > 0.4f)
        { _authLbl.text = "Shared";         _authLbl.color = new Color(1f, 0.82f, 0.2f); }
        else
        { _authLbl.text = "System";         _authLbl.color = sysCol; }
    }

    void RefreshSpeedLines(float vxMs)
    {
        if (_speedLineMat == null) return;
        _speedLineOff += vxMs * Time.deltaTime * 0.04f;
        float tiling = Mathf.SmoothStep(3f, 1f, vxMs / 50f);
        _speedLineMat.SetTextureOffset("_MainTex", new Vector2(0f, _speedLineOff));
        _speedLineMat.SetTextureScale ("_MainTex", new Vector2(1f, tiling));
        float alpha = Mathf.Clamp01(vxMs / 8.33f) * 0.5f;
        var c = new Color(1f, 1f, 1f, alpha);
        if (_speedLineL != null) _speedLineL.color = c;
        if (_speedLineR != null) _speedLineR.color = c;
    }

    void MaybeBeep()
    {
        if (beepClip == null || _audio == null) return;
        if (coordinator == null || !coordinator.NashActive) return;
        float gap = LeaderGap();
        if (gap >= 990f) return;
        if (gap < gapWarnThreshold && Time.time - _lastBeep > beepCooldown)
        {
            _audio.pitch = Mathf.Lerp(2f, 0.8f, gap / gapWarnThreshold);
            _audio.PlayOneShot(beepClip);
            _lastBeep = Time.time;
        }
    }

    // =========================================================================
    // Data helpers
    // =========================================================================

    float LeaderGap()
    {
        // If ego is the platoon leader (first vehicle), there is no real leader to show
        if (IsEgoLeader()) return 999f;
        if (coordinator != null && coordinator.Data.GapToLeader < 990f)
            return coordinator.Data.GapToLeader;
        return 999f;
    }

    bool IsEgoLeader()
    {
        var platoon = platoonManager != null ? platoonManager.GetPlatoonVehicles() : null;
        var ego     = platoonManager != null ? platoonManager.egoVehicle : null;
        if (platoon == null || ego == null) return false;
        // Ego is leader if no vehicle in the platoon is ahead of it
        float egoZ = ego.GetPosition().z;
        foreach (var v in platoon)
        {
            if (v == null || v == ego) continue;
            if (v.GetPosition().z > egoZ) return false; // someone is ahead
        }
        return true;
    }

    float FollowerGap()
    {
        var platoon = platoonManager != null ? platoonManager.GetPlatoonVehicles() : null;
        var ego     = platoonManager != null ? platoonManager.egoVehicle : null;
        if (platoon == null || ego == null) return 999f;
        float egoZ = ego.GetPosition().z, egoH = ego.GetLength() * 0.5f, min = 999f;
        foreach (var v in platoon)
        {
            if (v == null || v == ego) continue;
            float d = egoZ - v.GetPosition().z;
            if (d <= 0f) continue;
            float g = d - egoH - v.GetLength() * 0.5f;
            if (g < min) min = g;
        }
        return min;
    }

    bool       NashOn => coordinator != null && coordinator.NashActive;
    MergePhase Phase  => coordinator != null ? coordinator.Phase : MergePhase.Approach;

    string StatusText()
    {
        if (!NashOn) return "  MANUAL MODE  —  press J to join platoon";
        return Phase switch
        {
            MergePhase.Approach  => "  APPROACH — monitoring platoon",
            MergePhase.GapSearch => "  FINDING SAFE SLOT...",
            MergePhase.Merge     => "  MERGING INTO PLATOON",
            MergePhase.Following => "  PLATOON JOINED",
            _                    => "  ACTIVE"
        };
    }

    Color AccentColor()
    {
        if (!NashOn) return new Color(0.35f, 0.35f, 0.35f);
        return Phase switch
        {
            MergePhase.Approach  => new Color(0.35f, 0.35f, 0.35f),
            MergePhase.GapSearch => new Color(1f,    0.75f, 0.15f),
            MergePhase.Merge     => new Color(0.25f, 0.55f, 1f),
            MergePhase.Following => new Color(0.3f,  0.9f,  0.4f),
            _                    => new Color(0.25f, 0.55f, 1f)
        };
    }

    static Color GapColor(float g) =>
        g > 15f ? new Color(0.3f, 0.9f, 0.4f)
      : g >  8f ? new Color(1f,   0.82f, 0.2f)
                : new Color(1f,   0.28f, 0.28f);

    // =========================================================================
    // Panel builder
    // =========================================================================

    void BuildPanel(Transform canvasRoot)
    {
        var root = NewGO("NashHUD_Root", canvasRoot);
        _panelRt  = root.AddComponent<RectTransform>();
        _panelRt.anchorMin        = new Vector2(0.5f, 0f);
        _panelRt.anchorMax        = new Vector2(0.5f, 0f);
        _panelRt.pivot            = new Vector2(0.5f, 0f);
        _panelRt.anchoredPosition = panelOffset;
        _panelRt.sizeDelta        = new Vector2(PW, PH);
        var rrt = _panelRt;

        var bg = root.AddComponent<Image>();
        bg.color = new Color(0.03f, 0.04f, 0.06f, 0.92f);

        // Top accent line
        var acc = ChildImg(root, "Accent");
        var art = acc.rectTransform;
        art.anchorMin = new Vector2(0,1); art.anchorMax = new Vector2(1,1);
        art.pivot = new Vector2(0.5f,1); art.anchoredPosition = Vector2.zero;
        art.sizeDelta = new Vector2(0,2);
        acc.color = new Color(0.3f, 0.6f, 1f, 0.5f);

        // Vertical divider between centre and gauge column
        var div = ChildImg(root, "Div");
        TL(div.rectTransform, new Vector2(CentreW, -2f), new Vector2(1f, ProxH - 4f));
        div.color = new Color(1f, 1f, 1f, 0.07f);

        // Horizontal divider between RPM and Speed gauges
        var hdiv = ChildImg(root, "HDiv");
        TL(hdiv.rectTransform, new Vector2(CentreW, -GaugeH), new Vector2(GaugeW, 1f));
        hdiv.color = new Color(1f, 1f, 1f, 0.07f);

        BuildRpmCell(root);
        BuildSpdCell(root);
        BuildCentreCol(root);
        BuildStatusBar(root);
    }

    // ── RPM cell: top-right ───────────────────────────────────────────────────
    void BuildRpmCell(GameObject root)
    {
        float cx = CentreW + GaugeW * 0.5f;
        float cy = -(GaugeH * 0.5f);

        BuildGaugeRing(root, "Rpm", cx, cy, new Color(0.15f, 0.2f, 0.55f, 0.25f));
        _rpmSegs = BuildArcSegs(root, "Rpm", cx, cy);

        _rpmNum = ChildTxt(root, "RpmNum", "0", 22, true);
        _rpmNum.alignment = TextAlignmentOptions.Center;
        _rpmNum.color = new Color(0.4f, 0.6f, 1f);
        CA(_rpmNum.rectTransform, new Vector2(cx, cy + 6f), new Vector2(95f, 28f));

        _gearLbl = ChildTxt(root, "Gear", "1", 12, false);
        _gearLbl.alignment = TextAlignmentOptions.Center;
        _gearLbl.color = new Color(0.65f, 0.65f, 0.65f);
        CA(_gearLbl.rectTransform, new Vector2(cx, cy - 16f), new Vector2(46f, 16f));

        var cap = ChildTxt(root, "RpmCap", "RPM", 14, true);
        cap.alignment = TextAlignmentOptions.Center;
        cap.color = new Color(0.55f, 0.55f, 0.55f);
        CA(cap.rectTransform, new Vector2(cx, cy - GaugeH * 0.5f + 10f), new Vector2(60f, 18f));
    }

    // ── Speed cell: bottom-right ──────────────────────────────────────────────
    void BuildSpdCell(GameObject root)
    {
        float cx = CentreW + GaugeW * 0.5f;
        float cy = -(GaugeH + GaugeH * 0.5f);

        BuildGaugeRing(root, "Spd", cx, cy, new Color(0.1f, 0.45f, 0.15f, 0.25f));
        _spdSegs = BuildArcSegs(root, "Spd", cx, cy);

        _spdNum = ChildTxt(root, "SpdNum", "0", 22, true);
        _spdNum.alignment = TextAlignmentOptions.Center;
        _spdNum.color = new Color(0.15f, 0.85f, 0.4f);
        CA(_spdNum.rectTransform, new Vector2(cx, cy + 6f), new Vector2(95f, 28f));

        var cap = ChildTxt(root, "SpdCap", "km/h", 14, true);
        cap.alignment = TextAlignmentOptions.Center;
        cap.color = new Color(0.55f, 0.55f, 0.55f);
        CA(cap.rectTransform, new Vector2(cx, cy - GaugeH * 0.5f + 10f), new Vector2(60f, 18f));
    }

    void BuildGaugeRing(GameObject root, string n, float cx, float cy, Color glowCol)
    {
        if (spriteGaugeRing != null)
        {
            var ring = ChildImg(root, n + "Ring");
            ring.sprite = spriteGaugeRing;
            ring.preserveAspect = true;
            ring.color = new Color(0.08f, 0.10f, 0.14f);
            CA(ring.rectTransform, new Vector2(cx, cy), new Vector2(GaugeR * 2f, GaugeR * 2f));
        }
        if (spriteFalloff != null)
        {
            var fo = ChildImg(root, n + "Glow");
            fo.sprite = spriteFalloff;
            fo.preserveAspect = true;
            fo.color = glowCol;
            CA(fo.rectTransform, new Vector2(cx, cy), new Vector2(GaugeR * 2.3f, GaugeR * 2.3f));
        }
    }

    // 14 arc segments, spread -215° → +35° (clockwise from lower-left)
    Image[] BuildArcSegs(GameObject root, string prefix, float cx, float cy)
    {
        var segs = new Image[NSeg];
        float from = -215f, to = 35f;
        float step = (to - from) / (NSeg - 1);
        for (int i = 0; i < NSeg; i++)
        {
            float deg = from + i * step;
            float rad = deg * Mathf.Deg2Rad;
            var seg = ChildImg(root, $"{prefix}S{i}");
            seg.color = new Color(0.10f, 0.12f, 0.16f);
            var rt = seg.rectTransform;
            rt.anchorMin = new Vector2(0,1); rt.anchorMax = new Vector2(0,1);
            rt.pivot     = new Vector2(0.5f, 0f);
            rt.sizeDelta = new Vector2(4f, GaugeR * 0.4f);
            rt.anchoredPosition = new Vector2(cx + Mathf.Cos(rad) * (GaugeR - 5f),
                                              cy + Mathf.Sin(rad) * (GaugeR - 5f));
            rt.localRotation = Quaternion.Euler(0f, 0f, deg + 90f);
            segs[i] = seg;
        }
        return segs;
    }

    // ── Centre column ─────────────────────────────────────────────────────────

    void BuildCentreCol(GameObject root)
    {
        // Left column: x=0..CentreW, y=0..-ProxH
        float cx   = CentreW * 0.5f;
        float cyT  = -ProxH * 0.20f;
        float cyM  = -ProxH * 0.50f;
        float cyB  = -ProxH * 0.80f;

        // ── PROX ROOT (shown when Nash active) ────────────────────────────────
        _proxRoot = NewGO("ProxRoot", root.transform);
        var prt   = _proxRoot.AddComponent<RectTransform>();
        prt.anchorMin = prt.anchorMax = new Vector2(0,1);
        prt.pivot = new Vector2(0,1);
        prt.anchoredPosition = new Vector2(0f, 0f);   // starts at left edge
        prt.sizeDelta = new Vector2(CentreW, ProxH);
        _proxRoot.SetActive(false);

        // Danger lines: leader→YOU and YOU→follower, coloured by TTC
        float lineX1 = cx - 2f; float lineX2 = cx + 2f; // slight gap for YOU icon
        _dangerLineL = ChildImg(_proxRoot, "DangerL");
        _dangerLineL.color = new Color(0.2f, 0.8f, 0.3f, 0.6f);
        CA(_dangerLineL.rectTransform,
           new Vector2(cx, (cyT + cyM) * 0.5f),
           new Vector2(3f, Mathf.Abs(cyM - cyT) - 22f));

        _dangerLineF = ChildImg(_proxRoot, "DangerF");
        _dangerLineF.color = new Color(0.2f, 0.8f, 0.3f, 0.6f);
        CA(_dangerLineF.rectTransform,
           new Vector2(cx, (cyM + cyB) * 0.5f),
           new Vector2(3f, Mathf.Abs(cyB - cyM) - 22f));

        // Leader car shape — top, gap label above it
        // ProxH=212: cyT=-42, cyM=-106, cyB=-170
        _leaderShape = VehicleShape(_proxRoot, "Leader", false);
        CA(_leaderShape.rectTransform, new Vector2(cx, cyT), new Vector2(100f, 36f));

        _leaderGapLbl = ChildTxt(_proxRoot, "LGap", "—", 14, true);
        _leaderGapLbl.alignment = TextAlignmentOptions.Center;
        CA(_leaderGapLbl.rectTransform, new Vector2(cx, cyT + 28f), new Vector2(90f, 20f));

        _selfShape = VehicleShape(_proxRoot, "Self", true);
        CA(_selfShape.rectTransform, new Vector2(cx, cyM), new Vector2(112f, 42f));

        var youLbl = ChildTxt(_proxRoot, "YouLbl", "YOU", 11, true);
        youLbl.alignment = TextAlignmentOptions.Center;
        youLbl.color = new Color(0.75f, 0.75f, 0.75f);
        CA(youLbl.rectTransform, new Vector2(cx, cyM - 28f), new Vector2(50f, 16f));

        _followerShape = VehicleShape(_proxRoot, "Follower", false);
        CA(_followerShape.rectTransform, new Vector2(cx, cyB), new Vector2(100f, 36f));

        _followerGapLbl = ChildTxt(_proxRoot, "FGap", "—", 14, true);
        _followerGapLbl.alignment = TextAlignmentOptions.Center;
        CA(_followerGapLbl.rectTransform, new Vector2(cx, cyB - 26f), new Vector2(90f, 20f));

        // Speed lines flanking the YOU icon
        if (speedLineTexture != null)
        {
            speedLineTexture.wrapMode = TextureWrapMode.Repeat;
            _speedLineMat = new Material(Shader.Find("UI/Default"));
            _speedLineMat.mainTexture = speedLineTexture;

            _speedLineL = SpeedLine(_proxRoot, "SLL",
                new Vector2(cx - 70f, cyM), new Vector2(5f, ProxH - 24f));
            _speedLineR = SpeedLine(_proxRoot, "SLR",
                new Vector2(cx + 70f, cyM), new Vector2(5f, ProxH - 24f));
        }

        // ── MANUAL ROOT (shown when Nash inactive) ────────────────────────────
        _manualRoot = NewGO("ManualRoot", root.transform);
        var mrt     = _manualRoot.AddComponent<RectTransform>();
        mrt.anchorMin = mrt.anchorMax = new Vector2(0,1);
        mrt.pivot = new Vector2(0,1);
        mrt.anchoredPosition = new Vector2(0f, 0f);
        mrt.sizeDelta = new Vector2(CentreW, ProxH);

        var manLbl = ChildTxt(_manualRoot, "ManLbl",
            "MANUAL\nPlatoon view inactive", 26, true);
        manLbl.alignment = TextAlignmentOptions.Center;
        manLbl.color = new Color(0.38f, 0.38f, 0.38f);
        CA(manLbl.rectTransform, new Vector2(CentreW * 0.5f, -ProxH * 0.5f),
           new Vector2(CentreW - 20f, ProxH - 10f));

        var subLbl = ChildTxt(_manualRoot, "SubLbl", "press J to join", 16, false);
        subLbl.alignment = TextAlignmentOptions.Center;
        subLbl.color = new Color(0.28f, 0.28f, 0.28f);
        CA(subLbl.rectTransform,
           new Vector2(CentreW * 0.5f, -ProxH * 0.5f - 28f),
           new Vector2(CentreW - 20f, 20f));
    }

    // Draws a flat car silhouette using only UI primitives (no sprite needed).
    // isEgo = true → slightly larger, white; false → dimmer colour
    Image VehicleShape(GameObject parent, string name, bool isEgo)
    {
        // Body rectangle
        var body = ChildImg(parent, name + "Body");
        body.color = isEgo ? new Color(0.9f, 0.9f, 0.9f) : new Color(0.45f, 0.55f, 0.70f);

        // Windscreen strip (child of body)
        var ws = ChildImg(body.gameObject, name + "WS");
        ws.color = isEgo ? new Color(0.4f, 0.8f, 1f, 0.6f) : new Color(0.2f, 0.4f, 0.7f, 0.5f);
        var wsrt = ws.rectTransform;
        wsrt.anchorMin = new Vector2(0.2f, 0.55f); wsrt.anchorMax = new Vector2(0.8f, 0.92f);
        wsrt.offsetMin = wsrt.offsetMax = Vector2.zero;

        // Left wheel
        BuildWheel(body.gameObject, name + "WL", new Vector2(0.12f, 0f), new Vector2(0.28f, 0.22f));
        // Right wheel
        BuildWheel(body.gameObject, name + "WR", new Vector2(0.72f, 0f), new Vector2(0.88f, 0.22f));

        return body;
    }

    void BuildWheel(GameObject parent, string name, Vector2 amin, Vector2 amax)
    {
        var w = ChildImg(parent, name);
        w.color = new Color(0.15f, 0.15f, 0.15f);
        var rt = w.rectTransform;
        rt.anchorMin = amin; rt.anchorMax = amax;
        rt.offsetMin = rt.offsetMax = Vector2.zero;
    }

    void Dash(GameObject parent, float x1, float x2, float cy)
    {
        if (x2 <= x1) return;
        var d = ChildImg(parent, "Dash");
        d.color = new Color(1f, 1f, 1f, 0.12f);
        CA(d.rectTransform, new Vector2((x1 + x2) * 0.5f, cy), new Vector2(x2 - x1, 1f));
    }

    RawImage SpeedLine(GameObject parent, string name, Vector2 pos, Vector2 sz)
    {
        var go = NewGO(name, parent.transform);
        var ri = go.AddComponent<RawImage>();
        ri.texture  = speedLineTexture;
        ri.material = _speedLineMat;
        ri.color    = new Color(1f, 1f, 1f, 0f);
        CA(ri.rectTransform, pos, sz);
        return ri;
    }

    // ── Status bar ────────────────────────────────────────────────────────────

    void BuildStatusBar(GameObject root)
    {
        float yTop  = -ProxH;
        float barH  = StatusH - 18f;
        float barW  = 180f;
        // Layout of right portion: [SYS 32px][bar 180px][YOU 32px][state 90px]
        // Total right block = 334px, place it starting at PW-334
        float rightBlockW = 32f + barW + 32f + 90f;
        float barX  = PW - rightBlockW + 32f;  // bar starts after SYS label
        float segY  = yTop - (StatusH - barH) * 0.5f;

        _statusBg = ChildImg(root, "StatusBg");
        TL(_statusBg.rectTransform, new Vector2(0f, yTop), new Vector2(PW, StatusH));
        _statusBg.color = new Color(0.04f, 0.05f, 0.07f, 0.96f);

        // Phase label — left, stops before right block
        _statusLbl = ChildTxt(root, "StatusTxt",
            "  MANUAL MODE  —  press J to join platoon", 16, true);
        TL(_statusLbl.rectTransform,
           new Vector2(6f, yTop - 4f),
           new Vector2(PW - rightBlockW - 10f, StatusH - 8f));
        _statusLbl.alignment = TextAlignmentOptions.MidlineLeft;
        _statusLbl.color = new Color(0.5f, 0.5f, 0.5f);

        // ── Authority bar: SYS [░░░░████] YOU  State ─────────────────────────
        var sysLbl = ChildTxt(root, "SysLbl", "SYS", 11, false);
        sysLbl.alignment = TextAlignmentOptions.MidlineRight;
        sysLbl.color = new Color(0.25f, 0.55f, 1f);
        TL(sysLbl.rectTransform,
           new Vector2(PW - rightBlockW, yTop - 4f), new Vector2(30f, StatusH - 8f));

        var barBg = ChildImg(root, "BarBg");
        barBg.color = new Color(0.08f, 0.09f, 0.12f);
        TL(barBg.rectTransform, new Vector2(barX, segY), new Vector2(barW, barH));

        float segW = (barW - (BarSegs + 1) * 2f) / BarSegs;
        _barSegs = new Image[BarSegs];
        for (int i = 0; i < BarSegs; i++)
        {
            var seg = ChildImg(root, $"Bar{i}");
            float sx = barX + 2f + i * (segW + 2f);
            TL(seg.rectTransform, new Vector2(sx, segY - 2f), new Vector2(segW, barH - 4f));
            seg.color = new Color(0.12f, 0.14f, 0.18f);
            _barSegs[i] = seg;
        }

        _barThumb = ChildImg(root, "BarThumb");
        var trt = _barThumb.rectTransform;
        trt.SetParent(barBg.rectTransform, false);
        trt.anchorMin = new Vector2(0f, 0f);
        trt.anchorMax = new Vector2(0f, 1f);
        trt.pivot = new Vector2(0.5f, 0.5f);
        trt.sizeDelta = new Vector2(3f, 0f);
        trt.anchoredPosition = Vector2.zero;
        _barThumb.color = new Color(0.3f, 0.9f, 0.4f);

        var youLbl2 = ChildTxt(root, "YouLbl2", "YOU", 11, false);
        youLbl2.alignment = TextAlignmentOptions.MidlineLeft;
        youLbl2.color = new Color(0.3f, 0.9f, 0.4f);
        TL(youLbl2.rectTransform,
           new Vector2(barX + barW + 2f, yTop - 4f), new Vector2(30f, StatusH - 8f));

        _authLbl = ChildTxt(root, "AuthLbl", "Manual", 13, true);
        TL(_authLbl.rectTransform,
           new Vector2(barX + barW + 34f, yTop - 4f), new Vector2(88f, StatusH - 8f));
        _authLbl.alignment = TextAlignmentOptions.MidlineLeft;
        _authLbl.color = new Color(0.4f, 0.4f, 0.4f);
    }

    void OnDestroy() { if (_speedLineMat != null) Destroy(_speedLineMat); }

    // =========================================================================
    // UI helpers
    // =========================================================================

    static GameObject NewGO(string name, Transform parent)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        return go;
    }

    Image ChildImg(GameObject p, string name)
        => NewGO(name, p.transform).AddComponent<Image>();

    static TMP_FontAsset _cachedFont;

    TMP_Text ChildTxt(GameObject p, string name, string text, int size, bool bold)
    {
        var go = NewGO(name, p.transform);
        var t = go.AddComponent<TextMeshProUGUI>();
        t.text = text; t.fontSize = size;
        t.fontStyle = bold ? FontStyles.Bold : FontStyles.Normal;
        t.color = Color.white;
        t.overflowMode = TextOverflowModes.Overflow;
        t.raycastTarget = false;
        // Load LiberationSans SDF directly — prevents grey atlas-bleed background
        if (_cachedFont == null)
            _cachedFont = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");
        if (_cachedFont != null) t.font = _cachedFont;
        return t;
    }

    static void TL(RectTransform rt, Vector2 pos, Vector2 sz)
    {
        rt.anchorMin = rt.anchorMax = new Vector2(0,1);
        rt.pivot = new Vector2(0,1);
        rt.anchoredPosition = pos; rt.sizeDelta = sz;
    }

    // Centre-anchor using top-left anchor (position from panel top-left)
    static void CA(RectTransform rt, Vector2 pos, Vector2 sz)
    {
        rt.anchorMin = rt.anchorMax = new Vector2(0,1);
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = pos; rt.sizeDelta = sz;
    }

    static void SetCA(RectTransform rt, Vector2 pos)
        => rt.anchoredPosition = pos;
}
