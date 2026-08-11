using UnityEngine;

public class CameraController1 : MonoBehaviour
{
    public Transform            player;
    public Transform            driver;
    public AdvancedBicycleModel vehicle;
    public PlatoonManager       platoonManager;

    public float speed;

    [Header("Trigger — Left signal")]
    public KeyCode leftTriggerKey         = KeyCode.J;
    public int     leftTriggerWheelButton = 0;   // G29 button index (-1 = disabled)

    [Header("Trigger — Right signal")]
    public KeyCode rightTriggerKey         = KeyCode.K;
    public int     rightTriggerWheelButton = -1;  // disabled by default

    [Header("Gaze Settings")]
    [Range(0f, 60f)] public float sustainedYawDegRight = 20f; // mirror on the right — farther
    [Range(0f, 60f)] public float sustainedYawDegLeft  = 10f; // mirror on the left — closer
    [Range(1f, 60f)] public float sustainedFadeSpeed   = 20f;

    private Quaternion _localRestRotation;

    // One-shot gaze sequence
    private bool  _gazeActive = false;
    private float _gazeTime   = 0f;
    private float _gazePitch  = 0f;
    private float _gazeYaw    = 0f;
    private float _gazeSign   = 1f;  // +1 = right, -1 = left

    // Sustained merge-gaze
    private float _sustainedYaw        = 0f;
    private float _sustainedYawTarget  = 0f;
    private float _initialLateralDist  = 0f;

    private bool  _joinPending = false;

    private struct GazeKeyframe
    {
        public float startTime, endTime;
        public float fromYaw, fromPitch, toYaw, toPitch;
        public int   easeType; // 0=EaseOut, 1=EaseInOut, 2=Hold
    }

    private GazeKeyframe[] _gazeKeyframes;

    void Start()
    {
        speed = vehicle.GetVx();

        if (driver == null)
            driver = player.Find("driver").transform;

        _localRestRotation = transform.localRotation;
        BuildGazeSequence();

        string parentName = transform.parent != null ? transform.parent.name : "NO PARENT";
        Debug.Log($"[CameraController1] parent='{parentName}' localRot={transform.localRotation.eulerAngles}");
    }

    void BuildGazeSequence()
    {
        // All yaw values unsigned — _gazeSign is applied at render time.
        _gazeKeyframes = new GazeKeyframe[]
        {
            new GazeKeyframe { startTime=0.00f, endTime=0.25f, fromYaw=0f,  fromPitch=0f,  toYaw=20f, toPitch=0f,  easeType=0 },
            new GazeKeyframe { startTime=0.25f, endTime=0.40f, fromYaw=20f, fromPitch=0f,  toYaw=20f, toPitch=0f,  easeType=2 },
            new GazeKeyframe { startTime=0.40f, endTime=0.75f, fromYaw=20f, fromPitch=0f,  toYaw=25f, toPitch=-4f, easeType=1 },
            new GazeKeyframe { startTime=0.75f, endTime=1.20f, fromYaw=25f, fromPitch=-4f, toYaw=25f, toPitch=-4f, easeType=2 },
            new GazeKeyframe { startTime=1.20f, endTime=2.40f, fromYaw=25f, fromPitch=-4f, toYaw=0f,  toPitch=0f,  easeType=1 },
        };
    }

    void Update()
    {
        if (_gazeActive) return;

        bool leftPressed  = Input.GetKeyDown(leftTriggerKey)
                         || (leftTriggerWheelButton  >= 0 && Input.GetKeyDown("joystick button " + leftTriggerWheelButton));
        bool rightPressed = Input.GetKeyDown(rightTriggerKey)
                         || (rightTriggerWheelButton >= 0 && Input.GetKeyDown("joystick button " + rightTriggerWheelButton));

        if (leftPressed)  { _gazeSign = -1f; _joinPending = true; }
        if (rightPressed) { _gazeSign =  1f; _joinPending = true; }
    }

    void FixedUpdate()
    {
        if (_joinPending)
        {
            _joinPending        = false;
            _gazeActive         = true;
            _gazeTime           = 0f;
            _initialLateralDist = Mathf.Max(Mathf.Abs(GetLateralDiff()), 0.5f);
            Debug.Log($"[CameraController1] Gaze STARTED dir={(_gazeSign > 0 ? "RIGHT" : "LEFT")} initDist={_initialLateralDist:F2}m");
        }

        // ── One-shot gaze sequence ────────────────────────────────────────────
        if (_gazeActive)
        {
            _gazeTime += Time.fixedDeltaTime;
            float totalDuration = _gazeKeyframes[_gazeKeyframes.Length - 1].endTime;

            if (_gazeTime >= totalDuration)
            {
                _gazeActive = false;
                _gazePitch  = 0f;
                _gazeYaw    = 0f;
            }
            else
            {
                EvaluateGaze(_gazeTime);
            }
        }

        // ── Sustained merge-gaze ──────────────────────────────────────────────
        _sustainedYawTarget = ComputeSustainedTarget();
        _sustainedYaw = Mathf.MoveTowards(_sustainedYaw, _sustainedYawTarget,
                                           sustainedFadeSpeed * Time.fixedDeltaTime);

        // ── Compose final rotation ────────────────────────────────────────────
        float totalYaw = _gazeYaw * _gazeSign + _sustainedYaw;
        transform.localRotation = _localRestRotation * Quaternion.Euler(_gazePitch, totalYaw, 0f);
    }

    float GetLateralDiff()
    {
        if (platoonManager == null) return 0f;
        AdvancedBicycleModel[] platoon = platoonManager.GetPlatoonVehicles();
        if (platoon == null || platoon.Length == 0) return 0f;
        return platoon[0].GetPosition().x - vehicle.GetPosition().x;
    }

    float ComputeSustainedTarget()
    {
        if (!_gazeActive && _sustainedYaw == 0f && _sustainedYawTarget == 0f) return 0f;
        if (_initialLateralDist < 0.01f) return 0f;

        float diff = GetLateralDiff();
        if (Mathf.Abs(diff) < 0.5f) return 0f;

        float ratio      = Mathf.Clamp01(Mathf.Abs(diff) / _initialLateralDist);
        float maxYaw     = _gazeSign > 0f ? sustainedYawDegRight : sustainedYawDegLeft;
        return _gazeSign * maxYaw * ratio;
    }

    void EvaluateGaze(float t)
    {
        for (int i = 0; i < _gazeKeyframes.Length; i++)
        {
            GazeKeyframe kf = _gazeKeyframes[i];
            if (t >= kf.startTime && t < kf.endTime)
            {
                float duration = kf.endTime - kf.startTime;
                float localT   = (t - kf.startTime) / duration;
                float easedT   = ApplyEase(localT, kf.easeType);

                _gazeYaw   = Mathf.LerpUnclamped(kf.fromYaw,   kf.toYaw,   easedT);
                _gazePitch = Mathf.LerpUnclamped(kf.fromPitch, kf.toPitch, easedT);
                return;
            }
        }
    }

    float ApplyEase(float t, int easeType)
    {
        switch (easeType)
        {
            case 0: return 1f - (1f - t) * (1f - t);
            case 1: return t < 0.5f ? 2f * t * t : 1f - 2f * (1f - t) * (1f - t);
            default: return 0f;
        }
    }
}
