using UnityEngine;
using System.Collections;

public class EngineAudio : MonoBehaviour
{
    [Header("Audio Sources")]
    public AudioSource startingSound;
    public AudioSource runningSound;
    public AudioSource idleSound;
    public AudioSource reverseSound;

    [Header("Audio Clips")]
    public AudioClip engineStartClip;
    public AudioClip engineRunningClip;
    public AudioClip engineIdleClip;
    public AudioClip engineReverseClip;

    [Header("Volume Settings")]
    public float runningMaxVolume = 1.0f;
    public float reverseMaxVolume = 0.7f;
    public float idleMaxVolume = 0.5f;

    [Header("Pitch Settings")]
    public float runningMaxPitch = 2.0f;
    public float reverseMaxPitch = 1.5f;

    [Header("Rev Limiter Settings")]
    public float LimiterSound = 1f;
    public float LimiterFrequency = 3f;
    public float LimiterEngage = 0.8f;

    [Header("Audio Enable")]
    [SerializeField] private bool enableAudio = true;

    private float revLimiter;
    private AdvancedBicycleModel vehicle;
    private VehicleParameters vehicleParams;
    private Powertrain powertrain;
    private bool audioReady = false;

    private float currentRunningVolume = 0f;
    private float currentIdleVolume = 0f;
    private float volumeTransitionSpeed = 2f;

    void Awake()
    {
        if (!enableAudio) return;

        if (startingSound == null) startingSound = gameObject.AddComponent<AudioSource>();
        if (runningSound  == null) runningSound  = gameObject.AddComponent<AudioSource>();
        if (idleSound     == null) idleSound     = gameObject.AddComponent<AudioSource>();
        if (reverseSound  == null) reverseSound  = gameObject.AddComponent<AudioSource>();

        _spatialBlend = CompareTag("Player") ? 0f : 1f;
    }
    private float _spatialBlend = 0f;

    void Start()
    {
        if (!enableAudio) return;
        StartCoroutine(InitializeAudio());
    }

    private IEnumerator InitializeAudio()
    {
        yield return null;

        vehicle = GetComponent<AdvancedBicycleModel>();
        if (vehicle == null) { Debug.LogError("[EngineAudio] " + gameObject.name + ": AdvancedBicycleModel not found!"); yield break; }

        powertrain = vehicle.powertrain;
        if (powertrain == null) { Debug.LogError("[EngineAudio] " + gameObject.name + ": Powertrain not found!"); yield break; }

        vehicleParams = new VehicleParameters();

        // Borrow clips from another EngineAudio if this vehicle has none in Inspector
        if (engineStartClip == null || engineRunningClip == null || engineIdleClip == null)
        {
            EngineAudio[] all = FindObjectsByType<EngineAudio>(FindObjectsSortMode.None);
            foreach (EngineAudio other in all)
            {
                if (other == this) continue;
                if (other.engineStartClip != null)
                {
                    engineStartClip   = other.engineStartClip;
                    engineRunningClip = other.engineRunningClip;
                    engineIdleClip    = other.engineIdleClip;
                    engineReverseClip = other.engineReverseClip;
                    Debug.Log("[EngineAudio] " + gameObject.name + ": borrowed clips from " + other.gameObject.name);
                    break;
                }
            }
        }

        ConfigureSource(startingSound, engineStartClip,   false);
        ConfigureSource(runningSound,  engineRunningClip, true);
        ConfigureSource(idleSound,     engineIdleClip,    true);
        ConfigureSource(reverseSound,  engineReverseClip, true);

        startingSound.volume = 0f;
        runningSound.volume  = 0f;
        idleSound.volume     = 0f;
        reverseSound.volume  = 0f;

        if (idleSound.clip    != null) idleSound.Play();
        if (runningSound.clip != null) runningSound.Play();

        audioReady = true;
        Debug.Log("[EngineAudio] " + gameObject.name + ": audio ready. idleClip=" + (idleSound.clip != null ? idleSound.clip.name : "NULL"));
    }

    private void ConfigureSource(AudioSource src, AudioClip clip, bool loop)
    {
        if (src == null) return;
        src.clip         = clip;
        src.loop         = loop;
        src.playOnAwake  = false;
        src.spatialBlend = _spatialBlend;
        src.maxDistance  = 60f;
        src.minDistance  = 2f;
        src.rolloffMode  = AudioRolloffMode.Linear;
        src.volume       = 0f;
    }

    void FixedUpdate()
    {
        if (!audioReady || powertrain == null) return;

        UpdateEngineSounds();
    }

    private void UpdateEngineSounds()
    {
        if (powertrain.EngineState == 2)
        {
            float rpmRatio     = Mathf.Clamp01(powertrain.EngineRpm / 6700f);
            float currentSpeed = Mathf.Abs(vehicle.GetVx());
            float throttle     = vehicle.GetThrottleInput();
            float speedFactor  = Mathf.Clamp01(currentSpeed / 2f);

            if (rpmRatio > LimiterEngage)
                revLimiter = (Mathf.Sin(Time.time * LimiterFrequency) + 1f) * LimiterSound * (rpmRatio - LimiterEngage);
            else
                revLimiter = 0f;

            if (idleSound)
            {
                if (!idleSound.isPlaying) idleSound.Play();
                float target = (powertrain.EngineRpm < 1500f || currentSpeed < 2f)
                    ? idleMaxVolume * (1f - rpmRatio * 0.7f) : 0f;
                currentIdleVolume = Mathf.Lerp(currentIdleVolume, target, Time.deltaTime * volumeTransitionSpeed);
                idleSound.volume = currentIdleVolume;
                idleSound.pitch  = Mathf.Lerp(0.8f, 1.2f, rpmRatio * 0.3f);
            }

            if (runningSound)
            {
                if (!runningSound.isPlaying) runningSound.Play();
                float baseVol    = Mathf.Lerp(0.1f, runningMaxVolume, rpmRatio);
                float throttleEf = Mathf.Lerp(0.3f, 1.0f, throttle);
                float speedEf    = Mathf.Lerp(0.7f, 1.0f, speedFactor);
                float target     = baseVol * throttleEf * speedEf;
                currentRunningVolume = Mathf.Lerp(currentRunningVolume, target, Time.deltaTime * volumeTransitionSpeed);
                runningSound.volume = currentRunningVolume;
                runningSound.pitch  = Mathf.Lerp(0.8f, runningMaxPitch, rpmRatio) + revLimiter * 0.1f;
            }

            if (reverseSound)
            {
                if (vehicle.GetVx() < -0.1f)
                {
                    if (!reverseSound.isPlaying) reverseSound.Play();
                    reverseSound.volume = Mathf.Lerp(0f, reverseMaxVolume, rpmRatio);
                    reverseSound.pitch  = Mathf.Lerp(0.5f, reverseMaxPitch, rpmRatio);
                }
                else
                {
                    reverseSound.volume = Mathf.Lerp(reverseSound.volume, 0f, Time.deltaTime * volumeTransitionSpeed);
                    if (reverseSound.volume < 0.01f) reverseSound.Stop();
                }
            }
        }
        else
        {
            FadeOut(ref currentIdleVolume,    idleSound);
            FadeOut(ref currentRunningVolume, runningSound);
            if (reverseSound)
            {
                reverseSound.volume = Mathf.Lerp(reverseSound.volume, 0f, Time.deltaTime * volumeTransitionSpeed);
                if (reverseSound.volume < 0.01f) reverseSound.Stop();
            }
        }
    }

    private void FadeOut(ref float vol, AudioSource src)
    {
        if (src == null) return;
        vol        = Mathf.Lerp(vol, 0f, Time.deltaTime * volumeTransitionSpeed);
        src.volume = vol;
        if (vol < 0.01f) src.Stop();
    }

    public IEnumerator StartEngine()
    {
        Debug.Log("[EngineAudio] " + gameObject.name + ": Starting engine sequence");
        yield return new WaitUntil(() => audioReady);
        if (startingSound != null && engineStartClip != null)
            startingSound.PlayOneShot(engineStartClip);
    }

    public void StopEngine() { StopAllSounds(); }

    private void StopAllSounds()
    {
        if (idleSound    && idleSound.isPlaying)    idleSound.Stop();
        if (runningSound && runningSound.isPlaying) runningSound.Stop();
        if (reverseSound && reverseSound.isPlaying) reverseSound.Stop();
        currentIdleVolume    = 0f;
        currentRunningVolume = 0f;
    }

    void OnDisable() { StopAllSounds(); }
}
