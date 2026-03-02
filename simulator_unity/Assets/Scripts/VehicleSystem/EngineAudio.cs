using UnityEngine;
using System.Collections;

/// <summary>
/// Handles all engine-related audio effects including engine start, running, and idle sounds
/// </summary>
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

    private float speedRatio;
    private float revLimiter;
    private AdvancedBicycleModel vehicle;
    private VehicleParameters vehicleParams;
    private Powertrain powertrain;
    
    // ✅ IMPROVED: Better volume transition control
    private float currentRunningVolume = 0f;
    private float targetRunningVolume = 0f;
    private float currentIdleVolume = 0f;
    private float volumeTransitionSpeed = 2f; // How fast volume changes (NEW)

    void Start()
    {
        // We need to wait a frame to ensure AdvancedBicycleModel has initialized
        StartCoroutine(InitializeAudio());
    }

    private IEnumerator InitializeAudio()
    {
        // Wait for the next frame to ensure AdvancedBicycleModel is initialized
        yield return null;

        vehicle = GetComponent<AdvancedBicycleModel>();
        if (vehicle == null)
        {
            Debug.LogError("AdvancedBicycleModel not found!");
            yield break;
        }

        powertrain = vehicle.powertrain;
        if (powertrain == null)
        {
            Debug.LogError("Powertrain not found!");
            yield break;
        }

        vehicleParams = new VehicleParameters();

        // Configure audio sources
        SetupAudioSource(startingSound, engineStartClip, false);
        SetupAudioSource(runningSound, engineRunningClip, true);
        SetupAudioSource(idleSound, engineIdleClip, true);
        SetupAudioSource(reverseSound, engineReverseClip, true);

        SetAllVolumesToZero();

        Debug.Log("Audio initialization complete");
    }

    private void SetupAudioSource(AudioSource source, AudioClip clip, bool shouldLoop)
    {
        if (source && clip)
        {
            Debug.Log($"Setting up {source.name} with clip {clip.name}");
            source.clip = clip;
            source.loop = shouldLoop;
            source.playOnAwake = false;
            source.spatialBlend = 0f; // Force 2D sound
            Debug.Log($"Audio source {source.name} setup complete. Loop: {source.loop}, Volume: {source.volume}");
        }
        else
        {
            Debug.LogError($"Missing audio setup: Source: {(source != null ? source.name : "null")}, Clip: {(clip != null ? clip.name : "null")}");
        }
    }

    private void SetAllVolumesToZero()
    {
        if (idleSound) idleSound.volume = 0;
        if (runningSound) runningSound.volume = 0;
        if (reverseSound) reverseSound.volume = 0;
    }

    void FixedUpdate()
    {
        if (vehicle == null || powertrain == null) return;  // Safety check for required components

        float currentSpeed = vehicle.GetVx();
        float maxSpeed = vehicleParams.maxVelocity / 3.6f; // Convert km/h to m/s
        speedRatio = Mathf.Abs(currentSpeed / maxSpeed);

        // ✅ UPDATED: Calculate rev limiter effect based on actual engine RPM
        float rpmRatio = Mathf.Clamp01(powertrain.EngineRpm / 6700f);
        if (rpmRatio > LimiterEngage)
        {
            revLimiter = (Mathf.Sin(Time.time * LimiterFrequency) + 1f) * LimiterSound * (rpmRatio - LimiterEngage);
        }

        UpdateEngineSounds();
    }

    // ✅ IMPROVED: Better engine sound logic with smoother transitions
    private void UpdateEngineSounds()
    {
        if (powertrain.EngineState == 2) // Engine running
        {
            // ✅ UPDATED: Use actual engine RPM instead of speed-based calculation
            float rpmRatio = Mathf.Clamp01(powertrain.EngineRpm / 6700f);
            float currentSpeed = Mathf.Abs(vehicle.GetVx());
            float acceleration = vehicle.GetAx();
            float throttleInput = VehicleInputs.Instance.ThrottleInput;

            // ✅ IMPROVED: Better speed factor calculation
            float speedFactor = Mathf.Clamp01(currentSpeed / 2f); // Normalize for better audio response

            // ✅ IMPROVED: Idle sound with smoother transitions
            if (idleSound)
            {
                if (!idleSound.isPlaying) idleSound.Play();

                // Idle sound plays when RPM is low or speed is very low
                float targetIdleVolume = (powertrain.EngineRpm < 1500f || currentSpeed < 2f) ? 
                    idleMaxVolume * (1f - rpmRatio * 0.7f) : 0f;

                // ✅ NEW: Smooth volume transitions
                currentIdleVolume = Mathf.Lerp(currentIdleVolume, targetIdleVolume, 
                    Time.deltaTime * volumeTransitionSpeed);
                
                idleSound.volume = currentIdleVolume;
                
                // ✅ IMPROVED: Idle pitch varies slightly with RPM
                idleSound.pitch = Mathf.Lerp(0.8f, 1.2f, rpmRatio * 0.3f);
            }

            // ✅ IMPROVED: Running sound with better response to engine state
            if (runningSound)
            {
                if (!runningSound.isPlaying && (throttleInput > 0 || powertrain.EngineRpm > 1200f)) 
                    runningSound.Play();

                // ✅ IMPROVED: Volume based on RPM and throttle, not just acceleration
                float baseVolume = Mathf.Lerp(0.1f, runningMaxVolume, rpmRatio);
                
                // ✅ NEW: Throttle response - more volume when accelerating
                float throttleEffect = Mathf.Lerp(0.3f, 1.0f, throttleInput);
                
                // ✅ NEW: Speed effect for highway sound
                float speedEffect = Mathf.Lerp(0.7f, 1.0f, speedFactor);
                
                targetRunningVolume = baseVolume * throttleEffect * speedEffect;

                // ✅ NEW: Smooth volume transitions
                currentRunningVolume = Mathf.Lerp(currentRunningVolume, targetRunningVolume, 
                    Time.deltaTime * volumeTransitionSpeed);
                
                runningSound.volume = currentRunningVolume;
                
                // ✅ IMPROVED: Pitch based on actual RPM with rev limiter effect
                float basePitch = Mathf.Lerp(0.8f, runningMaxPitch, rpmRatio);
                runningSound.pitch = basePitch + revLimiter * 0.1f; // Add rev limiter crackling
            }

            // ✅ IMPROVED: Better reverse sound handling (if needed in future)
            if (reverseSound)
            {
                if (currentSpeed < -0.1f) // Moving backwards
                {
                    if (!reverseSound.isPlaying) reverseSound.Play();
                    reverseSound.volume = Mathf.Lerp(0f, reverseMaxVolume, rpmRatio);
                    reverseSound.pitch = Mathf.Lerp(0.5f, reverseMaxPitch, rpmRatio);
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
            // ✅ IMPROVED: Smooth fade out when engine stops
            if (idleSound)
            {
                currentIdleVolume = Mathf.Lerp(currentIdleVolume, 0f, Time.deltaTime * volumeTransitionSpeed);
                idleSound.volume = currentIdleVolume;
                if (currentIdleVolume < 0.01f) idleSound.Stop();
            }

            if (runningSound)
            {
                currentRunningVolume = Mathf.Lerp(currentRunningVolume, 0f, Time.deltaTime * volumeTransitionSpeed);
                runningSound.volume = currentRunningVolume;
                if (currentRunningVolume < 0.01f) runningSound.Stop();
            }

            if (reverseSound)
            {
                reverseSound.volume = Mathf.Lerp(reverseSound.volume, 0f, Time.deltaTime * volumeTransitionSpeed);
                if (reverseSound.volume < 0.01f) reverseSound.Stop();
            }
        }
    }

    // ✅ IMPROVED: Better engine start sequence
    public IEnumerator StartEngine()
    {
        Debug.Log("Starting engine sequence");

        // Play the initial starting sound
        if (startingSound && engineStartClip)
        {
            startingSound.PlayOneShot(engineStartClip);
        }

        yield return new WaitForSeconds(0.6f);

        // ✅ IMPROVED: Start idle sound at appropriate volume
        if (idleSound)
        {
            idleSound.volume = idleMaxVolume * 0.6f; // Start at 60% of max idle volume
            idleSound.Play();
        }
    }

    public void StopEngine()
    {
        StopAllSounds();
    }

    private void StopAllSounds()
    {
        if (idleSound && idleSound.isPlaying) idleSound.Stop();
        if (runningSound && runningSound.isPlaying) runningSound.Stop();
        if (reverseSound && reverseSound.isPlaying) reverseSound.Stop();
        
        // ✅ NEW: Reset volume variables
        currentIdleVolume = 0f;
        currentRunningVolume = 0f;
    }

    void OnDisable()
    {
        StopAllSounds();
    }
}