// SimulationRunner — MonoBehaviour (Script Execution Order -100).
//
// Mirrors unified/simulation/simulator.py:
//   Sets Time.fixedDeltaTime from settings (100 Hz physics step).
//   Assigns SimCfg.I so all NashPlatoon scripts share the same config asset.
//
// Script Execution Order (set in Project Settings → Script Execution Order):
//   SimulationRunner     -100   Awake: SimCfg.I = settings, FixedDt = 0.01
//   NashCoordinator       -50   Nash solve → ego.NashThrottle/Brake/SteerNorm
//   PlatoonManager        -40   Rajamani for platoon; skips ego if NashModeActive
//   AdvancedBicycleModel  -30   reads NashThrottle/Brake/SteerNorm → dynamics
//   PlatoonDataVisualizer   0   logging after everything settles
using UnityEngine;

public class SimulationRunner : MonoBehaviour
{
    [SerializeField] NashPlatoonSettings settings;

    void Awake()
    {
        if (settings == null)
        {
            Debug.LogError("[SimulationRunner] NashPlatoonSettings not assigned!");
            return;
        }
        SimCfg.I                    = settings;
        Time.fixedDeltaTime         = settings.Timing.NashDtLong / 10f;  // 0.01 s (100 Hz)
        Application.targetFrameRate = -1;  // uncapped — let physics drive loop
        Debug.Log($"[SimulationRunner] FixedDt={Time.fixedDeltaTime:F4}s  "
                + $"LongNashDt={settings.Timing.NashDtLong}s  "
                + $"LatNashDt={settings.Timing.NashDtLat}s");
    }
}
