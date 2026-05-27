using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/Settings")]
public class NashPlatoonSettings : ScriptableObject {
    public SimTimingConfig   Timing;
    public NashWeightsConfig Nash;
    public SafetyFieldConfig SafetyField;
    public AuthorityConfig   Authority;
}

// Global accessor — assigned in SimulationRunner.Awake()
public static class SimCfg {
    public static NashPlatoonSettings I;
}
