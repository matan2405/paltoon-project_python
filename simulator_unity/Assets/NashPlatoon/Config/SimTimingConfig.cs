using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/Timing")]
public class SimTimingConfig : ScriptableObject {
    public float SimDt         => Time.fixedDeltaTime;
    public float NashDtLong    = 0.1f;
    public float NashDtLat     = 0.05f;
    public int   NashNp        = 20;
    public int   NashNu        = 10;
    public float PhaseLockTime = 5.0f;
    public float MaxSteerRate  = 0.087f;
}
