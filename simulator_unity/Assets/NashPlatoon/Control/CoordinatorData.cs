// CoordinatorData — per-step logging payload written by NashCoordinator,
// read by NashHUD and PlatoonDataVisualizer.

public enum MergePhase { Approach, GapSearch, Merge, Following }

public class CoordinatorData
{
    // Longitudinal
    public float ULong;       // u_shared = u1 + u2 [m/s²]
    public float U1Long;      // system player 1 control
    public float U2Long;      // human  player 2 control
    public float LongLambda;  // authority ratio λ
    public float LongForce;   // EMA-filtered risk force [N]

    // Lateral
    public float DeltaLat;    // shared steering angle [rad]
    public float U1Lat;
    public float U2Lat;
    public float LatLambda;
    public float LatForce;

    // Phase
    public MergePhase Phase;

    // Diagnostics (written by NashCoordinator, read by PlatoonDataVisualizer / NashHUD)
    public float GapToLeader;   // bumper-to-bumper gap [m], 999 when no leader
    public float TTC;           // time-to-collision [s], 999 when safe
    public float THW;           // time headway (Wang 2016) [s], 999 when stationary
    public float LongForceRaw;  // DSF force before EMA [N]
    public float R1VTarget;     // system reference velocity at k=0 [m/s]
    public float R2VTarget;     // human  reference velocity at k=0 [m/s]
    public bool  SafetyOverride; // true when 100 Hz hard-brake override is active

    // Longitudinal algorithm state
    public float VxEgo;         // ego longitudinal speed [m/s]
    public float GapErr;        // gap tracking error: desired − actual [m], + = too close
    public float VelErr;        // velocity error: ego.vx − leader.vx [m/s]
    public float LnSync;        // synchronisation progress l_n ∈ [0,1]
    public float R2LongEff;     // effective R2_long after activity scaling
    public float LongActivity;  // driver pedal activity [0,1] (Lazcano scaling input)

    // Lateral algorithm state
    public float YErr;          // lateral error: ego.y − platoonLaneY [m]
    public float ConflictMav;   // conflict moving-average [0,1] (Lazcano conflict boost)
    public float R2LatEff;      // effective R2_lat after activity + conflict scaling
    public float LatActivity;   // driver steering magnitude [0,1]
    public float LnLatSync;     // lateral synchronisation progress l_n ∈ [0,1]
}
