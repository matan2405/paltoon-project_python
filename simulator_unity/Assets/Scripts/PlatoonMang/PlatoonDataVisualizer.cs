using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;

// Writes two CSV files per simulation run:
//
//   platoon_gaps.csv  — one row per FixedUpdate, _nPlatoon gap slots.
//                       Slot k covers the pair at that convoy position at each timestep.
//                       Before ego joins: slot _nPlatoon-1 is NaN (ego not yet inserted).
//                       After ego joins:  ego occupies one slot; pairs re-map continuously.
//                       Empty cell = NaN in Python (no stitching needed).
//
//   ego_nash.csv      — Nash coordinator telemetry per FixedUpdate (zeros when Nash off).
//
[RequireComponent(typeof(PlatoonManager))]
public class PlatoonDataVisualizer : MonoBehaviour
{
    // ── One row in platoon_gaps.csv ───────────────────────────────────────────
    struct GapRow
    {
        public float   time;
        public string  phase;
        public float   target_vx_kmh;
        public float[] actual;  // bumper-to-bumper [m], NaN when slot not occupied
        public float[] des;     // desired gap (Rajamani) [m]
    }

    // ── One row in ego_nash.csv ───────────────────────────────────────────────
    struct NashRow
    {
        public float time;
        public string phase;
        // Longitudinal Nash
        public float u_long;
        public float u1_long;
        public float u2_long;
        public float lambda_long;
        public float long_force;
        public float long_force_raw;
        public float gap_to_leader;
        public float des_gap_leader;  // = gap_to_leader + gap_err
        public float gap_err;
        public float vel_err;
        public float ttc;
        public float ln_sync;
        public float r2_long_eff;
        public float long_activity;
        // Lateral Nash
        public float delta_lat;
        public float u1_lat;
        public float u2_lat;
        public float lambda_lat;
        public float lat_force;
        public float y_err;
        public float r2_lat_eff;
        public float lat_activity;
        public float ln_lat_sync;
        public float conflict_mav;
        // Diagnostics
        public float vx_ego;
        public bool  safety_override;
    }

    private readonly List<GapRow>  _gapRows  = new List<GapRow>();
    private readonly List<NashRow> _nashRows = new List<NashRow>();

    private PlatoonManager _platoonManager;
    // Number of gap slots = nPlatoon (fixed at Start, never changes).
    // When ego joins at index k:  slot k-1 = leader→ego, slot k = ego→next.
    private int _nPlatoon;

    [Header("Nash Logging (optional)")]
    [SerializeField] NashCoordinator nashCoordinator;

    static string PhaseName(MergePhase p) =>
        p == MergePhase.Approach ? "Approach" :
        p == MergePhase.GapSearch ? "GapSearch" :
        p == MergePhase.Merge ? "Merge" :
        p == MergePhase.Following ? "Following" : p.ToString();

    static float BumperGap(AdvancedBicycleModel ahead, AdvancedBicycleModel behind) =>
        ahead.GetPosition().z - behind.GetPosition().z
        - ahead.GetVehicleParameters().length / 2f
        - behind.GetVehicleParameters().length / 2f;

    static float RajamaniDes(float headway, float standstillDist, AdvancedBicycleModel follower) =>
        standstillDist + headway * follower.GetVx();

    void Start()
    {
        _platoonManager = GetComponent<PlatoonManager>();
        var v  = _platoonManager.GetPlatoonVehicles();
        // _nPlatoon gap-slots = nPlatoon vehicles (one slot per consecutive pair + ego slot)
        _nPlatoon = v != null ? v.Length : 0;
    }

    void FixedUpdate()
    {
        var  vehicles    = _platoonManager.GetPlatoonVehicles();
        var  ego         = _platoonManager.egoVehicle;
        bool egoIn       = _platoonManager._egoInPlatoon;
        var  nashPhase   = nashCoordinator != null ? nashCoordinator.Data.Phase : MergePhase.Approach;

        // ── Gap row ───────────────────────────────────────────────────────────
        float[] actual = new float[_nPlatoon];
        float[] des    = new float[_nPlatoon];
        for (int s = 0; s < _nPlatoon; s++) { actual[s] = float.NaN; des[s] = float.NaN; }

        if (vehicles != null && vehicles.Length >= 2)
        {
            int slot = 0;
            for (int i = 0; i < vehicles.Length - 1 && slot < _nPlatoon; i++)
            {
                actual[slot] = BumperGap(vehicles[i], vehicles[i + 1]);
                des[slot]    = RajamaniDes(_platoonManager.TimeGap, _platoonManager.StandstillDist, vehicles[i + 1]);
                slot++;
            }
        }

        _gapRows.Add(new GapRow
        {
            time          = Time.fixedTime,
            phase         = PhaseName(nashPhase),
            target_vx_kmh = _platoonManager.GetTargetVelocity() * 3.6f,
            actual        = actual,
            des           = des,
        });

        // ── Nash row ──────────────────────────────────────────────────────────
        var nRow = new NashRow { time = Time.fixedTime, phase = PhaseName(nashPhase) };

        if (nashCoordinator != null)
        {
            var d = nashCoordinator.Data;
            nRow.u_long         = d.ULong;
            nRow.u1_long        = d.U1Long;
            nRow.u2_long        = d.U2Long;
            nRow.lambda_long    = d.LongLambda;
            nRow.long_force     = d.LongForce;
            nRow.long_force_raw = d.LongForceRaw;
            nRow.gap_to_leader  = d.GapToLeader;
            nRow.des_gap_leader = d.GapToLeader + d.GapErr;
            nRow.gap_err        = d.GapErr;
            nRow.vel_err        = d.VelErr;
            nRow.ttc            = d.TTC;
            nRow.ln_sync        = d.LnSync;
            nRow.r2_long_eff    = d.R2LongEff;
            nRow.long_activity  = d.LongActivity;
            nRow.delta_lat      = d.DeltaLat;
            nRow.u1_lat         = d.U1Lat;
            nRow.u2_lat         = d.U2Lat;
            nRow.lambda_lat     = d.LatLambda;
            nRow.lat_force      = d.LatForce;
            nRow.y_err          = d.YErr;
            nRow.r2_lat_eff     = d.R2LatEff;
            nRow.lat_activity   = d.LatActivity;
            nRow.ln_lat_sync    = d.LnLatSync;
            nRow.conflict_mav   = d.ConflictMav;
            nRow.vx_ego         = d.VxEgo;
            nRow.safety_override = d.SafetyOverride;
        }

        _nashRows.Add(nRow);
    }

    public void SaveDataToCSV()
    {
        if (_gapRows.Count == 0)
        {
            Debug.LogWarning("[PlatoonDataVisualizer] No data to save.");
            return;
        }

        string dir = Path.Combine(Application.dataPath, "..", "VehicleData", "data");
        Directory.CreateDirectory(dir);

        SaveGapsCsv(dir);
        SaveNashCsv(dir);
    }

    static string F(float v) => float.IsNaN(v) ? "" : v.ToString("G6");

    void SaveGapsCsv(string dir)
    {
        var sb = new StringBuilder();

        sb.Append("Time,phase,target_vx_kmh");
        for (int s = 0; s < _nPlatoon; s++) sb.Append($",actual_gap_{s}");
        for (int s = 0; s < _nPlatoon; s++) sb.Append($",des_gap_{s}");
        sb.AppendLine();

        foreach (var r in _gapRows)
        {
            sb.Append($"{r.time:G6},{r.phase},{r.target_vx_kmh:F4}");
            for (int s = 0; s < _nPlatoon; s++) sb.Append($",{F(r.actual[s])}");
            for (int s = 0; s < _nPlatoon; s++) sb.Append($",{F(r.des[s])}");
            sb.AppendLine();
        }

        string path = Path.Combine(dir, "platoon_gaps.csv");
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[PlatoonDataVisualizer] Saved {_gapRows.Count} rows → {path}");
    }

    void SaveNashCsv(string dir)
    {
        var sb = new StringBuilder();
        sb.AppendLine(
            "Time,phase," +
            "u_long,u1_long,u2_long,lambda_long,long_force,long_force_raw," +
            "gap_to_leader,des_gap_leader,gap_err,vel_err,ttc,ln_sync,r2_long_eff,long_activity," +
            "delta_lat,u1_lat,u2_lat,lambda_lat,lat_force,y_err,r2_lat_eff,lat_activity,ln_lat_sync,conflict_mav," +
            "vx_ego,safety_override");

        foreach (var r in _nashRows)
        {
            sb.AppendLine(
                $"{r.time:G6},{r.phase}," +
                $"{r.u_long:G6},{r.u1_long:G6},{r.u2_long:G6},{r.lambda_long:G6},{r.long_force:G6},{r.long_force_raw:G6}," +
                $"{r.gap_to_leader:G6},{r.des_gap_leader:G6},{r.gap_err:G6},{r.vel_err:G6},{r.ttc:G6},{r.ln_sync:G6},{r.r2_long_eff:G6},{r.long_activity:G6}," +
                $"{r.delta_lat:G6},{r.u1_lat:G6},{r.u2_lat:G6},{r.lambda_lat:G6},{r.lat_force:G6},{r.y_err:G6},{r.r2_lat_eff:G6},{r.lat_activity:G6},{r.ln_lat_sync:G6},{r.conflict_mav:G6}," +
                $"{r.vx_ego:G6},{(r.safety_override ? 1 : 0)}");
        }

        string path = Path.Combine(dir, "ego_nash.csv");
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[PlatoonDataVisualizer] Saved {_nashRows.Count} rows → {path}");
    }

    void OnDestroy() => SaveDataToCSV();
}
