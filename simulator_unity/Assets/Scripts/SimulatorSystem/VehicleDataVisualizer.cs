using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;

// Writes one CSV per vehicle:
//
//   vehicle_{name}.csv   — platoon vehicle (longitudinal-only; no lateral columns)
//   ego_vehicle.csv      — ego/Player vehicle (full 6-DOF state + all inputs)
//
// Column selection is based on whether the Rigidbody is tagged "Player".
// Lateral/yaw columns (Y, vy, yDot, Psi, PsiDot) are omitted for platoon
// vehicles because they never change lane and their values are always zero.
// EngineRpm is included for all vehicles (useful for powertrain verification).
[RequireComponent(typeof(AdvancedBicycleModel))]
public class VehicleDataVisualizer : MonoBehaviour
{
    // ── Shared fields (all vehicles) ──────────────────────────────────────────
    struct CommonRow
    {
        public float time;
        public float posZ;        // forward position [m]  (Unity Z = forward)
        public float vx;          // longitudinal speed [m/s]
        public float ax;          // longitudinal acceleration [m/s²]
        public float throttle;    // normalised throttle [0,1]
        public float brake;       // normalised brake [0,1]
        public float fx_tot;      // net longitudinal force [N]  (Fx_contact − Fx_resist)
        public float fx_contact;  // contact-patch drive/brake force [N]
        public float fx_resist;   // drag + rolling + grade resistance [N]
        public int   gear;
        public float engineRpm;
    }

    // ── Ego-only extra fields ─────────────────────────────────────────────────
    struct EgoExtraRow
    {
        public float posX;     // lateral world position [m]  (Unity X = lateral)
        public float steering; // normalised steering [-1,1]
        public float lambda;   // front wheel angle [rad]
        public float vy;       // body-frame lateral velocity [m/s]
        public float yDot;     // world-frame lateral velocity [m/s]
        public float psi;      // yaw angle [rad]
        public float psiDot;   // yaw rate [rad/s]
    }

    private readonly List<CommonRow>   _common = new List<CommonRow>();
    private readonly List<EgoExtraRow> _ego    = new List<EgoExtraRow>();

    private AdvancedBicycleModel _vehicle;
    private bool                 _isEgo;
    private string               _csvFilename; // locked at Start, stable for entire run

    void Start()
    {
        _vehicle = GetComponent<AdvancedBicycleModel>();
        var rb   = GetComponent<Rigidbody>();
        _isEgo   = rb != null && rb.CompareTag("Player");
        // Use the GameObject name (e.g. "Car1", "Car2") as the stable file identifier.
        // Platoon index is reassigned every time SetupPlatoon() is called, so it
        // cannot be used as a stable filename across a run.
        _csvFilename = _isEgo ? "ego_vehicle.csv" : $"vehicle_{name}.csv";
    }

    void FixedUpdate()
    {
        var row = new CommonRow
        {
            time        = Time.fixedTime,
            posZ        = _vehicle.transform.position.z,
            vx          = _vehicle.GetVx(),
            ax          = _vehicle.GetAx(),
            throttle    = _vehicle.GetThrottleRaw(),
            brake       = _vehicle.GetBrakeRaw(),
            fx_tot      = _vehicle.GetFx_tot(),
            fx_contact  = _vehicle.GetFx_contact(),
            fx_resist   = _vehicle.GetFx_resist(),
            gear        = _vehicle.GetCurrentGear(),
            engineRpm   = _vehicle.GetEngineRpm(),
        };
        _common.Add(row);

        if (_isEgo)
        {
            _ego.Add(new EgoExtraRow
            {
                posX     = _vehicle.transform.position.x,
                steering = _vehicle.GetSteeringRaw(),
                lambda   = _vehicle.GetLambda(),
                vy       = _vehicle.GetVy(),
                yDot     = _vehicle.GetYDot(),
                psi      = _vehicle.GetPsi(),
                psiDot   = _vehicle.GetPsiDot(),
            });
        }
    }

    void SaveDataToCSVInternal()
    {
        if (_common.Count == 0)
        {
            Debug.LogWarning($"[VehicleDataVisualizer] {name}: no data to save.");
            return;
        }

        string path = Path.Combine(Application.dataPath, "..", "VehicleData", "data", _csvFilename);
        Directory.CreateDirectory(Path.GetDirectoryName(path));

        var sb = new StringBuilder();

        if (_isEgo)
        {
            sb.AppendLine("Time,posZ,posX,vx,ax,vy,yDot,psi,psiDot," +
                          "throttle,brake,steering,lambda,fx_contact,fx_resist,fx_tot,gear,engineRpm");
            for (int i = 0; i < _common.Count; i++)
            {
                var c = _common[i];
                var e = _ego[i];
                sb.AppendLine(
                    $"{c.time:G6},{c.posZ:G6},{e.posX:G6},{c.vx:G6},{c.ax:G6}," +
                    $"{e.vy:G6},{e.yDot:G6},{e.psi:G6},{e.psiDot:G6}," +
                    $"{c.throttle:G6},{c.brake:G6},{e.steering:G6},{e.lambda:G6}," +
                    $"{c.fx_contact:G6},{c.fx_resist:G6},{c.fx_tot:G6},{c.gear},{c.engineRpm:G6}");
            }
        }
        else
        {
            sb.AppendLine("Time,posZ,vx,ax,throttle,brake,fx_contact,fx_resist,fx_tot,gear,engineRpm");
            foreach (var c in _common)
            {
                sb.AppendLine(
                    $"{c.time:G6},{c.posZ:G6},{c.vx:G6},{c.ax:G6}," +
                    $"{c.throttle:G6},{c.brake:G6}," +
                    $"{c.fx_contact:G6},{c.fx_resist:G6},{c.fx_tot:G6},{c.gear},{c.engineRpm:G6}");
            }
        }

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[VehicleDataVisualizer] {name} → {path}  ({_common.Count} rows)");
    }

    bool _saved = false;

    public void SaveDataToCSV()
    {
        if (_saved) return;
        _saved = true;
        SaveDataToCSVInternal();
    }

    void OnDestroy() => SaveDataToCSV();
}
