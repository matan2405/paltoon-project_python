// NashHUD — runtime UI overlay for NashCoordinator diagnostics.
//
// Displays the same key signals logged by NashCoordinator.Data:
//   • Phase             — APPROACH / GAP_SEARCH / MERGE / FOLLOWING
//   • λ_long, λ_lat     — authority ratios (0.1 = human, ∞ = system)
//   • u_long, δ_lat     — final shared control outputs
//   • F_long, F_lat     — EMA-filtered DSF risk forces
//   • Gap to leader     — actual bumper-to-bumper gap [m]
//   • IBR iter / conv   — solver diagnostics (optional, requires solver accessor)
//
// Mirrors visual style of Game_Manager.cs (TMP_Text labels).
using TMPro;
using UnityEngine;

public class NashHUD : MonoBehaviour
{
    [Header("Required wiring")]
    [SerializeField] NashCoordinator coordinator;

    [Header("Optional: gap source")]
    [SerializeField] PlatoonManager platoonManager;
    [Tooltip("Index of the gap to display (1 = first gap behind leader)")]
    [SerializeField] int            gapIndex = 1;

    [Header("Phase / Safety")]
    [SerializeField] TMP_Text phaseText;
    [SerializeField] TMP_Text safetyOverrideText;

    [Header("Longitudinal — outputs")]
    [SerializeField] TMP_Text uLongText;
    [SerializeField] TMP_Text u1LongText;
    [SerializeField] TMP_Text lambdaLongText;

    [Header("Longitudinal — vehicle state")]
    [SerializeField] TMP_Text vxEgoText;
    [SerializeField] TMP_Text gapText;
    [SerializeField] TMP_Text gapErrText;
    [SerializeField] TMP_Text velErrText;
    [SerializeField] TMP_Text ttcText;

    [Header("Longitudinal — algorithm state")]
    [SerializeField] TMP_Text lnSyncText;
    [SerializeField] TMP_Text r2LongText;
    [SerializeField] TMP_Text longActivityText;
    [SerializeField] TMP_Text longForceText;
    [SerializeField] TMP_Text r1VTargetText;
    [SerializeField] TMP_Text r2VTargetText;

    [Header("Lateral — outputs")]
    [SerializeField] TMP_Text deltaLatText;
    [SerializeField] TMP_Text u1LatText;
    [SerializeField] TMP_Text lambdaLatText;

    [Header("Lateral — vehicle state")]
    [SerializeField] TMP_Text yErrText;
    [SerializeField] TMP_Text latForceText;

    [Header("Lateral — algorithm state")]
    [SerializeField] TMP_Text lnLatText;
    [SerializeField] TMP_Text r2LatText;
    [SerializeField] TMP_Text conflictMavText;
    [SerializeField] TMP_Text latActivityText;

    void Update()
    {
        if (coordinator == null) return;
        var d = coordinator.Data;

        // Phase / Safety
        if (phaseText          != null) phaseText.text          = $"Phase: {d.Phase}";
        if (safetyOverrideText != null) safetyOverrideText.text = d.SafetyOverride ? "!! SAFETY BRAKE !!" : "";

        // Longitudinal — outputs
        if (uLongText      != null) uLongText.text      = $"u = {d.ULong:+0.00;-0.00} m/s²";
        if (u1LongText     != null) u1LongText.text     = $"u1 = {d.U1Long:+0.00;-0.00}  u2 = {d.U2Long:+0.00;-0.00}";
        if (lambdaLongText != null) lambdaLongText.text = $"λL = {d.LongLambda:F2}";

        // Longitudinal — vehicle state
        if (vxEgoText  != null) vxEgoText.text  = $"vx = {d.VxEgo:F1} m/s";
        if (gapText    != null)
        {
            if (d.GapToLeader < 999f)
                gapText.text = $"Gap = {d.GapToLeader:F1} m";
            else if (platoonManager != null)
                gapText.text = $"Gap = {platoonManager.GetActualGap(gapIndex):F1} / {platoonManager.GetDesiredGap(gapIndex):F1} m";
        }
        if (gapErrText != null) gapErrText.text = $"ΔGap = {d.GapErr:+0.0;-0.0} m";
        if (velErrText != null) velErrText.text = $"Δvx = {d.VelErr:+0.00;-0.00} m/s";
        if (ttcText    != null) ttcText.text    = d.TTC < 999f ? $"TTC = {d.TTC:F1} s" : "TTC = —";

        // Longitudinal — algorithm state
        if (lnSyncText       != null) lnSyncText.text       = $"lₙ = {d.LnSync:F2}";
        if (r2LongText       != null) r2LongText.text       = $"R2L = {d.R2LongEff:F1}";
        if (longActivityText != null) longActivityText.text = $"act_L = {d.LongActivity:F2}";
        if (longForceText    != null) longForceText.text    = $"FL = {d.LongForce:F0} (raw {d.LongForceRaw:F0}) N";
        if (r1VTargetText    != null) r1VTargetText.text    = $"R1v = {d.R1VTarget:F1}  R2v = {d.R2VTarget:F1} m/s";
        if (r2VTargetText    != null) r2VTargetText.text    = $"R2v = {d.R2VTarget:F1} m/s";

        // Lateral — outputs
        if (deltaLatText  != null) deltaLatText.text  = $"δ = {d.DeltaLat * Mathf.Rad2Deg:F1}°";
        if (u1LatText     != null) u1LatText.text     = $"u1T = {d.U1Lat:+0.000;-0.000}  u2T = {d.U2Lat:+0.000;-0.000}";
        if (lambdaLatText != null) lambdaLatText.text = $"λT = {d.LatLambda:F2}";

        // Lateral — vehicle state
        if (yErrText    != null) yErrText.text    = $"yErr = {d.YErr:+0.2;-0.2} m";
        if (latForceText!= null) latForceText.text = $"FT = {d.LatForce:F0} N";

        // Lateral — algorithm state
        if (lnLatText        != null) lnLatText.text        = $"lₙT = {d.LnLatSync:F2}";
        if (r2LatText        != null) r2LatText.text        = $"R2T = {d.R2LatEff:F1}";
        if (conflictMavText  != null) conflictMavText.text  = $"conflict = {d.ConflictMav:F2}";
        if (latActivityText  != null) latActivityText.text  = $"act_T = {d.LatActivity:F2}";
    }
}
