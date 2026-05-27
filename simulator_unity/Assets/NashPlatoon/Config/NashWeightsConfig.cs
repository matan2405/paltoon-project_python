using UnityEngine;

[CreateAssetMenu(menuName = "NashPlatoon/NashWeights")]
public class NashWeightsConfig : ScriptableObject {
    [Header("Long tracking")]
    public float Q_pos = 5000f, Q_vel = 1500f;
    public float Q_pos_terminal = 50000f, Q_vel_terminal = 6000f;

    [Header("Long effort")]
    public float R1_long = 75000f, R2_long = 120000f;

    [Header("Long — Adaptive R2 EMA (R2LongStart → R2LongFollow per l_n)")]
    public float R2LongStart           = 120000f;  // R2 when far from platoon (system leads)
    public float R2LongFollow          =  80000f;  // R2 in steady-state following (human leads)
    public float R2LongEmaAlpha        =    0.05f; // slow EMA smoothing constant
    public float R2LongUpdateThreshold =    50f;   // min |ΔR2| to trigger solver rebuild

    [Header("Long constraints")]
    public float U1Min = -8f, U1Max = 5f;
    public float U2Min = -5f, U2Max = 4f;
    public float Du1Max = 1.5f, Du2Max = 2.0f;
    public float VMin = 0f, VMax = 50f;
    public float GapMin = 7f;

    [Header("Lat tracking")]
    public float Q_y = 800f,   Q_psi = 10000f;
    public float Q_y_terminal = 8000f, Q_psi_terminal = 40000f;

    [Header("Lat effort")]
    public float R1_lat = 50000f, R2_lat = 50000f;

    [Header("Lat — Adaptive R2 EMA (R2LatStart → R2LatFollow per l_n)")]
    public float R2LatStart           =  50000f;  // R2 when far from target lane
    public float R2LatFollow          =  28000f;  // R2 once lane-centred
    public float R2LatEmaAlpha        =    0.05f;
    public float R2LatUpdateThreshold =   200f;

    [Header("Lat constraints")]
    public float DeltaMin = -0.436f, DeltaMax = 0.436f;
    public float DDeltaMax = 0.087f;

    [Header("IBR")]
    public int   IbrMaxIter = 15;
    public float IbrTol = 1e-4f;

    [Header("GP/MA-IDM — Zhang & Sun 2024")]
    public float SigmaK = 1f, Ell = 2f, GammaRisk = 0.5f;
}
