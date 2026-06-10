// MOBILLaneChange — Kesting, Treiber, Helbing (2007) lane-change decision model.
//
// Mirrors unified/control/mobil_lane_change.py.  Used by NashCoordinator to gate
// the APPROACH → GAP_SEARCH phase transition with a proper MOBIL criterion instead
// of a naive distance check.
//
// Criterion (Mandatory mode, default for platoon merge):
//   safety:    a_n_tilde >= -b_safe       — new follower won't need to brake harder
//   gaps:      front/rear gaps >= min_gap (or no neighbour in that direction)
//   approved   = safety AND gaps
//
// Discretionary mode also checks the incentive:
//   incentive  = (a_c_tilde - a_c) + p·(a_n_tilde - a_n) > a_th
//
// Acceleration model: standard IDM (Intelligent Driver Model, Kesting et al.).
using UnityEngine;

public class MOBILLaneChange
{
    // ── IDM parameters (Kesting et al. defaults) ──────────────────────────────
    public struct IDMParams
    {
        public float v0;       // desired speed [m/s]
        public float T;        // safe time gap [s]
        public float a;        // comfortable acceleration [m/s²]
        public float b;        // comfortable braking [m/s²]
        public float s0;       // minimum gap [m]
        public float delta;    // free-road exponent
        public float L;        // vehicle length [m]

        public static IDMParams Default() => new IDMParams {
            v0 = 33.3f, T = 1.2f, a = 1.5f, b = 2.0f,
            s0 = 2f, delta = 4f, L = 4.0f
        };
    }

    // ── MOBIL parameters (Kesting et al. Table 1) ─────────────────────────────
    public struct MOBILParams
    {
        public float p;          // politeness [0..1]
        public float bSafe;      // max acceptable decel for follower [m/s²]
        public float aTh;        // incentive threshold [m/s²]
        public bool  mandatory;  // true for platoon merge (no incentive check)
        public float minGap;     // hard gap floor for safety [m]

        public static MOBILParams Default() => new MOBILParams {
            p = 0.5f, bSafe = 4.0f, aTh = 0.1f,
            mandatory = true, minGap = 8.0f
        };
    }

    public IDMParams   Idm   = IDMParams.Default();
    public MOBILParams Mobil = MOBILParams.Default();
    public string      LastStatus { get; private set; } = "no evaluation";

    // ── Politeness presets per driver type ────────────────────────────────────
    public void SetPoliteness(string driverType)
    {
        switch (driverType)
        {
            case "cautious":   Mobil.p = 0.8f; Mobil.aTh = 0.4f; break;
            case "aggressive": Mobil.p = 0.2f; Mobil.aTh = 0.1f; break;
            default:           Mobil.p = 0.5f; Mobil.aTh = 0.2f; break;  // normal
        }
    }

    // ── IDM helper functions ──────────────────────────────────────────────────

    float IdmDesiredGap(float v, float deltaV)
    {
        float interaction = Mathf.Max(0f, (v * deltaV) / (2f * Mathf.Sqrt(Idm.a * Idm.b)));
        return Idm.s0 + v * Idm.T + interaction;
    }

    float IdmAccel(float v, float s, float deltaV)
    {
        float freeRoad = Idm.v0 > 0.1f ? 1f - Mathf.Pow(v / Idm.v0, Idm.delta) : 0f;
        float interaction;
        if (s > 0.1f)
        {
            float sStar = IdmDesiredGap(v, deltaV);
            interaction = (sStar / s) * (sStar / s);
        }
        else { interaction = 100f; }
        return Mathf.Clamp(Idm.a * (freeRoad - interaction), -Idm.b * 2f, Idm.a);
    }

    float IdmFreeRoadAccel(float v) =>
        Idm.v0 > 0.1f ? Idm.a * (1f - Mathf.Pow(v / Idm.v0, Idm.delta)) : 0f;

    // ── Core MOBIL decision ───────────────────────────────────────────────────

    /// <summary>
    /// Check whether changing lanes from {leader} to {newLeader/newFollower} is
    /// safe (and beneficial in discretionary mode). Pass null for absent neighbour.
    /// </summary>
    public bool CheckLaneChange(
        float egoX, float egoV,
        float? leaderX, float? leaderV,
        float? newFollowerX, float? newFollowerV,
        float? newLeaderX, float? newLeaderV)
    {
        float L = Idm.L;

        // a_c: ego's current acceleration in source lane
        float aC = leaderX.HasValue && leaderV.HasValue
            ? IdmAccel(egoV, leaderX.Value - egoX - L, egoV - leaderV.Value)
            : IdmFreeRoadAccel(egoV);

        // a_c_tilde: ego's acceleration after merge
        float aCtilde = newLeaderX.HasValue && newLeaderV.HasValue
            ? IdmAccel(egoV, newLeaderX.Value - egoX - L, egoV - newLeaderV.Value)
            : IdmFreeRoadAccel(egoV);

        // New follower accelerations (a_n: before, a_n_tilde: after)
        float aN = 0f, aNtilde = 0f;
        if (newFollowerX.HasValue && newFollowerV.HasValue)
        {
            float nfx = newFollowerX.Value, nfv = newFollowerV.Value;
            aNtilde = IdmAccel(nfv, egoX - nfx - L, nfv - egoV);
            aN = newLeaderX.HasValue
                ? IdmAccel(nfv, newLeaderX.Value - nfx - L, nfv - newLeaderV.Value)
                : IdmFreeRoadAccel(nfv);
        }

        // Safety criterion
        bool safetyOk = aNtilde >= -Mobil.bSafe;

        // Gap check
        bool gapFrontOk = true, gapRearOk = true;
        float? gapFront = null, gapRear = null;
        if (newLeaderX.HasValue)
        {
            gapFront = newLeaderX.Value - egoX - L;
            gapFrontOk = newFollowerX.HasValue
                ? gapFront.Value >= Mobil.minGap
                : gapFront.Value >= -L;
        }
        if (newFollowerX.HasValue)
        {
            gapRear = egoX - newFollowerX.Value - L;
            gapRearOk = newLeaderX.HasValue
                ? gapRear.Value >= Mobil.minGap
                : gapRear.Value >= -L;
        }
        bool gapsOk = gapFrontOk && gapRearOk;

        // Incentive (only checked in discretionary mode)
        float incentive = (aCtilde - aC) + Mobil.p * (aNtilde - aN);
        bool incentiveOk = incentive > Mobil.aTh;

        bool approved = Mobil.mandatory
            ? (safetyOk && gapsOk)
            : (safetyOk && incentiveOk && gapsOk);

        LastStatus = $"MOBIL {(approved ? "APPROVED" : "REJECTED")}: safety={safetyOk}, "
                   + $"gaps={gapsOk} (front={gapFront?.ToString("F1") ?? "-"}, "
                   + $"rear={gapRear?.ToString("F1") ?? "-"}), "
                   + $"a_n_tilde={aNtilde:F2}";
        return approved;
    }

    // ── High-level platoon merge check ────────────────────────────────────────

    /// <summary>
    /// Determines new neighbours based on ego position relative to platoon
    /// (front / middle / rear), then delegates to CheckLaneChange.
    /// </summary>
    public bool CheckPlatoonMerge(AdvancedBicycleModel ego, AdvancedBicycleModel[] platoon)
    {
        if (platoon == null || platoon.Length == 0)
        {
            LastStatus = "MOBIL APPROVED: no platoon";
            return true;
        }

        // Sort by x descending (front to back)
        var sorted = (AdvancedBicycleModel[])platoon.Clone();
        System.Array.Sort(sorted, (a, b) => b.GetPosition().z.CompareTo(a.GetPosition().z));

        float ex = ego.GetPosition().z;
        float ev = ego.GetVx();

        float? nlX = null, nlV = null;     // new leader
        float? nfX = null, nfV = null;     // new follower

        if (ex > sorted[0].GetPosition().z)
        {
            // BEFORE platoon — ego is in front, only a follower behind
            nfX = sorted[0].GetPosition().z;
            nfV = sorted[0].GetVx();
        }
        else if (ex < sorted[sorted.Length - 1].GetPosition().z)
        {
            // AFTER platoon — ego is behind, only a leader in front
            nlX = sorted[sorted.Length - 1].GetPosition().z;
            nlV = sorted[sorted.Length - 1].GetVx();
        }
        else
        {
            // MIDDLE — find surrounding pair
            for (int i = 0; i < sorted.Length - 1; i++)
            {
                float frontX = sorted[i].GetPosition().z;
                float backX  = sorted[i + 1].GetPosition().z;
                if (backX < ex && ex < frontX)
                {
                    nlX = frontX; nlV = sorted[i].GetVx();
                    nfX = backX;  nfV = sorted[i + 1].GetVx();
                    break;
                }
            }
        }

        return CheckLaneChange(
            egoX: ex, egoV: ev,
            leaderX: null, leaderV: null,      // ego in adjacent lane has no leader yet
            newFollowerX: nfX, newFollowerV: nfV,
            newLeaderX: nlX, newLeaderV: nlV);
    }
}
