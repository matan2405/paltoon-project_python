// LongitudinalSafetyField — Li et al. (2019) elliptic Driving Safety Field.
//
// Mirrors coordinator.py:
//   _long_dsf_params()       — dynamic s, Mo, Ri, DR with EMA and position/velocity scaling
//   _long_leader_force_raw() — repulsive/attractive leader force
//   _long_follower_force_raw()— repulsive-only follower force
//   _long_safety_force()     — combined force with soft transition in FOLLOWING
//
// Sign convention:
//   F > 0  repulsive (too close — slow down / brake)
//   F < 0  attractive (too far  — speed up / catch-up)
using UnityEngine;

public class LongitudinalSafetyField
{
    readonly SafetyFieldConfig _c;

    // EMA filter state (mirrors coordinator._long_*_filt)
    float _sFilt;
    float _moFilt;
    float _riFilt;
    float _drFilt;
    float _forceFilt;

    public float LastRaw      { get; private set; }  // raw force before EMA (for diagnostics)
    public float LastGap      { get; private set; }  // bumper-to-bumper gap to leader [m]
    public float LastTTC      { get; private set; }  // time-to-collision [s], 999 when safe

    public LongitudinalSafetyField(SafetyFieldConfig cfg)
    {
        _c      = cfg;
        _sFilt  = cfg.LongBaseRadius;
        _moFilt = cfg.LongObstacleMass;
        _riFilt = cfg.LongInfluenceFactor;
        _drFilt = cfg.LongDriverRisk;
    }

    // ── Public entry points ───────────────────────────────────────────────────

    /// <summary>
    /// Variant for Following phase with no real leader: uses a virtual leader
    /// (Kennedy 2023) so LeaderForceRaw receives a valid gap and velocity.
    /// </summary>
    public float ComputeWithVirtual(
        AdvancedBicycleModel ego,
        VirtualLeaderVehicle virt,
        AdvancedBicycleModel follower,
        MergePhase phase)
    {
        float vx = ego.GetVx();
        var (s, mo, ri, dr) = DsfParams(vx, hasLeader: true, hasFollower: follower != null);

        float desGap = _c.StandstillDist + _c.PlatoonTimeGap * vx;

        // Virtual leader force (attractive when ego is too fast, repulsive when too slow)
        float egoZ  = ego.GetPosition().z;
        float egoL  = ego.GetLength();
        float virtZ = virt.GetPositionZ();
        float gap   = virtZ - virt.GetLength() * 0.5f - egoZ - egoL * 0.5f;

        // Virtual mirrors Car1: gap < 0 means data inconsistency, not a real collision.
        if (gap < 0f)
        {
            LastRaw = 0f; LastGap = gap; LastTTC = 999f;
            _forceFilt = _c.LongEmaAlpha * 0f + (1f - _c.LongEmaAlpha) * _forceFilt;
            return _forceFilt;
        }

        float gapErr = desGap - gap;
        float vRel   = vx - virt.GetVx();

        float fLeader;
        if (gap < _c.LongMinSafeDist)
            fLeader = _c.LongMaxForce;
        else if (gap < _c.LongEmergencyDist)
        {
            float t2 = 1f - (gap - _c.LongMinSafeDist) /
                       Mathf.Max(_c.LongEmergencyDist - _c.LongMinSafeDist, 1e-3f);
            fLeader = t2 * _c.LongMaxForce;
        }
        else if (gapErr <= 0f)
        {
            // Gap >= desired: attractive pull toward virtual leader
            fLeader = Mathf.Clamp(
                gapErr / Mathf.Max(desGap, 1f) * _c.LongMaxForce,
                -0.25f * _c.LongMaxForce, 0f);
        }
        else
        {
            float rEll    = gapErr / (s + _c.LongEpsilon);
            float potential = (mo * ri) / ((rEll + _c.LongEpsilon) * (rEll + _c.LongEpsilon));
            float wDist   = Mathf.Exp(-gapErr / _c.LongDistanceDecay);
            float wVel    = Mathf.Exp(Mathf.Max(0f, vRel) / 5f);
            float wRisk   = 1f + dr;
            fLeader = Mathf.Min(potential * wDist * wVel * wRisk, _c.LongMaxForce);
        }

        // Soft transition: quadratic fade near desGap to prevent saturated force at small gapErr.
        // Following: fade threshold = FollowingGapErrorFactor × desGap (comfort, ~7.5m at 50m).
        // Merge:     fade threshold = MergeGapErrorFactor × desGap (tighter, ~4m at 50m) so
        //            full force is still used when gapErr is large (>4m) but fades as ego settles.
        // Critically, FollowerForceRaw is NOT faded — follower safety is unconditional.
        if (phase == MergePhase.Following || phase == MergePhase.Merge)
        {
            float factor = (phase == MergePhase.Following)
                ? _c.FollowingGapErrorFactor : _c.MergeGapErrorFactor;
            float thr = factor * Mathf.Max(desGap, 1f);
            if (thr > 0f)
            {
                float absGapErr = Mathf.Abs(gapErr);
                fLeader *= Mathf.Min(1f, (absGapErr / thr) * (absGapErr / thr));
            }
        }

        float fFollowerRaw = follower != null
            ? FollowerForceRaw(ego, follower, s, mo, ri, dr, desGap) : 0f;
        float fFollower = _c.LongFollowerWeight * fFollowerRaw;
        if (phase == MergePhase.Following && follower != null)
        {
            float thr = _c.FollowingGapErrorFactor * Mathf.Max(desGap, 1f);
            if (thr > 0f)
            {
                float gapR    = ego.GetPosition().z - follower.GetPosition().z - egoL * 0.5f - follower.GetLength() * 0.5f;
                float gapErrR = Mathf.Abs(desGap - gapR);
                fFollower *= Mathf.Min(1f, (gapErrR / thr) * (gapErrR / thr));
            }
        }

        float raw = fLeader + fFollower;
        LastRaw = raw;
        LastGap = gap;
        LastTTC = (vRel > 0.1f && gap > 0f) ? gap / vRel : 999f;

        _forceFilt = _c.LongEmaAlpha * raw + (1f - _c.LongEmaAlpha) * _forceFilt;
        return _forceFilt;
    }

    /// <summary>
    /// Compute total EMA-filtered longitudinal safety force [N].
    /// follower == null when ego is at the platoon tail or has no rear vehicle.
    /// </summary>
    public float Compute(
        AdvancedBicycleModel ego,
        AdvancedBicycleModel leader,
        AdvancedBicycleModel follower,
        MergePhase phase)
    {
        bool hasLeader   = leader   != null;
        bool hasFollower = follower != null;

        var (s, mo, ri, dr) = DsfParams(ego.GetVx(), hasLeader, hasFollower);

        float desGap = _c.StandstillDist + _c.PlatoonTimeGap * ego.GetVx();

        float fLeader = hasLeader
            ? LeaderForceRaw(ego, leader, s, mo, ri, dr, desGap)
            : 0f;

        float fFollowerRaw = hasFollower
            ? FollowerForceRaw(ego, follower, s, mo, ri, dr, desGap)
            : 0f;

        float fFollower = _c.LongFollowerWeight * fFollowerRaw;

        // Quadratic soft transition in FOLLOWING and MERGE.
        // Prevents the elliptic potential (which is >> LongMaxForce for all gapErr>0)
        // from saturating to LongMaxForce when the gap error is already small.
        // Following: factor=FollowingGapErrorFactor (large threshold, comfort priority).
        // Merge:     factor=MergeGapErrorFactor (smaller threshold so full force until ~4m err).
        // FollowerForceRaw is intentionally NOT faded — follower safety is unconditional.
        if (phase == MergePhase.Following || phase == MergePhase.Merge)
        {
            float factor = (phase == MergePhase.Following)
                ? _c.FollowingGapErrorFactor : _c.MergeGapErrorFactor;
            float thr = factor * Mathf.Max(desGap, 1f);
            if (thr > 0f && hasLeader)
            {
                float gap    = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength() / 2f - ego.GetLength() / 2f;
                float gapErr = Mathf.Abs(desGap - gap);
                fLeader *= Mathf.Min(1f, (gapErr / thr) * (gapErr / thr));
            }
            if (phase == MergePhase.Following && thr > 0f && hasFollower)
            {
                float gapR    = ego.GetPosition().z - follower.GetPosition().z - ego.GetLength() / 2f - follower.GetLength() / 2f;
                float gapErrR = Mathf.Abs(desGap - gapR);
                fFollower *= Mathf.Min(1f, (gapErrR / thr) * (gapErrR / thr));
            }
        }

        float raw = fLeader + fFollower;
        LastRaw = raw;

        // Update gap/TTC diagnostics from leader
        if (hasLeader)
        {
            float gap = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength() / 2f - ego.GetLength() / 2f;
            float vRel = ego.GetVx() - leader.GetVx();
            LastGap = gap;
            LastTTC = (vRel > 0.1f && gap > 0f) ? gap / vRel : 999f;

            // Bypass EMA in emergency zone — filter lag causes the authority allocator
            // to see an attenuated force (0.2×Max on the first step), giving the Nash
            // insufficient authority to brake hard before impact.
            if (gap < _c.LongEmergencyDist)
            {
                _forceFilt = raw;
                return _forceFilt;
            }
        }
        else
        {
            LastGap = 999f;
            LastTTC = 999f;
        }

        _forceFilt = _c.LongEmaAlpha * raw + (1f - _c.LongEmaAlpha) * _forceFilt;
        return _forceFilt;
    }

    // ── Dynamic DSF params ────────────────────────────────────────────────────

    /// <summary>
    /// Compute position- and velocity-scaled DSF parameters with EMA blending.
    /// Mirrors coordinator._long_dsf_params().
    /// </summary>
    (float s, float mo, float ri, float dr) DsfParams(float vx, bool hasLeader, bool hasFollower)
    {
        float alpha = _c.LongEmaAlpha;

        // Position multiplier
        float posMult = PositionMult(hasLeader, hasFollower);

        // s: velocity-scaled safety radius
        float velFactor = 1f + _c.LongVelocityScaling * (vx / _c.LongVelocityReference - 1f);
        velFactor = Mathf.Clamp(velFactor, 0.5f, 2f);
        float sTarget = _c.LongBaseRadius * posMult * velFactor;
        _sFilt = alpha * sTarget + (1f - alpha) * _sFilt;

        // Mo: position-weighted obstacle mass
        float moTarget = _c.LongObstacleMass * posMult;
        _moFilt = alpha * moTarget + (1f - alpha) * _moFilt;

        // Ri: platoon coherence in middle position
        float riTarget = _c.LongInfluenceFactor;
        if (hasLeader && hasFollower) riTarget *= _c.LongPlatoonCoherence;
        _riFilt = alpha * riTarget + (1f - alpha) * _riFilt;

        // DR: base driver risk (constant in practice)
        _drFilt = alpha * _c.LongDriverRisk + (1f - alpha) * _drFilt;

        return (_sFilt, _moFilt, _riFilt, _drFilt);
    }

    float PositionMult(bool hasLeader, bool hasFollower)
    {
        if (!hasLeader)   return _c.LongLeaderPosMult;
        if (hasFollower)  return _c.LongMiddlePosMult;
        return _c.LongFollowerPosMult;
    }

    // ── Force computations ────────────────────────────────────────────────────

    float LeaderForceRaw(
        AdvancedBicycleModel ego, AdvancedBicycleModel leader,
        float s, float mo, float ri, float dr, float desGap)
    {
        float gap = leader.GetPosition().z - ego.GetPosition().z - leader.GetLength() / 2f - ego.GetLength() / 2f;

        // Ego overtook its locked leader (gap < 0): return strong attractive (negative) force.
        // Without this, gap<0 satisfies gap<LongMinSafeDist and returns +LongMaxForce (repulsive),
        // which pushes ego further ahead and breaks the authority loop.
        if (gap < 0f)
            return Mathf.Clamp(gap / _c.LongDistanceDecay * _c.LongMaxForce, -_c.LongMaxForce, 0f);

        // Hard safety zones (inner = MIN_SAFE, outer = EMERGENCY_BRAKE)
        if (gap < _c.LongMinSafeDist)
            return _c.LongMaxForce;
        if (gap < _c.LongEmergencyDist)
        {
            float t = 1f - (gap - _c.LongMinSafeDist) /
                      Mathf.Max(_c.LongEmergencyDist - _c.LongMinSafeDist, 1e-3f);
            return t * _c.LongMaxForce;
        }

        float gapErr = desGap - gap;       // >0 too close, <0 too far
        float vRel   = ego.GetVx() - leader.GetVx();  // >0 closing

        if (gapErr <= 0f)
        {
            // Gap >= desired: linear attractive pull (bidirectional, capped)
            return Mathf.Clamp(
                gapErr / Mathf.Max(desGap, 1f) * _c.LongMaxForce,
                -0.25f * _c.LongMaxForce, 0f);
        }

        float rEll    = gapErr / (s + _c.LongEpsilon);
        float potential = (mo * ri) / ((rEll + _c.LongEpsilon) * (rEll + _c.LongEpsilon));
        float wDist   = Mathf.Exp(-gapErr / _c.LongDistanceDecay);
        float wVel    = Mathf.Exp(Mathf.Max(0f, vRel) / 5f);
        float wRisk   = 1f + dr;
        float elliptic = potential * wDist * wVel * wRisk;

        // TTC floor: active whenever gap < desGap (ego is too close relative to desired).
        // This prevents overtaking the locked leader at any distance, not just when
        // physically adjacent. Without this guard the ego accelerates past the leader
        // during GapSearch because the elliptic potential alone is too weak at mid-range.
        // TtcCritical = 1.5 s follows FHWA SSAM / Hayward (1972).
        float ttc      = (vRel > 0.1f && gap > 0f) ? gap / vRel : 999f;
        float ttcFrac  = 1f - Mathf.Clamp01(
            (ttc - _c.LongTtcCritical) /
            Mathf.Max(_c.LongTtcWarn - _c.LongTtcCritical, 1e-3f));
        float ttcFloor = ttcFrac * _c.LongMaxForce;

        return Mathf.Min(Mathf.Max(elliptic, ttcFloor), _c.LongMaxForce);
    }

    float FollowerForceRaw(
        AdvancedBicycleModel ego, AdvancedBicycleModel follower,
        float s, float mo, float ri, float dr, float desGap)
    {
        float gapR = ego.GetPosition().z - follower.GetPosition().z - ego.GetLength() / 2f - follower.GetLength() / 2f;

        // Locked follower overtook ego (fast platoon passed stationary ego).
        // gapR < 0 means the vehicle is now ahead of ego — no follower force applies.
        if (gapR < 0f) return 0f;

        if (gapR < _c.LongMinSafeDist)
            return _c.LongMaxForce;
        if (gapR < _c.LongEmergencyDist)
        {
            float t = 1f - (gapR - _c.LongMinSafeDist) /
                      Mathf.Max(_c.LongEmergencyDist - _c.LongMinSafeDist, 1e-3f);
            return t * _c.LongMaxForce;
        }

        float gapErr = desGap - gapR;   // >0 follower too close
        if (gapErr <= 0f) return 0f;    // follower is safe distance away

        float vRel     = follower.GetVx() - ego.GetVx();  // >0 follower closing
        float rEll     = gapErr / (s + _c.LongEpsilon);
        float potential = (mo * ri) / ((rEll + _c.LongEpsilon) * (rEll + _c.LongEpsilon));
        float wDist    = Mathf.Exp(-gapErr / _c.LongDistanceDecay);
        float wVel     = Mathf.Exp(Mathf.Max(0f, vRel) / 5f);
        float wRisk    = 1f + dr;
        return Mathf.Min(potential * wDist * wVel * wRisk, _c.LongMaxForce);
    }
}
