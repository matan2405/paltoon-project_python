// VirtualLeaderVehicle — phantom Car1 that converges to platoon target speed.
//
// When ego joins the platoon as the first vehicle (no real leader ahead),
// this virtual vehicle is placed desGap(Car1.vx) ahead and runs
// FreeRoadAcceleration toward platoonTargetVelocity — exactly as Car1 does.
// This guarantees the virtual is always ahead of ego (gap never negative) while
// guiding ego to the correct cruise speed rather than Car1's transient speed.
using UnityEngine;

public class VirtualLeaderVehicle
{
    float _halfEgoLen;
    float _halfVirtLen;
    float _virtZ;
    float _virtVx;
    float _targetVx;
    float _aMax;
    float _aMin;

    public bool IsActive { get; private set; }

    public float GetLength() => _halfVirtLen * 2f;

    // ── Activate: call once when transitioning to Following with no real leader ──

    /// <summary>
    /// Arm the virtual leader.
    /// initialVx     — Car1's current velocity [m/s] (virtual starts here)
    /// targetVx      — platoon cruise speed [m/s]    (virtual converges here)
    /// aMax / aMin   — acceleration bounds [m/s²] (positive values)
    /// </summary>
    public void Activate(float egoZ, float egoLen, float initialVx, float targetVx,
                         float standstill, float timeGap, float aMax, float aMin)
    {
        _halfEgoLen  = egoLen * 0.5f;
        _halfVirtLen = _halfEgoLen;
        _virtVx      = initialVx;
        _targetVx    = targetVx;
        _aMax        = aMax;
        _aMin        = aMin;

        float desGapNow = standstill + timeGap * initialVx;
        _virtZ = egoZ + _halfEgoLen + desGapNow + _halfVirtLen;

        IsActive = true;
        Debug.Log($"[VirtualLeader] Activated: egoZ={egoZ:F1} initialVx={initialVx:F2} m/s " +
                  $"targetVx={targetVx:F2} m/s desGap={desGapNow:F1} m virtZ={_virtZ:F1}");
    }

    public void Deactivate()
    {
        IsActive = false;
    }

    // ── Step: advance with FreeRoadAcceleration toward target speed ─────────────

    /// <summary>
    /// Call once per FixedUpdate. Runs same FreeRoadAcceleration as Car1.
    /// </summary>
    public void Step(float dt)
    {
        if (!IsActive) return;
        float a  = PlatoonManager.FreeRoadAcceleration(_virtVx, _targetVx, _aMax, _aMin);
        _virtVx  = Mathf.Clamp(_virtVx + a * dt, 0f, _targetVx * 1.5f);
        _virtZ  += _virtVx * dt;
    }

    // ── Getters used by LongitudinalSafetyField and RunLongNashStep ─────────────

    public float GetPositionZ() => _virtZ;
    public float GetVx()        => _virtVx;

    // ── Derived helpers ──────────────────────────────────────────────────────────

    /// <summary>Bumper-to-bumper gap from ego to virtual leader [m].</summary>
    public float BumperGap(float egoZ, float egoLen)
        => _virtZ - _halfVirtLen - egoZ - egoLen * 0.5f;

    /// <summary>
    /// Gap error = desiredGap(egoVx) - actualGap.
    /// Converges to zero when ego matches platoon leader speed and gap.
    /// </summary>
    public float GapError(float standstill, float timeGap, float egoZ, float egoLen, float egoVx)
    {
        float desired = standstill + timeGap * egoVx;
        float actual  = BumperGap(egoZ, egoLen);
        return desired - actual;
    }

    /// <summary>Velocity error = egoVx - virtualVx. Converges to zero when speeds match.</summary>
    public float VelocityError(float egoVx) => egoVx - _virtVx;
}
