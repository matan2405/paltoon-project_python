"""
unified/control/coordinator.py — Unified longitudinal + lateral Nash coordinator.

Manages two independent Nash solvers (10 Hz long, 20 Hz lat) over a single
Vehicle6D ego. Handles APPROACH → MERGE → FOLLOWING phase transitions.

Design:
  - All imports are from unified/ sub-modules only — no dependency on the
    split Longitudinal/ or lateral/ module trees.
  - Safety fields, authority allocators, and reference generators are implemented
    inline using parameters from unified/config.py.
  - MOBIL lane-change decision: unified/control/mobil_lane_change.py.

References:
  Li et al. 2019        — Nash GNE framework, authority allocation
  Pustilnik & Borrelli 2025 — non-normalized GNE, α = 1/(1+λ)
  Rajamani Ch. 6        — CTH platoon controller, transitional reference
  Wang et al. 2015/2016 — elliptic driving safety field
  Swain & Rath 2023     — lateral authority sigmoid (Eq. 15)
  Kesting et al. 2007   — MOBIL lane-change model
"""

import os
import sys
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List

# ── root of this repo (two levels up from unified/control/) ──────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ── unified config ───────────────────────────────────────────────────────────
sys.path.insert(0, _REPO_ROOT)
from unified.config import (                                          # noqa: E402
    SIMULATION_DT,
    LONG_NASH_CONTROL_DT, LAT_NASH_CONTROL_DT,
    LONG_NASH_STEP_INTERVAL, LAT_NASH_STEP_INTERVAL,
    NASH_NP, NASH_NU,
    LONG_NASH_Q_POS, LONG_NASH_Q_VEL,
    LONG_NASH_R1, LONG_NASH_R2, LONG_NASH_S1, LONG_NASH_S2,
    LONG_NASH_U1_MIN, LONG_NASH_U1_MAX, LONG_NASH_U2_MIN, LONG_NASH_U2_MAX,
    LONG_NASH_DU1_MAX, LONG_NASH_DU2_MAX,
    LONG_NASH_V_MIN, LONG_NASH_V_MAX, LONG_NASH_GAP_MIN,
    LONG_NASH_LAMBDA_LEVELS,
    LAT_NASH_Q_Y, LAT_NASH_Q_PSI,
    LAT_NASH_Q_Y_TERMINAL_FAC, LAT_NASH_Q_PSI_TERMINAL_FAC,
    LAT_NASH_R1, LAT_NASH_R2, LAT_NASH_S1, LAT_NASH_S2,
    AUTHORITY_GAP_ERROR_MAX, AUTHORITY_VEL_ERROR_MAX, FOLLOWING_GAP_ERROR_FACTOR,
    GAP_SEARCH_DURATION, LANE_CHANGE_MIN_TIME,
    DSF_SPEED_COEFF, DSF_SPEED_EXPONENT, DSF_SPEED_OFFSET, DSF_VEHICLE_MASS, DSF_EPSILON,
    LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX,
    LAT_NASH_Y_MAX, LAT_NASH_PSI_MAX, LAT_NASH_LAMBDA_LEVELS,
    LONG_AUTHORITY_LAMBDA_MIN, LONG_AUTHORITY_LAMBDA_MAX,
    LONG_AUTHORITY_FORCE_MIDPOINT, LONG_AUTHORITY_K_STEEPNESS,
    LONG_AUTHORITY_ALPHA_BASE, LONG_AUTHORITY_ALPHA_FAST,
    LAT_AUTHORITY_LAMBDA_MIN, LAT_AUTHORITY_LAMBDA_MAX,
    LAT_AUTHORITY_FORCE_MIDPOINT, LAT_AUTHORITY_K_STEEPNESS,
    LAT_AUTHORITY_ALPHA_BASE, LAT_AUTHORITY_ALPHA_FAST,
    LAT_AUTHORITY_ALPHA_FAST_THR, LAT_AUTHORITY_ALPHA_BASE_THR,
    LAT_AUTHORITY_SIGMOID_M1, LAT_AUTHORITY_SIGMOID_M2,
    LONG_SAFETY_MIN_SAFE_DISTANCE, LONG_SAFETY_EMERGENCY_BRAKE_DIST,
    LONG_SAFETY_MAX_REPULSIVE_FORCE, LONG_SAFETY_FILTER_ALPHA,
    LAT_DSF_TS, LAT_DSF_TAU, LAT_DSF_A_MIN, LAT_DSF_G,
    LAT_DSF_K1, LAT_DSF_K2, LAT_DSF_DR,
    LAT_SAFETY_MAX_FORCE, LAT_SAFETY_FILTER_ALPHA,
    PLATOON_TIME_GAP, PLATOON_STANDSTILL_DISTANCE, PLATOON_TARGET_VELOCITY,
    PLATOON_VEHICLE_LENGTH,
    RAJAMANI_H, RAJAMANI_K1, RAJAMANI_K2, RAJAMANI_K3, RAJAMANI_K4, RAJAMANI_K5,
    MOBIL_IDM_V0, MOBIL_IDM_T, MOBIL_IDM_A_MAX, MOBIL_IDM_B,
    MOBIL_IDM_S0, MOBIL_IDM_DELTA, MOBIL_IDM_L,
    MOBIL_P, MOBIL_B_SAFE, MOBIL_A_TH, MOBIL_MIN_GAP,
    APPROACH_MOBIL_CHECK_DISTANCE,
    MERGE_COMPLETE_Y_ERROR, MERGE_COMPLETE_PSI_ERROR, MERGE_COMPLETE_GAP_RATIO,
    PHASE_TRANSITION_HOLD_TIME,
    PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y, LANE_WIDTH,
    DRIVER_PARAMS,
    MAX_ACCELERATION, MAX_DECELERATION,
    NOMINAL_VELOCITY, VEHICLE_LENGTH, VEHICLE_MASS,
    LONG_NASH_REBUILD_DVX, LAT_NASH_REBUILD_DVX,
)


# ── Import Nash solvers and MOBIL from unified module ────────────────────────
from unified.nash_solver.longitudinal_constrained_nash_solver import (
    ConstrainedLongitudinalNashSolver, ConstrainedNashParams as LongNashParams,
)
from unified.nash_solver.lateral_constrained_nash_solver import (
    ConstrainedLateralNashSolver, ConstrainedLateralNashParams as LatNashParams,
)
from unified.control.mobil_lane_change import MOBILLaneChange


# =============================================================================
# Phase enum
# =============================================================================

class MergePhase(Enum):
    APPROACH   = 'APPROACH'    # In adjacent lane; approaching platoon; MOBIL checking
    GAP_SEARCH = 'GAP_SEARCH'  # MOBIL approved; brief gap-search pause before lateral Nash
    MERGE      = 'MERGE'       # Active lane change; both Nash solvers active
    FOLLOWING  = 'FOLLOWING'   # Lane change done; longitudinal Nash only


# =============================================================================
# UnifiedCoordinator
# =============================================================================

class UnifiedCoordinator:
    """
    Coordinates longitudinal + lateral Nash solvers for a Vehicle6D ego.

    Scheduling:
      Longitudinal Nash: every LONG_NASH_STEP_INTERVAL=10 sim steps (10 Hz)
      Lateral Nash:      every LAT_NASH_STEP_INTERVAL=5  sim steps (20 Hz)
                         active only during MERGE phase

    Authority:
      Long: sigmoid on gap-based repulsive force (Wang 2015/2016)
      Lat:  Swain & Rath 2023 Eq. 15 sigmoid on lateral error + DSF force

    Reference trajectories:
      R1 (system): Rajamani CTH for long; 5th-order polynomial to target lane for lat
      R2 (human):  IDM free-road for long; hold-lane or natural lane change for lat
    """

    def __init__(self, ego, driver_type: str = 'normal', merge_scenario: str = 'back'):
        """
        Parameters
        ----------
        ego : Vehicle6D
        driver_type : str
            'cautious' | 'normal' | 'aggressive'
        merge_scenario : str
            'back' | 'middle' | 'front'
        """
        self.ego         = ego
        self.driver_type = driver_type
        self.merge_scenario   = merge_scenario   # 'back' | 'middle' | 'front'
        self.locked_leader    = None  # set at APPROACH→MERGE transition
        self.locked_follower  = None  # set at APPROACH→MERGE transition
        self._merge_locked    = False
        self.phase       = MergePhase.APPROACH

        dp = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self._tlc     = dp['tlc']              # lane-change duration [s]
        self._sys_tlc = self._tlc * dp['system_tlc_multiplier']

        # ── Nash step counters / ZOH cache ────────────────────────────────────
        self._long_counter = LONG_NASH_STEP_INTERVAL  # run immediately first step
        self._lat_counter  = LAT_NASH_STEP_INTERVAL
        self._last_u_long  = 0.0
        self._last_delta_lat = 0.0

        # ── Phase hold timer (for FOLLOWING transition) ───────────────────────
        self._phase_hold_start: Optional[float] = None

        # ── GAP_SEARCH phase timer ────────────────────────────────────────────
        self._gap_search_start: Optional[float] = None

        # ── Authority smoothing state ─────────────────────────────────────────
        self._long_lambda = LONG_AUTHORITY_LAMBDA_MIN
        self._lat_lambda  = LAT_AUTHORITY_LAMBDA_MIN

        # ── Safety force filter state ─────────────────────────────────────────
        self._long_force_filt = 0.0
        self._lat_force_filt  = 0.0

        # ── MOBIL ─────────────────────────────────────────────────────────────
        self.mobil         = MOBILLaneChange()
        self.mobil_approved = False
        self.mobil.set_politeness(driver_type)

        # ── Long Nash solver ──────────────────────────────────────────────────
        long_params = LongNashParams(
            Np=NASH_NP, Nu=NASH_NU, dt=LONG_NASH_CONTROL_DT,
            Q_pos=LONG_NASH_Q_POS,        Q_vel=LONG_NASH_Q_VEL,
            Q_pos_terminal=10.0 * LONG_NASH_Q_POS,
            Q_vel_terminal= 4.0 * LONG_NASH_Q_VEL,
            R1=LONG_NASH_R1, R2=LONG_NASH_R2,
            S1=LONG_NASH_S1, S2=LONG_NASH_S2,
            u1_min=LONG_NASH_U1_MIN, u1_max=LONG_NASH_U1_MAX,
            u2_min=LONG_NASH_U2_MIN, u2_max=LONG_NASH_U2_MAX,
            du1_max=LONG_NASH_DU1_MAX, du2_max=LONG_NASH_DU2_MAX,
            v_min=LONG_NASH_V_MIN,     v_max=LONG_NASH_V_MAX,
            gap_min=LONG_NASH_GAP_MIN,
            lambda_levels=LONG_NASH_LAMBDA_LEVELS,
        )
        self.long_nash = ConstrainedLongitudinalNashSolver(
            vehicle=ego.long_proxy,
            params=long_params,
        )

        # ── Lat Nash solver ───────────────────────────────────────────────────
        lat_params = LatNashParams(
            Np=NASH_NP, Nu=NASH_NU, dt=LAT_NASH_CONTROL_DT,
        )
        # Override weights from unified config
        lat_params.Q_y               = LAT_NASH_Q_Y
        lat_params.Q_psi             = LAT_NASH_Q_PSI
        lat_params.Q_y_terminal      = LAT_NASH_Q_Y_TERMINAL_FAC  * LAT_NASH_Q_Y
        lat_params.Q_psi_terminal    = LAT_NASH_Q_PSI_TERMINAL_FAC * LAT_NASH_Q_PSI
        lat_params.R1                = LAT_NASH_R1
        lat_params.R2                = LAT_NASH_R2
        lat_params.S1                = LAT_NASH_S1
        lat_params.S2                = LAT_NASH_S2
        lat_params.delta_min         = LAT_NASH_DELTA_MIN
        lat_params.delta_max         = LAT_NASH_DELTA_MAX
        lat_params.lambda_levels     = LAT_NASH_LAMBDA_LEVELS

        self.lat_nash = ConstrainedLateralNashSolver(
            vehicle=ego.lat_proxy,
            params=lat_params,
        )
        self._vx_at_last_lat_rebuild: float = self.ego.state.vx

        # ── Logging ───────────────────────────────────────────────────────────
        self.data: dict = {k: [] for k in (
            'u_long', 'delta_lat',
            'long_lambda', 'lat_lambda',
            'long_force', 'long_force_leader', 'long_force_follower',
            'lat_force',
            'ego_rear_gap',       # bumper-to-bumper gap to locked_follower
            'u1_long', 'u2_long', 'u1_lat', 'u2_lat',
            'phase',
        )}

    # =========================================================================
    # Main step
    # =========================================================================

    def step(self,
             sim_time: float,
             dt: float,
             platoon_vehicles: list) -> Tuple[float, float]:
        """Execute one simulation step.

        Returns
        -------
        u_long : float  — desired longitudinal acceleration [m/s²]
        delta_lat : float — desired front steering angle [rad]
        """
        # ── Longitudinal Nash (10 Hz) ──────────────────────────────────────────
        self._long_counter += 1
        if self._long_counter >= LONG_NASH_STEP_INTERVAL:
            self._long_counter = 0
            self._run_long_nash_step(platoon_vehicles)

        # ── Lateral Nash (20 Hz, MERGE + FOLLOWING phases) ────────────────────
        if self.phase in (MergePhase.MERGE, MergePhase.FOLLOWING):
            self._lat_counter += 1
            if self._lat_counter >= LAT_NASH_STEP_INTERVAL:
                self._lat_counter = 0
                self._run_lat_nash_step(platoon_vehicles)
        else:
            self._last_delta_lat = 0.0

        # ── Phase transitions ─────────────────────────────────────────────────
        self._update_phase(sim_time, platoon_vehicles)

        # ── Log ───────────────────────────────────────────────────────────────
        self.data['u_long'].append(self._last_u_long)
        self.data['delta_lat'].append(self._last_delta_lat)
        self.data['phase'].append(self.phase.value)

        return self._last_u_long, self._last_delta_lat

    # =========================================================================
    # Longitudinal Nash step
    # =========================================================================

    def _run_long_nash_step(self, platoon_vehicles: list):
        """Compute one longitudinal Nash solve."""
        self.long_nash.update_linearization(dt=LONG_NASH_CONTROL_DT)
        ego   = self.ego
        x0    = ego.get_long_state()          # [x, vx]

        # Find nearest platoon vehicle ahead (immediate leader for ego)
        leader   = self._find_leader(platoon_vehicles)
        follower = self.locked_follower if self._merge_locked else None

        # Pre-compute raw components for logging (unfiltered)
        f_leader_raw   = self._long_leader_force_raw(leader)   if leader   is not None else 0.0
        f_follower_raw = self._long_follower_force_raw(follower) if follower is not None else 0.0
        rear_gap = (ego.state.x - follower.state.x - VEHICLE_LENGTH
                    ) if follower is not None else float('nan')

        if leader is None:
            # 'front' scenario or free road: no forward obstacle.
            # Mirrors split-module: F_leader=0, F_total=F_follower (line 401).
            force = self._long_safety_force(leader=None, follower=follower)
            # F_follower > 0 when follower is too close → ego accelerates forward
            u_ff = float(np.clip(
                0.3 * (PLATOON_TARGET_VELOCITY - ego.state.vx)
                + force / max(VEHICLE_MASS, 1.0),
                MAX_DECELERATION, MAX_ACCELERATION))
            self._last_u_long = u_ff
            self.data['u1_long'].append(u_ff)
            self.data['u2_long'].append(0.0)
            self.data['long_lambda'].append(self._long_lambda)
            self.data['long_force'].append(force)
            self.data['long_force_leader'].append(0.0)
            self.data['long_force_follower'].append(f_follower_raw)
            self.data['ego_rear_gap'].append(rear_gap)
            return

        # Safety field force: F_total = F_leader + F_follower (split-module line 401)
        force = self._long_safety_force(leader, follower=follower)

        # Compute gap_error and velocity_error for authority computation
        _gap_err = self._long_gap_error(leader)   # positive = too close
        _vel_err = self.ego.state.vx - float(
            getattr(leader, 'v', getattr(leader.state, 'vx', PLATOON_TARGET_VELOCITY)))

        # Authority (safety + gap/velocity Swain & Rath)
        lam = self._long_authority(force, gap_error=_gap_err, vel_error=_vel_err)

        # Reference trajectories
        R1 = self._long_sys_ref(leader, NASH_NP, LONG_NASH_CONTROL_DT)
        R2 = self._long_hum_ref(leader, NASH_NP, LONG_NASH_CONTROL_DT)

        # Nash solve
        try:
            u1, u2 = self.long_nash.solve_nash_equilibrium(
                x0=x0, R1_ref=R1, R2_ref=R2,
                lambda_k=lam, field_force=force,
            )
            u_shared = u1 + u2
        except Exception as exc:
            print(f"[LongNash] solve failed: {exc}")
            u_shared = np.clip(
                RAJAMANI_K1 * self._long_gap_error(leader), MAX_DECELERATION, MAX_ACCELERATION)
            u1 = u_shared; u2 = 0.0

        self._last_u_long = float(np.clip(u_shared, MAX_DECELERATION, MAX_ACCELERATION))
        self.data['u1_long'].append(float(u1))
        self.data['u2_long'].append(float(u2))
        self.data['long_lambda'].append(lam)
        self.data['long_force'].append(force)
        self.data['long_force_leader'].append(f_leader_raw)
        self.data['long_force_follower'].append(f_follower_raw)
        self.data['ego_rear_gap'].append(rear_gap)

    # =========================================================================
    # Lateral Nash step
    # =========================================================================

    def _rebuild_lat_nash_once(self):
        """Full rebuild of lateral Nash prediction matrices at the current operating point.

        Called ONCE when entering MERGE phase so that H, U, HQ1H, and the CVXPY
        P1/P2 matrices all reflect the vehicle's actual cruise speed.

        Must NOT be called every Nash step — calling _build_state_space /
        _build_prediction_matrices alone without also rebuilding the CVXPY problems
        (which bake in HQ1H) creates an H/HQ1H inconsistency that destabilises
        the IBR solver.
        """
        print(f"[LatNash] Rebuilding prediction matrices at vx={self.ego.state.vx:.1f} m/s")
        self.lat_nash._build_state_space()
        self.lat_nash._build_prediction_matrices()
        self.lat_nash._build_base_cost_matrices()
        self.lat_nash._build_dmpc_problems()
        self._vx_at_last_lat_rebuild = self.ego.state.vx

    def _run_lat_nash_step(self, platoon_vehicles: list):
        """Compute one lateral Nash solve."""
        # Rebuild lateral Nash matrices if vx has drifted beyond physics threshold
        if abs(self.ego.state.vx - self._vx_at_last_lat_rebuild) > LAT_NASH_REBUILD_DVX:
            self._rebuild_lat_nash_once()

        ego     = self.ego
        x0      = ego.get_lat_state_vector()   # [y, vy, psi, psi_dot]
        target_y = PLATOON_LANE_Y

        # Safety force (DSF)
        force = self._lat_safety_force(platoon_vehicles)

        # Authority
        y_err = ego.state.y - target_y
        lam   = self._lat_authority(force, y_err)

        # Reference trajectories
        R1 = self._lat_sys_ref(target_y, NASH_NP, LAT_NASH_CONTROL_DT)
        R2 = self._lat_hum_ref(NASH_NP, LAT_NASH_CONTROL_DT)

        # Nash solve
        try:
            u1, u2 = self.lat_nash.solve_nash_equilibrium(
                x0=x0, R1_ref=R1, R2_ref=R2,
                lambda_k=lam, field_force=force,
            )
            delta = u1 + u2
        except Exception as exc:
            print(f"[LatNash] solve failed: {exc}")
            delta = float(np.clip(-0.1 * y_err, LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX))
            u1 = delta; u2 = 0.0

        self._last_delta_lat = float(
            np.clip(delta, LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX))
        self.data['u1_lat'].append(float(u1))
        self.data['u2_lat'].append(float(u2))
        self.data['lat_lambda'].append(lam)
        self.data['lat_force'].append(force)

    # =========================================================================
    # Phase management
    # =========================================================================

    def _update_phase(self, sim_time: float, platoon_vehicles: list):
        """Update phase state machine."""
        ego = self.ego

        if self.phase == MergePhase.APPROACH:
            # Check if close enough to run MOBIL
            nearest_dist = self._nearest_platoon_dist(platoon_vehicles)
            if nearest_dist < APPROACH_MOBIL_CHECK_DISTANCE:
                if not self.mobil_approved:
                    self.mobil_approved = self._check_mobil(platoon_vehicles)
                if self.mobil_approved:
                    self._lock_merge_position(platoon_vehicles)
                    self.phase = MergePhase.GAP_SEARCH
                    self._gap_search_start = sim_time
                    print(f"[Coordinator] Phase: APPROACH -> GAP_SEARCH at t={sim_time:.1f}s")

        elif self.phase == MergePhase.GAP_SEARCH:
            # Wait GAP_SEARCH_DURATION seconds then transition to MERGE
            if (self._gap_search_start is not None and
                    sim_time - self._gap_search_start >= GAP_SEARCH_DURATION):
                self.phase = MergePhase.MERGE
                # Rebuild lateral Nash matrices once at cruise speed before first MERGE solve
                self._rebuild_lat_nash_once()
                print(f"[Coordinator] Phase: GAP_SEARCH -> MERGE at t={sim_time:.1f}s")

        elif self.phase == MergePhase.MERGE:
            # Check merge completion conditions
            target_y    = PLATOON_LANE_Y
            y_err       = abs(ego.state.y - target_y)
            psi_err     = abs(ego.state.psi)
            leader      = self._find_leader(platoon_vehicles)
            if leader is not None:
                gap       = leader.state.x - ego.state.x - VEHICLE_LENGTH
                des_gap   = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx
                gap_ratio = abs(gap - des_gap) / max(des_gap, 1e-3)
            else:
                # 'front' scenario: ego is platoon head, no leader exists.
                # Gap condition is not applicable — only y and psi matter.
                gap_ratio = 0.0

            conditions_met = (y_err    < MERGE_COMPLETE_Y_ERROR   and
                              psi_err  < MERGE_COMPLETE_PSI_ERROR  and
                              gap_ratio < MERGE_COMPLETE_GAP_RATIO)

            if conditions_met:
                if self._phase_hold_start is None:
                    self._phase_hold_start = sim_time
                elif sim_time - self._phase_hold_start >= PHASE_TRANSITION_HOLD_TIME:
                    self.phase = MergePhase.FOLLOWING
                    self._last_delta_lat = 0.0
                    print(f"[Coordinator] Phase: MERGE -> FOLLOWING at t={sim_time:.1f}s")
            else:
                self._phase_hold_start = None   # reset hold timer on regression

    # =========================================================================
    # MOBIL check
    # =========================================================================

    def _check_mobil(self, platoon_vehicles: list) -> bool:
        """Return True when MOBIL approves the lane change."""
        ego = self.ego
        pv = [{'x': v.state.x, 'v': getattr(v, 'v', getattr(v.state, 'vx', PLATOON_TARGET_VELOCITY))}
              for v in platoon_vehicles]
        if not pv:
            return True

        sorted_pv = sorted(pv, key=lambda d: d['x'], reverse=True)
        hx = ego.state.x
        if   hx > sorted_pv[0]['x']:  pos = 'before'
        elif hx < sorted_pv[-1]['x']: pos = 'after'
        else:                          pos = 'middle'

        approved, _ = self.mobil.check_platoon_merge(
            human_x=hx,
            human_v=ego.state.vx,
            platoon_vehicles=pv,
            merge_position=pos,
        )
        if approved:
            print(f"[Coordinator] MOBIL approved lane change (pos={pos})")
        return approved

    # =========================================================================
    # Merge position locking
    # =========================================================================

    def _lock_merge_position(self, platoon_vehicles: list):
        """Lock leader and follower based on actual ego x position at lock time.

        merge_scenario sets the initial ego placement (simulator.py), but by the
        time MOBIL fires the ego may have drifted (aggressive driver can overtake
        the platoon tail during the 30 s approach).  Position-based locking is
        always correct regardless of scenario or drift.
        """
        ego = self.ego
        sorted_pv = sorted(platoon_vehicles, key=lambda v: v.state.x, reverse=True)
        if not sorted_pv:
            return

        ego_x = ego.state.x

        if ego_x > sorted_pv[0].state.x:
            # Ego is ahead of the entire platoon → insert at front
            self.locked_leader   = None
            self.locked_follower = sorted_pv[0]
        elif ego_x <= sorted_pv[-1].state.x:
            # Ego is at or behind the platoon tail → insert at back
            self.locked_leader   = sorted_pv[-1]
            self.locked_follower = None
        else:
            # Ego is within the platoon spacing → find the surrounding gap
            self.locked_leader   = None
            self.locked_follower = None
            for i, pv in enumerate(sorted_pv):
                if pv.state.x <= ego_x:
                    self.locked_leader   = sorted_pv[i - 1]  # i > 0 guaranteed here
                    self.locked_follower = pv
                    break

        self._merge_locked = True
        lname = self.locked_leader.vehicle_id   if self.locked_leader   else 'None'
        fname = self.locked_follower.vehicle_id if self.locked_follower else 'None'
        print(f"[Coordinator] Merge locked — leader={lname}, follower={fname}, "
              f"scenario={self.merge_scenario}")

    # =========================================================================
    # Longitudinal safety field  (Wang et al. 2015/2016, simplified)
    # =========================================================================

    def _long_gap_error(self, leader) -> float:
        """Signed gap error: positive = too close."""
        ego = self.ego
        gap     = leader.state.x - ego.state.x - VEHICLE_LENGTH
        des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx
        return des_gap - gap

    def _long_leader_force_raw(self, leader) -> float:
        """Raw repulsive/attractive force from the leader [N].

        Force > 0 ↔ leader too close (repulsive); Force < 0 ↔ gap too large (attractive).
        """
        ego = self.ego
        gap     = leader.state.x - ego.state.x - VEHICLE_LENGTH
        des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx

        if gap < LONG_SAFETY_EMERGENCY_BRAKE_DIST:
            return LONG_SAFETY_MAX_REPULSIVE_FORCE
        elif gap < LONG_SAFETY_MIN_SAFE_DISTANCE:
            t = 1.0 - (gap - LONG_SAFETY_EMERGENCY_BRAKE_DIST) / max(
                LONG_SAFETY_MIN_SAFE_DISTANCE - LONG_SAFETY_EMERGENCY_BRAKE_DIST, 1e-3)
            return t * LONG_SAFETY_MAX_REPULSIVE_FORCE
        else:
            gap_error = des_gap - gap
            return float(np.clip(
                gap_error / max(des_gap, 1.0) * LONG_SAFETY_MAX_REPULSIVE_FORCE,
                -0.25 * LONG_SAFETY_MAX_REPULSIVE_FORCE,
                LONG_SAFETY_MAX_REPULSIVE_FORCE,
            ))

    def _long_follower_force_raw(self, follower) -> float:
        """Raw repulsive force from the follower [N].

        Force >= 0: follower too close → ego must accelerate (push-forward risk).
        No attractive component — we do not penalise the ego for being too far ahead.
        Mirrors the split-module _compute_force_to_vehicle(is_leader=False) logic
        (EllipseLongitudinalSafetyField, longitudinal_safety_field.py line 395–399).
        """
        ego = self.ego
        gap_rear = ego.state.x - follower.state.x - VEHICLE_LENGTH
        des_gap  = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx

        if gap_rear < LONG_SAFETY_EMERGENCY_BRAKE_DIST:
            return LONG_SAFETY_MAX_REPULSIVE_FORCE
        elif gap_rear < LONG_SAFETY_MIN_SAFE_DISTANCE:
            t = 1.0 - (gap_rear - LONG_SAFETY_EMERGENCY_BRAKE_DIST) / max(
                LONG_SAFETY_MIN_SAFE_DISTANCE - LONG_SAFETY_EMERGENCY_BRAKE_DIST, 1e-3)
            return t * LONG_SAFETY_MAX_REPULSIVE_FORCE
        else:
            gap_error = des_gap - gap_rear   # positive when follower too close
            return float(max(0.0, gap_error / max(des_gap, 1.0) * LONG_SAFETY_MAX_REPULSIVE_FORCE))

    def _long_safety_force(self, leader, follower=None) -> float:
        """Compute total longitudinal safety field force [N].

        Matches the split-module EllipseLongitudinalSafetyField architecture
        (longitudinal_safety_field.py line 368–402):
          F_total = F_leader + F_follower

        F_leader > 0 ↔ leader too close (brake); < 0 ↔ gap too large (attract).
        F_follower ≥ 0 ↔ follower too close (accelerate).
        Combined force is EMA low-pass filtered to reduce noise.
        """
        F_leader   = self._long_leader_force_raw(leader)   if leader   is not None else 0.0
        F_follower = self._long_follower_force_raw(follower) if follower is not None else 0.0
        raw = F_leader + F_follower

        self._long_force_filt = (LONG_SAFETY_FILTER_ALPHA * raw
                                 + (1 - LONG_SAFETY_FILTER_ALPHA) * self._long_force_filt)
        return self._long_force_filt

    # =========================================================================
    # Longitudinal authority allocator  (sigmoid on risk force)
    # =========================================================================

    def _long_authority(self, risk_force: float,
                        gap_error: float = 0.0, vel_error: float = 0.0) -> float:
        """Map longitudinal risk force + gap/velocity errors → authority ratio λ.

        Implements the LongitudinalAuthorityAllocator logic (Swain & Rath 2023, Eq. 15):
          1. Safety authority:   sigmoid on |risk_force|
          2. Gap+velocity:       Swain & Rath sigmoid using min(l_n_gap, l_n_vel)
          3. Fusion:             max(lambda_safety, lambda_gap)
          4. Adaptive EMA smoothing
        """
        # --- 1. SAFETY Authority (sigmoid on risk force) ---
        exponent = -LONG_AUTHORITY_K_STEEPNESS * (abs(risk_force) - LONG_AUTHORITY_FORCE_MIDPOINT)
        lam_safety = (LONG_AUTHORITY_LAMBDA_MAX - LONG_AUTHORITY_LAMBDA_MIN) / (1.0 + np.exp(exponent))
        lam_safety += LONG_AUTHORITY_LAMBDA_MIN

        # --- 2. GAP+VELOCITY Authority (Swain & Rath 2023, Eq. 15) ---
        abs_gap_err = abs(gap_error)
        abs_vel_err = abs(vel_error)
        l_n_gap = (AUTHORITY_GAP_ERROR_MAX - abs_gap_err) / AUTHORITY_GAP_ERROR_MAX
        l_n_vel = (AUTHORITY_VEL_ERROR_MAX - abs_vel_err) / AUTHORITY_VEL_ERROR_MAX
        l_n     = float(np.clip(min(l_n_gap, l_n_vel), 0.0, 1.0))

        # Use the lateral authority sigmoid parameters (Swain & Rath Eq. 15)
        # Re-use LAT_AUTHORITY_SIGMOID_M1/M2 as they carry the same meaning
        gamma1  = 1.0 / (1.0 + np.exp(LAT_AUTHORITY_SIGMOID_M1 * (-l_n + LAT_AUTHORITY_SIGMOID_M2)))
        gamma2  = 1.0 - gamma1
        lam_gap = gamma2 / max(gamma1, 1e-6)

        # --- 3. Fusion: max urgency ---
        lam_raw = min(max(lam_safety, lam_gap), LONG_AUTHORITY_LAMBDA_MAX)

        # --- 4. Adaptive EMA ---
        combined_error = max(abs_gap_err, abs_vel_err)
        if combined_error > 5.0:
            alpha = LONG_AUTHORITY_ALPHA_FAST
        elif combined_error > 2.0:
            alpha = LONG_AUTHORITY_ALPHA_BASE + (LONG_AUTHORITY_ALPHA_FAST - LONG_AUTHORITY_ALPHA_BASE) * (
                combined_error - 2.0) / 3.0
        else:
            alpha = LONG_AUTHORITY_ALPHA_BASE
        self._long_lambda = alpha * lam_raw + (1.0 - alpha) * self._long_lambda
        return self._long_lambda

    # =========================================================================
    # Lateral safety field  (DSF based on Wang 2015/2016 / Li 2019)
    # =========================================================================

    def _lat_safety_force(self, platoon_vehicles: list) -> float:
        """Compute lateral DSF risk force [N] using full Wang et al. 2015/2016 formula.

        Implements the complete Driving Safety Field:
          - Virtual mass: M = m * (speed_coeff * v^speed_exp + speed_offset)   (Wang 2016, Eq. 17)
          - Elliptic distance (Li 2019, Eq. 12-13): r* = sqrt((dx/a)^2+(dy/b)^2)
          - Field strength: E = G * M_obs / r*                                  (Eq. 11)
          - Kinetic correction: E *= exp(k1 * v_obs * cos(theta_v))             (Eq. 18)
          - Field force: Fr = E * M_ego * exp(-k2 * v_ego * cos(theta_i)) * (1+DR) (Eq. 21)
          - Direction smoothing: tanh(dy / sigma)
          - Road boundary forces
        """
        ego = self.ego
        total_force = 0.0
        ego_x = ego.state.x
        ego_y = ego.state.y
        ego_vx = max(ego.state.vx, 1.0)

        # Virtual mass of ego (Wang 2016, Eq. 17)
        M_ego = DSF_VEHICLE_MASS * (DSF_SPEED_COEFF * abs(ego_vx) ** DSF_SPEED_EXPONENT
                                    + DSF_SPEED_OFFSET)
        # Ego velocity direction (unit vector along x-axis)
        v_ego_dir = np.array([1.0, 0.0])

        for pv in platoon_vehicles:
            obs_x  = pv.state.x
            obs_y  = getattr(pv.state, 'y', 0.0)
            obs_vx = float(getattr(pv, 'v', getattr(pv.state, 'vx', PLATOON_TARGET_VELOCITY)))

            dx = ego_x - obs_x   # positive = ego ahead
            dy = ego_y - obs_y   # positive = ego to the right

            # --- Elliptic distance (Li 2019, Eq. 12-13) ---
            a = max(abs(ego_vx - obs_vx) * LAT_DSF_TS, LAT_DSF_A_MIN)
            b = LAT_DSF_TAU
            r_star = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
            r_star = max(r_star, DSF_EPSILON)

            # Unit vector along elliptic gradient direction
            r_star_unit = np.array([dx / a, dy / b]) / r_star

            # --- Virtual mass of obstacle (Wang 2016, Eq. 17) ---
            M_obs = DSF_VEHICLE_MASS * (DSF_SPEED_COEFF * abs(obs_vx) ** DSF_SPEED_EXPONENT
                                        + DSF_SPEED_OFFSET)

            # --- Field strength (Li 2019, Eq. 11, R=1) ---
            E = LAT_DSF_G * M_obs / r_star

            # --- Kinetic correction (Eq. 18): obstacle velocity projection ---
            v_obs_dir = np.array([1.0, 0.0]) if abs(obs_vx) > 0.1 else np.array([0.0, 0.0])
            cos_theta_v = float(np.dot(v_obs_dir, r_star_unit))
            E *= np.exp(LAT_DSF_K1 * abs(obs_vx) * cos_theta_v)

            # --- Field force (Eq. 21): ego velocity projection on Cartesian gradient ---
            field_cart = np.array([dx / (a * a), dy / (b * b)])
            field_cart_norm = np.linalg.norm(field_cart)
            if field_cart_norm > 1e-9:
                field_cart_unit = field_cart / field_cart_norm
            else:
                field_cart_unit = np.array([0.0, 1.0])
            cos_theta_i = float(np.dot(field_cart_unit, v_ego_dir))

            Fr = E * M_ego * np.exp(-LAT_DSF_K2 * abs(ego_vx) * cos_theta_i) * (1.0 + LAT_DSF_DR)

            # Lateral direction: smooth tanh repulsion (push ego away from obstacle)
            _sigma = 0.5   # direction smoothing width [m]
            lateral_direction = np.tanh(dy / _sigma)
            total_force += Fr * lateral_direction

        # Road boundary forces
        road_half_width = 7.0    # half road width [m]
        boundary_gain   = 150.0  # [N]
        boundary_scale  = 2.0
        boundary_prox   = 3.0    # activate within this distance [m]
        _eps = 0.1

        dist_right = road_half_width - ego_y
        if 0 < dist_right < boundary_prox:
            potential = 1.0 / max(dist_right, _eps)
            total_force -= boundary_gain * np.tanh(potential / boundary_scale)

        dist_left = road_half_width + ego_y
        if 0 < dist_left < boundary_prox:
            potential = 1.0 / max(dist_left, _eps)
            total_force += boundary_gain * np.tanh(potential / boundary_scale)

        total_force = float(np.clip(total_force, -LAT_SAFETY_MAX_FORCE, LAT_SAFETY_MAX_FORCE))
        self._lat_force_filt = (LAT_SAFETY_FILTER_ALPHA * total_force
                                + (1.0 - LAT_SAFETY_FILTER_ALPHA) * self._lat_force_filt)
        return self._lat_force_filt

    # =========================================================================
    # Lateral authority allocator  (Swain & Rath 2023, Eq. 15)
    # =========================================================================

    def _lat_authority(self, force: float, y_error: float) -> float:
        """Lateral λ using Swain & Rath Eq. 15 with lane-width normalization (Eq. 15).

        Implements:
            l_o_max = LANE_WIDTH / 2
            l_n     = (l_o_max - |y_error|) / l_o_max   # normalized lateral offset
            gamma1  = 1 / (1 + exp(M1 * (-l_n + M2)))
            gamma2  = 1 - gamma1
            lambda_p = gamma2 / gamma1
        Combined with safety sigmoid on |force|.
        """
        # --- 1. SAFETY Authority (sigmoid on risk force) ---
        exp_s = -LAT_AUTHORITY_K_STEEPNESS * (abs(force) - LAT_AUTHORITY_FORCE_MIDPOINT)
        lam_s = (LAT_AUTHORITY_LAMBDA_MAX - LAT_AUTHORITY_LAMBDA_MIN) / (1.0 + np.exp(exp_s))
        lam_s += LAT_AUTHORITY_LAMBDA_MIN

        # --- 2. LATERAL-OFFSET Authority (Swain & Rath 2023, Eq. 15) ---
        # l_n: normalized distance index (1 = lane centre, 0 = lane boundary)
        l_o_max = LANE_WIDTH / 2.0   # = 1.75 m for LANE_WIDTH=3.5 m
        l_n     = (l_o_max - abs(y_error)) / l_o_max
        l_n     = float(np.clip(l_n, 0.0, 1.0))

        m1, m2  = LAT_AUTHORITY_SIGMOID_M1, LAT_AUTHORITY_SIGMOID_M2
        gamma1  = 1.0 / (1.0 + np.exp(m1 * (-l_n + m2)))
        gamma2  = 1.0 - gamma1
        lam_p   = gamma2 / max(gamma1, 1e-6)
        lam_p   = float(np.clip(lam_p, LAT_AUTHORITY_LAMBDA_MIN, LAT_AUTHORITY_LAMBDA_MAX))

        lam_raw = max(lam_s, lam_p)

        # --- 3. Adaptive EMA ---
        e_abs = abs(y_error)
        if   e_abs > LAT_AUTHORITY_ALPHA_FAST_THR:  alpha = LAT_AUTHORITY_ALPHA_FAST
        elif e_abs < LAT_AUTHORITY_ALPHA_BASE_THR:  alpha = LAT_AUTHORITY_ALPHA_BASE
        else:
            blend = (e_abs - LAT_AUTHORITY_ALPHA_BASE_THR) / (
                LAT_AUTHORITY_ALPHA_FAST_THR - LAT_AUTHORITY_ALPHA_BASE_THR)
            alpha = LAT_AUTHORITY_ALPHA_BASE + (LAT_AUTHORITY_ALPHA_FAST - LAT_AUTHORITY_ALPHA_BASE) * blend
        self._lat_lambda = alpha * lam_raw + (1.0 - alpha) * self._lat_lambda
        return self._lat_lambda

    # =========================================================================
    # Longitudinal reference trajectories
    # =========================================================================

    def _long_sys_ref(self, leader, Np: int, dt: float) -> np.ndarray:
        """R1: Rajamani Chapter 6.7 Transitional Controller propagated Np steps.

        Implements the CONTINUOUS parabola-based velocity tracking (Rajamani Sec. 6.7.2):
          1. COLLISION_AVOIDANCE: TTC < 1.2s or gap < MIN_CRITICAL_GAP → emergency decel
          2. CRUISE: No leader or far away → free-road acceleration toward platoon speed
          3. TRANSITIONAL (default): Parabola equation defines target velocity:
             v_target = v_leader + sign(gap_error) * sqrt(2 * a_comfort * |gap_error|)
             a_ref = K_v * (v_target - v_ego)
             Blended with CTG: a_ctg = K1 * gap_error + K2 * R_dot  (for |gap_error| < 5m)
          4. Feedforward: +0.5 * a_leader if available

        Returns (Np*2,) flattened [x0,vx0, x1,vx1, ...].
        """
        # Rajamani transitional parameters (mirrors SystemReferenceGenerator)
        _a_comfort   = 1.5    # comfortable deceleration [m/s²]
        _K_v         = 0.4    # velocity tracking gain [1/s]
        _K1_ctg      = 0.05   # CTG position gain
        _K2_ctg      = 0.4    # CTG velocity gain
        _h           = RAJAMANI_H   # time headway [s]
        _d0          = PLATOON_STANDSTILL_DISTANCE   # standstill distance [m]
        _detection   = 300.0  # detection range [m]
        _catchup     = 1.05   # cruise speed multiplier when far behind
        _min_crit_gap = 2.0 + VEHICLE_LENGTH * 0.5
        _ttc_thresh   = 1.2
        _max_emerg    = -5.0
        _free_delta   = 4.0   # IDM exponent for free-road acc

        x  = float(self.ego.state.x)
        vx = float(self.ego.state.vx)
        lx = float(leader.state.x)
        lv = float(getattr(leader, 'v', getattr(leader.state, 'vx', PLATOON_TARGET_VELOCITY)))
        la = float(getattr(leader.state, 'ax', 0.0) if hasattr(leader, 'state') else 0.0)

        desired_gap = _d0 + _h * vx   # initial desired gap

        ref = np.empty(Np * 2)
        for k in range(Np):
            L = VEHICLE_LENGTH
            R = lx - x - L                   # bumper-to-bumper gap [m]
            R_desired = _d0 + _h * vx       # Rajamani CTG policy (matches standalone: L + h*v)
            gap_error = R - R_desired        # positive = too far behind
            R_dot = lv - vx                  # range rate (positive = gap increasing)
            dist_to_leader = lx - x

            # Time-to-collision
            ttc = R / abs(R_dot) if R_dot < 0.0 else float('inf')

            # --- Controller mode ---
            if R < _min_crit_gap or ttc < _ttc_thresh:
                # Collision avoidance
                a_ref = _max_emerg
            elif dist_to_leader > _detection:
                # Cruise — accelerate toward platoon target speed
                target_v = PLATOON_TARGET_VELOCITY * _catchup
                v_ratio  = max(vx / max(target_v, 1.0), 0.0)
                a_ref    = MAX_ACCELERATION * (1.0 - v_ratio ** _free_delta)
            else:
                # TRANSITIONAL: Parabola velocity tracking (Rajamani Sec. 6.7.2)
                if gap_error >= 0:
                    v_target = lv + np.sqrt(2.0 * _a_comfort * gap_error)
                else:
                    v_target = lv - np.sqrt(2.0 * _a_comfort * abs(gap_error))

                v_err = v_target - vx
                a_ref = _K_v * v_err

                # CTG feedback (Rajamani Eq. 6.15) — blend near equilibrium
                a_ctg     = _K1_ctg * gap_error + _K2_ctg * R_dot
                gap_mag   = abs(gap_error)
                if gap_mag < 5.0:
                    blend = (1.0 - gap_mag / 5.0) ** 2
                    a_ref = (1.0 - blend) * a_ref + blend * a_ctg

                # Feedforward from leader acceleration
                a_ref += 0.5 * la

            a_ref = float(np.clip(a_ref, MAX_DECELERATION, MAX_ACCELERATION))

            # Advance leader (constant velocity)
            lx = lx + lv * dt
            # Propagate ego (double integrator)
            vx = float(np.clip(vx + a_ref * dt, 0.0, LONG_NASH_V_MAX))
            x  = x + vx * dt

            ref[2 * k]     = x
            ref[2 * k + 1] = vx
        return ref

    def _long_hum_ref(self, leader, Np: int, dt: float) -> np.ndarray:
        """R2: Human — IDM free-road + interaction prediction (mirrors HumanDriver).

        Free-road: a_free = a_max * (1 - (v/v0)^delta)
        With leader: full IDM using gap, relative velocity, plan_time_headway, plan_decel
        Returns (Np*2,) flattened [x0,vx0, x1,vx1, ...].
        """
        dp = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])

        # IDM parameters (from driver profile / longitudinal config defaults)
        a_max      = MOBIL_IDM_A_MAX         # [m/s²]
        v0         = PLATOON_TARGET_VELOCITY + dp.get('velocity_offset', 0.0)  # desired speed
        delta_idm  = MOBIL_IDM_DELTA         # acceleration exponent
        plan_T     = 2.0                     # planning time headway (more tolerant) [s]
        plan_b     = 4.0                     # planning deceleration [m/s²]
        s0         = MOBIL_IDM_S0            # minimum spacing [m]
        a_min      = MAX_DECELERATION        # maximum deceleration [m/s²]

        x  = float(self.ego.state.x)
        vx = float(self.ego.state.vx)

        # Simulate leader at constant velocity (planning assumption)
        lx = float(leader.state.x) if leader is not None else None
        lv = float(getattr(leader, 'v', getattr(leader.state, 'vx', PLATOON_TARGET_VELOCITY))) if leader is not None else 0.0

        ref = np.empty(Np * 2)
        for k in range(Np):
            # Free-road term
            v_ratio = vx / max(v0, 1.0)
            free_road_term = 1.0 - v_ratio ** delta_idm

            # Interaction term (IDM, only when leader exists)
            interaction_term = 0.0
            if lx is not None:
                s = lx - x - VEHICLE_LENGTH
                delta_v = vx - lv
                s_star = s0 + vx * plan_T + (vx * delta_v) / (2.0 * np.sqrt(max(a_max * plan_b, 1e-6)))
                s_safe = max(s, 0.1)
                interaction_term = -(s_star / s_safe) ** 2

            a_human = float(np.clip(a_max * (free_road_term + interaction_term), a_min, a_max))

            # Advance leader (constant velocity)
            if lx is not None:
                lx = lx + lv * dt
            # Propagate ego
            vx = float(np.clip(vx + a_human * dt, 0.0, LONG_NASH_V_MAX))
            x  = x + vx * dt

            ref[2 * k]     = x
            ref[2 * k + 1] = vx
        return ref

    # =========================================================================
    # Lateral reference trajectories
    # =========================================================================

    def _lat_sys_ref(self, target_y: float, Np: int, dt: float) -> np.ndarray:
        """R1 (system): 5th-order polynomial lane-change to target_y.

        Returns (Np*2,) flattened [y0,psi0, y1,psi1, ...].
        """
        y0   = float(self.ego.state.y)
        dy   = target_y - y0
        horizon = Np * dt                                   # prediction horizon [s]

        # Heading-constrained minimum TLC (mirrors lateral/nash_solver/system_reference_generator.py
        # DynamicTrajectoryParams.compute_min_T_lc):
        #   T_min = 1.875 × |Δy| / (vx × tan(max_heading))
        # This ensures the quintic trajectory never exceeds the driver's heading limit.
        dp = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])
        max_heading_rad = np.radians(dp.get('max_heading_deg', 5.0))
        vx = max(float(self.ego.state.vx), 1.0)
        if abs(dy) > 0.1 and vx > 1.0:
            T_min_heading = 1.875 * abs(dy) / (vx * np.tan(max_heading_rad))
        else:
            T_min_heading = horizon

        # Also clip to [horizon, 2×horizon] so Nash sees meaningful movement
        # within the prediction window (avoids near-zero steering incentive).
        T = float(max(T_min_heading, np.clip(self._sys_tlc, horizon, 2.0 * horizon)))

        # 5th-order polynomial coefficients (boundary: y(0)=y0, y(T)=yf, ẏ=0, ÿ=0 at both ends)
        # y(t) = y0 + dy * (10τ³ - 15τ⁴ + 6τ⁵),  τ = t/T
        ref = np.empty(Np * 2)
        for k in range(Np):
            t   = (k + 1) * dt
            tau = min(t / T, 1.0)
            y   = y0 + dy * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            # ẏ = dy/T * (30τ² - 60τ³ + 30τ⁴)
            psi = (dy / max(T, 1e-6)) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
            psi = float(np.clip(psi / max(self.ego.state.vx, 0.1), -0.3, 0.3))
            ref[2 * k]     = y
            ref[2 * k + 1] = psi
        return ref

    def _lat_hum_ref(self, Np: int, dt: float) -> np.ndarray:
        """R2 (human): quintic polynomial lane-change / lane-keeping prediction.

        Mirrors lateral/nash_solver/human_reference_generator.py:
          - LANE_KEEPING / APPROACH: hold current y, exponential psi decay
          - MERGE phase: cubic polynomial from current (y, vy, ay=0) to target y over T_lc

        Returns (Np*2,) flattened [y0,psi0, y1,psi1, ...].
        """
        y0    = float(self.ego.state.y)
        psi0  = float(self.ego.state.psi)
        vy0   = float(self.ego.state.vy)
        vx    = max(float(self.ego.state.vx), 1.0)
        T_lc  = max(self._tlc, 1.0)   # human lane-change duration
        target_y = PLATOON_LANE_Y

        ref = np.empty(Np * 2)

        if self.phase == MergePhase.FOLLOWING:
            # FOLLOWING: human wants to stay at platoon lane centre — both players agree on target_y
            for k in range(Np):
                t = (k + 1) * dt
                ref[2 * k]     = target_y
                ref[2 * k + 1] = psi0 * np.exp(-t / max(T_lc, 1e-3))
        elif self.phase != MergePhase.MERGE:
            # APPROACH / GAP_SEARCH: human holds current lane, psi decays
            for k in range(Np):
                t = (k + 1) * dt
                ref[2 * k]     = y0
                ref[2 * k + 1] = psi0 * np.exp(-t / max(T_lc, 1e-3))
        else:
            # LANE_CHANGE mode: cubic polynomial trajectory to target_y
            # Boundary conditions: y(0)=y0, y(T)=y_target, ẏ(0)=vy0, ẏ(T)=0
            delta_y = target_y - y0

            # Clip initial velocity to prevent polynomial overshoot
            if abs(delta_y) > 1e-6:
                y_dot_max = 1.5 * abs(delta_y) / T_lc
                y_dot_0   = float(np.clip(vy0, -y_dot_max, y_dot_max))
            else:
                y_dot_0 = 0.0

            # Cubic polynomial coefficients (tau = t / T_lc, tau in [0, 1])
            a0 = y0
            a1 = y_dot_0 * T_lc
            a2 = 3.0 * delta_y - 2.0 * y_dot_0 * T_lc
            a3 = -2.0 * delta_y + y_dot_0 * T_lc

            for k in range(Np):
                t   = (k + 1) * dt
                tau = min(t / T_lc, 1.0)
                if tau < 1.0:
                    y_ref     = a0 + a1 * tau + a2 * tau ** 2 + a3 * tau ** 3
                    y_dot_ref = (a1 + 2.0 * a2 * tau + 3.0 * a3 * tau ** 2) / T_lc
                    psi_ref   = y_dot_ref / vx
                else:
                    y_ref   = target_y
                    psi_ref = 0.0
                ref[2 * k]     = y_ref
                ref[2 * k + 1] = psi_ref
        return ref

    # =========================================================================
    # Utility
    # =========================================================================

    def _find_leader(self, platoon_vehicles: list):
        """Locked leader after merge trigger; None (free-road) before trigger.

        Before _merge_locked the ego drives free-road with no platoon awareness.
        After the trigger locked_leader is used (may be None for the front scenario).
        The follower is accessed separately via self.locked_follower.
        """
        if not self._merge_locked:
            return None
        return self.locked_leader

    def _nearest_platoon_dist(self, platoon_vehicles: list) -> float:
        """Distance from ego to nearest platoon vehicle [m]."""
        if not platoon_vehicles:
            return float('inf')
        ego = self.ego
        return min(abs(pv.state.x - ego.state.x) for pv in platoon_vehicles)

    def reset(self):
        """Reset coordinator state for re-use in a new scenario."""
        self.phase              = MergePhase.APPROACH
        self._long_counter      = LONG_NASH_STEP_INTERVAL
        self._lat_counter       = LAT_NASH_STEP_INTERVAL
        self._last_u_long       = 0.0
        self._last_delta_lat    = 0.0
        self._phase_hold_start  = None
        self._gap_search_start  = None
        self._long_lambda       = LONG_AUTHORITY_LAMBDA_MIN
        self._lat_lambda        = LAT_AUTHORITY_LAMBDA_MIN
        self._long_force_filt   = 0.0
        self._lat_force_filt    = 0.0
        self.mobil_approved     = False
        self.locked_leader      = None
        self.locked_follower    = None
        self._merge_locked      = False
        self.long_nash.reset()
        self.lat_nash.reset()
        for lst in self.data.values():
            lst.clear()
