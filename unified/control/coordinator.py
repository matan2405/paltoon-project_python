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
    LONG_NASH_NP, LONG_NASH_NU, LAT_NASH_NP, LAT_NASH_NU,
    LONG_NASH_Q_POS, LONG_NASH_Q_VEL,
    LONG_NASH_R1, LONG_NASH_R2, LONG_NASH_S1, LONG_NASH_S2,
    LONG_NASH_R2_START, LONG_NASH_R2_FOLLOW,
    LONG_NASH_U1_MIN, LONG_NASH_U1_MAX, LONG_NASH_U2_MIN, LONG_NASH_U2_MAX,
    LONG_NASH_DU1_MAX, LONG_NASH_DU2_MAX,
    LONG_NASH_V_MIN, LONG_NASH_V_MAX, LONG_NASH_GAP_MIN,
    LONG_NASH_LAMBDA_LEVELS,
    LAT_NASH_Q_Y, LAT_NASH_Q_PSI,
    LAT_NASH_Q_Y_TERMINAL_FAC, LAT_NASH_Q_PSI_TERMINAL_FAC,
    LAT_NASH_R1, LAT_NASH_R2, LAT_NASH_R2_START, LAT_NASH_R2_FOLLOW, LAT_NASH_S1, LAT_NASH_S2,
    AUTHORITY_GAP_ERROR_MAX, AUTHORITY_VEL_ERROR_MAX, FOLLOWING_GAP_ERROR_FACTOR,
    GAP_SEARCH_DURATION, LANE_CHANGE_MIN_TIME,
    DSF_SPEED_COEFF, DSF_SPEED_EXPONENT, DSF_SPEED_OFFSET, DSF_VEHICLE_MASS, DSF_EPSILON,
    LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX,
    LAT_NASH_Y_MAX, LAT_NASH_PSI_MAX, LAT_NASH_LAMBDA_LEVELS,
    LONG_AUTHORITY_LAMBDA_MIN, LONG_AUTHORITY_LAMBDA_MAX,
    LONG_AUTHORITY_FORCE_MIDPOINT, LONG_AUTHORITY_K_STEEPNESS,
    LONG_AUTHORITY_ALPHA_BASE, LONG_AUTHORITY_ALPHA_FAST,
    LONG_AUTHORITY_ALPHA_FOLLOWING,
    LONG_AUTHORITY_LAMBDA_MAX_FOLLOWING,
    LAT_AUTHORITY_LAMBDA_MIN, LAT_AUTHORITY_LAMBDA_MAX,
    LAT_AUTHORITY_FORCE_MIDPOINT, LAT_AUTHORITY_K_STEEPNESS,
    LAT_AUTHORITY_ALPHA_BASE, LAT_AUTHORITY_ALPHA_FAST,
    LAT_AUTHORITY_ALPHA_FAST_THR, LAT_AUTHORITY_ALPHA_BASE_THR,
    LAT_AUTHORITY_SIGMOID_M1, LAT_AUTHORITY_SIGMOID_M2,
    LONG_SAFETY_MIN_SAFE_DISTANCE, LONG_SAFETY_EMERGENCY_BRAKE_DIST,
    LONG_SAFETY_MAX_REPULSIVE_FORCE, LONG_SAFETY_FILTER_ALPHA,
    LONG_SAFETY_BASE_RADIUS, LONG_SAFETY_OBSTACLE_MASS, LONG_SAFETY_INFLUENCE_FACTOR,
    LONG_SAFETY_DRIVER_RISK, LONG_SAFETY_EPSILON, LONG_SAFETY_DISTANCE_DECAY,
    LONG_SAFETY_FOLLOWING_SOFT_FACTOR,
    LONG_SAFETY_LEADER_POSITION_MULT, LONG_SAFETY_MIDDLE_POSITION_MULT,
    LONG_SAFETY_FOLLOWER_POSITION_MULT, LONG_SAFETY_VELOCITY_REFERENCE,
    LONG_SAFETY_VELOCITY_SCALING, LONG_SAFETY_PLATOON_COHERENCE,
    LONG_SAFETY_FOLLOWER_WEIGHT,
    LAT_DSF_TS, LAT_DSF_TAU, LAT_DSF_A_MIN, LAT_DSF_G,
    LAT_DSF_K1, LAT_DSF_K2, LAT_DSF_DR,
    LAT_SAFETY_MAX_FORCE, LAT_SAFETY_FILTER_ALPHA,
    LAT_SAFETY_ROAD_HALF_WIDTH, LAT_SAFETY_BOUNDARY_FORCE_GAIN,
    LAT_SAFETY_BOUNDARY_FORCE_SCALE, LAT_SAFETY_BOUNDARY_PROXIMITY,
    LAT_SAFETY_BOUNDARY_EPSILON, LAT_SAFETY_DIRECTION_SMOOTH_SIGMA,
    PLATOON_TIME_GAP, PLATOON_STANDSTILL_DISTANCE, PLATOON_TARGET_VELOCITY,
    PLATOON_VEHICLE_LENGTH,
    RAJAMANI_H, RAJAMANI_K1, RAJAMANI_K2, RAJAMANI_K3, RAJAMANI_K4, RAJAMANI_K5,
    MOBIL_IDM_V0, MOBIL_IDM_T, MOBIL_IDM_A_MAX, MOBIL_IDM_B,
    MOBIL_IDM_S0, MOBIL_IDM_DELTA, MOBIL_IDM_L,
    MOBIL_P, MOBIL_B_SAFE, MOBIL_A_TH, MOBIL_MIN_GAP,
    APPROACH_MOBIL_CHECK_DISTANCE,
    MERGE_COMPLETE_Y_ERROR, MERGE_COMPLETE_PSI_ERROR, MERGE_COMPLETE_GAP_RATIO,
    PHASE_TRANSITION_HOLD_TIME,
    PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y, LANE_WIDTH, LAT_HUMAN_Y_BIAS,
    DRIVER_PARAMS,
    MAX_ACCELERATION, MAX_DECELERATION,
    NOMINAL_VELOCITY, VEHICLE_LENGTH, VEHICLE_MASS,
    VEHICLE_LF, VEHICLE_LR,
    LONG_NASH_REBUILD_DVX, LAT_NASH_REBUILD_DVX,
    MA_IDM_ENABLED, MA_IDM_PARAMS, MA_IDM_LAT_PARAMS,
    MA_IDM_WINDOW_SIZE, MA_IDM_OBS_NOISE, MA_IDM_LAT_OBS_NOISE,
    NASH_RISK_GAMMA,
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
# SE-kernel GP residual (MA-IDM, Zhang & Sun 2024, Eq. 9)
# =============================================================================

class _SEKernelGP:
    """Full squared-exponential (SE) kernel GP for sequential simulation.

    Implements exactly Eq. 9 of Zhang & Sun 2024:
        k(t, t') = σ_k² · exp(−|t−t'|² / (2ℓ²))

    Sequential sampling uses the GP conditional:
        p(ε(t+1) | history) = N(K_*X (K_XX)^{-1} hist, σ_k² − K_*X (K_XX)^{-1} K_*X')

    K_XX and its inverse are time-invariant (stationary kernel) and precomputed
    at init — each step costs only one O(W) matrix-vector product.

    predict_mean() returns the GP posterior mean E[ε(t+k)|history] for k=1..horizon,
    conditioned on the fixed current history (not rolled forward) — suitable for
    Nash planning horizon feedforward.
    """

    def __init__(self, sigma_k: float, ell: float, dt: float,
                 window: int = MA_IDM_WINDOW_SIZE,
                 obs_noise: float = MA_IDM_OBS_NOISE):
        self._sigma_k  = sigma_k
        self._ell      = ell
        self._dt       = dt
        self._W        = window

        # History buffer: hist[0]=oldest, hist[-1]=most recent ε
        self._hist = np.zeros(window)
        self._eps  = 0.0

        # ── Pre-compute K_XX (W×W, stationary — never changes) ───────────────
        j = np.arange(window)
        dt_mat = np.abs(j[:, None] - j[None, :]) * dt          # time-difference matrix
        K_XX = sigma_k ** 2 * np.exp(-dt_mat ** 2 / (2.0 * ell ** 2))
        K_XX += obs_noise * np.eye(window)                       # numerical stability
        self._K_XX_inv = np.linalg.inv(K_XX)

        # ── K_*X: cross-covariance of ε(t+1) with history ───────────────────
        # hist[j] is at time t − (W−1−j)·dt; next step at t+dt
        # → distance = (W − j)·dt
        dist_next = (window - j) * dt                            # [W·dt, …, dt]
        self._K_star_X = sigma_k ** 2 * np.exp(-dist_next ** 2 / (2.0 * ell ** 2))

        # Predictive variance (scalar, time-invariant)
        pred_var = float(sigma_k ** 2
                         - self._K_star_X @ self._K_XX_inv @ self._K_star_X)
        self._pred_std = float(np.sqrt(max(pred_var, 0.0)))

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self) -> float:
        """Sample ε(t+1) from GP conditional; advance history buffer."""
        mu      = float(self._K_star_X @ (self._K_XX_inv @ self._hist))
        eps_new = mu + self._pred_std * np.random.randn()
        self._hist[:-1] = self._hist[1:]   # shift left (drop oldest)
        self._hist[-1]  = eps_new          # append newest
        self._eps       = eps_new
        return eps_new

    def predict_mean(self, horizon: int, dt_query: float = None) -> np.ndarray:
        """GP posterior mean E[ε(t+k·dt_query)|history] for k=1..horizon.

        Parameters
        ----------
        horizon   : number of steps to predict
        dt_query  : time between query points [s].  Defaults to self._dt (simulation dt).
                    Pass the Nash control DT (e.g. LONG_NASH_CONTROL_DT) so that
                    predictions align with the Nash reference horizon instead of the
                    100 Hz simulation grid.

        Conditioned on the fixed current history — does not roll forward.
        """
        dt_q  = dt_query if dt_query is not None else self._dt
        alpha = self._K_XX_inv @ self._hist          # W-vector, reused for all k
        j     = np.arange(self._W)
        means = np.empty(horizon)
        for k in range(1, horizon + 1):
            # hist[j] is at t − (W−1−j)·dt_sim; query at t + k·dt_query
            # distance = k·dt_query + (W−1−j)·dt_sim
            dist  = k * dt_q + (self._W - 1 - j) * self._dt
            K_k_X = self._sigma_k ** 2 * np.exp(-dist ** 2 / (2.0 * self._ell ** 2))
            means[k - 1] = float(K_k_X @ alpha)
        return means

    @property
    def current(self) -> float:
        return self._eps

    @property
    def sigma_k(self) -> float:
        return self._sigma_k

    @property
    def ell(self) -> float:
        return self._ell


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

    def __init__(self, ego, driver_type: str = 'normal', merge_scenario: str = 'back',
                 noise_mode: str = 'gp',
                 fixed_lambda_long: Optional[float] = None,
                 fixed_lambda_lat: Optional[float] = None,
                 weight_overrides: Optional[dict] = None):
        """
        Parameters
        ----------
        ego : Vehicle6D
        driver_type : str
            'cautious' | 'normal' | 'aggressive'
        merge_scenario : str
            'back' | 'middle' | 'front'
        noise_mode : str
            'deterministic' — no noise injected
            'iid'           — i.i.d. Gaussian N(0, sigma_k²) per step (B-IDM baseline)
            'gp'            — MA-IDM SE-kernel GP (default, Zhang & Sun 2024)
        fixed_lambda_long : float or None
            If set, bypass the Safety Field and use this fixed λ for longitudinal Nash.
        fixed_lambda_lat : float or None
            If set, bypass the DSF and use this fixed λ for lateral Nash.
        weight_overrides : dict or None
            Override Nash cost weights. Keys (all optional):
              'long_Q_pos', 'long_Q_vel', 'long_R1', 'long_R2'
              'long_R2_start', 'long_R2_follow'   ← R2 adaptive range endpoints (long)
              'lat_Q_y', 'lat_Q_psi', 'lat_R1', 'lat_R2'
              'lat_R2_start',  'lat_R2_follow'    ← R2 adaptive range endpoints (lat)
        """
        self.ego         = ego
        self.driver_type = driver_type
        self.merge_scenario   = merge_scenario   # 'back' | 'middle' | 'front'
        self.noise_mode       = noise_mode        # 'deterministic' | 'iid' | 'gp'
        self._fixed_lambda_long = fixed_lambda_long
        self._fixed_lambda_lat  = fixed_lambda_lat
        self.locked_leader    = None  # set at APPROACH→MERGE transition
        self.locked_follower  = None  # set at APPROACH→MERGE transition
        self._merge_locked    = False
        self.phase       = MergePhase.APPROACH

        _wo = weight_overrides or {}   # resolved early — used by R2 state init below

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

        # ── Simulation time (updated every step, used by settling trajectory) ─
        self._sim_time: float = 0.0
        self._following_entry_time: Optional[float] = None

        # ── MERGE entry state (locked at GAP_SEARCH→MERGE transition) ─────────
        self._merge_entry_time: Optional[float] = None
        self._merge_entry_y: float = 0.0
        self._merge_T_lc_sys: Optional[float] = None
        self._merge_T_lc_hum: Optional[float] = None

        # ── GAP_SEARCH phase timer ────────────────────────────────────────────
        self._gap_search_start: Optional[float] = None

        # ── Authority smoothing state ─────────────────────────────────────────
        self._long_lambda = LONG_AUTHORITY_LAMBDA_MIN
        self._lat_lambda  = LAT_AUTHORITY_LAMBDA_MIN

        # ── Adaptive R2 state (interpolated per-step with merge progress) ─────
        # Endpoints can be overridden per-experiment via weight_overrides so that
        # the R-weight sweep (B2) tests different effort scales without breaking
        # the adaptive mechanism.  Defaults: config constants.
        self._r2_long_start  = float(_wo.get('long_R2_start', LONG_NASH_R2_START))
        self._r2_long_follow = float(_wo.get('long_R2_follow', LONG_NASH_R2_FOLLOW))
        self._r2_lat_start   = float(_wo.get('lat_R2_start',  LAT_NASH_R2_START))
        self._r2_lat_follow  = float(_wo.get('lat_R2_follow', LAT_NASH_R2_FOLLOW))
        self._r2_long = self._r2_long_start
        self._r2_lat  = self._r2_lat_start

        # ── Safety force filter state ─────────────────────────────────────────
        self._long_force_filt = 0.0
        self._lat_force_filt  = 0.0

        # ── Longitudinal DSF parameter filter state (EllipseLongitudinalSafetyField) ─
        # Mirrors split module: _s_filtered, _Mo_filtered, _Ri_filtered, _DRi_filtered.
        # Note: split module never updates these after init, so they act as fixed-blend
        # anchors (alpha*dynamic + (1-alpha)*base_value).
        self._long_s_filt  = LONG_SAFETY_BASE_RADIUS
        self._long_Mo_filt = LONG_SAFETY_OBSTACLE_MASS
        self._long_Ri_filt = LONG_SAFETY_INFLUENCE_FACTOR
        self._long_DR_filt = LONG_SAFETY_DRIVER_RISK

        # ── MA-IDM SE-kernel GP residual states (Zhang & Sun 2024, Eq. 9) ───────
        _ma     = MA_IDM_PARAMS.get(driver_type, MA_IDM_PARAMS['normal'])
        _ma_lat = MA_IDM_LAT_PARAMS.get(driver_type, MA_IDM_LAT_PARAMS['normal'])
        self._gp     = _SEKernelGP(
            sigma_k=_ma['sigma_k'], ell=_ma['ell'], dt=SIMULATION_DT,
        )
        self._gp_lat = _SEKernelGP(
            sigma_k=_ma_lat['sigma_k'], ell=_ma_lat['ell'], dt=SIMULATION_DT,
            obs_noise=MA_IDM_LAT_OBS_NOISE,
        )

        # ── Neuromuscular IIR filter state (Swain & Rath 2023) ───────────────
        # [psi_exec[k-1], psi_exec[k-2]] — real heading history for IIR initial conditions
        self._neuro_psi_state = np.array([
            float(ego.state.psi), float(ego.state.psi)
        ])

        # ── MOBIL ─────────────────────────────────────────────────────────────
        self.mobil         = MOBILLaneChange()
        self.mobil_approved = False
        self.mobil.set_politeness(driver_type)

        # ── Long Nash solver ──────────────────────────────────────────────────
        _gamma = NASH_RISK_GAMMA if (MA_IDM_ENABLED and noise_mode == 'gp') else 0.0
        long_params = LongNashParams(
            Np=LONG_NASH_NP, Nu=LONG_NASH_NU, dt=LONG_NASH_CONTROL_DT,
            Q_pos=_wo.get('long_Q_pos', LONG_NASH_Q_POS),
            Q_vel=_wo.get('long_Q_vel', LONG_NASH_Q_VEL),
            Q_pos_terminal=10.0 * _wo.get('long_Q_pos', LONG_NASH_Q_POS),
            Q_vel_terminal= 4.0 * _wo.get('long_Q_vel', LONG_NASH_Q_VEL),
            R1=_wo.get('long_R1', LONG_NASH_R1),
            R2=_wo.get('long_R2', LONG_NASH_R2),
            S1=LONG_NASH_S1, S2=LONG_NASH_S2,
            u1_min=LONG_NASH_U1_MIN, u1_max=LONG_NASH_U1_MAX,
            u2_min=LONG_NASH_U2_MIN, u2_max=LONG_NASH_U2_MAX,
            du1_max=LONG_NASH_DU1_MAX, du2_max=LONG_NASH_DU2_MAX,
            v_min=LONG_NASH_V_MIN,     v_max=LONG_NASH_V_MAX,
            gap_min=LONG_NASH_GAP_MIN,
            lambda_levels=LONG_NASH_LAMBDA_LEVELS,
            sigma_k=_ma['sigma_k'],  ell=_ma['ell'],  gamma_risk=_gamma,
        )
        self.long_nash = ConstrainedLongitudinalNashSolver(
            vehicle=ego.long_proxy,
            params=long_params,
        )

        # ── Lat Nash solver ───────────────────────────────────────────────────
        lat_params = LatNashParams(
            Np=LAT_NASH_NP, Nu=LAT_NASH_NU, dt=LAT_NASH_CONTROL_DT,
        )
        # Override weights from unified config (or from weight_overrides)
        _lat_Q_y   = _wo.get('lat_Q_y',  LAT_NASH_Q_Y)
        _lat_Q_psi = _wo.get('lat_Q_psi', LAT_NASH_Q_PSI)
        lat_params.Q_y               = _lat_Q_y
        lat_params.Q_psi             = _lat_Q_psi
        lat_params.Q_y_terminal      = LAT_NASH_Q_Y_TERMINAL_FAC   * _lat_Q_y
        lat_params.Q_psi_terminal    = LAT_NASH_Q_PSI_TERMINAL_FAC  * _lat_Q_psi
        lat_params.R1                = _wo.get('lat_R1', LAT_NASH_R1)
        lat_params.R2                = _wo.get('lat_R2', LAT_NASH_R2)
        self._lat_human_y_bias       = float(_wo.get('lat_human_y_bias', LAT_HUMAN_Y_BIAS))
        lat_params.S1                = LAT_NASH_S1
        lat_params.S2                = LAT_NASH_S2
        lat_params.delta_min         = LAT_NASH_DELTA_MIN
        lat_params.delta_max         = LAT_NASH_DELTA_MAX
        lat_params.lambda_levels     = LAT_NASH_LAMBDA_LEVELS
        lat_params.sigma_k           = _ma_lat['sigma_k']
        lat_params.ell               = _ma_lat['ell']
        lat_params.gamma_risk        = _gamma

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
        self._sim_time = sim_time

        # ── MA-IDM: advance GP residual states every sim step (execution) ──────
        if MA_IDM_ENABLED and self.noise_mode == 'gp':
            self._gp.step()
            self._gp_lat.step()

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

        # ── Noise injection (mode-dependent) ─────────────────────────────────
        u_long_out    = self._last_u_long
        delta_lat_out = self._last_delta_lat
        _in_lat_phase = self.phase in (MergePhase.MERGE, MergePhase.FOLLOWING)

        if self.noise_mode == 'gp' and MA_IDM_ENABLED:
            # MA-IDM: SE-kernel GP residual (Zhang & Sun 2024)
            u_long_out = float(np.clip(
                u_long_out + self._gp.current, MAX_DECELERATION, MAX_ACCELERATION
            ))
            if _in_lat_phase:
                _L_wb = VEHICLE_LF + VEHICLE_LR
                _vx   = max(float(self.ego.state.vx), 1.0)
                delta_lat_out = float(np.clip(
                    delta_lat_out + self._gp_lat.current * _L_wb / _vx,
                    LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX
                ))
        elif self.noise_mode == 'iid':
            # B-IDM baseline: i.i.d. Gaussian N(0, σ_k²) — no temporal correlation
            eps_long = float(self._gp.sigma_k * np.random.randn())
            u_long_out = float(np.clip(
                u_long_out + eps_long, MAX_DECELERATION, MAX_ACCELERATION
            ))
            if _in_lat_phase:
                eps_lat = float(self._gp_lat.sigma_k * np.random.randn())
                _L_wb = VEHICLE_LF + VEHICLE_LR
                _vx   = max(float(self.ego.state.vx), 1.0)
                delta_lat_out = float(np.clip(
                    delta_lat_out + eps_lat * _L_wb / _vx,
                    LAT_NASH_DELTA_MIN, LAT_NASH_DELTA_MAX
                ))
        # 'deterministic': no noise — u_long_out and delta_lat_out unchanged

        return u_long_out, delta_lat_out

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

        # Compute dynamic DSF parameters (position mult, velocity scaling, EMA blend)
        # mirrors EllipseLongitudinalSafetyField._compute_dynamic_* methods
        _s, _Mo, _Ri, _DR = self._long_dsf_params(
            float(ego.state.vx),
            has_leader=(leader is not None),
            has_follower=(follower is not None),
        )

        # Pre-compute raw components for logging (unfiltered)
        f_leader_raw   = self._long_leader_force_raw(leader,   _s, _Mo, _Ri, _DR) if leader   is not None else 0.0
        f_follower_raw = self._long_follower_force_raw(follower, _s, _Mo, _Ri, _DR) if follower is not None else 0.0
        rear_gap = (ego.state.x - follower.state.x - VEHICLE_LENGTH
                    ) if follower is not None else float('nan')

        if leader is None:
            # 'front' scenario or free road: no forward obstacle.
            # Mirrors split-module: F_leader=0, F_total=F_follower (line 401).
            force = self._long_safety_force(leader=None, follower=follower,
                                            s=_s, Mo=_Mo, Ri=_Ri, DR=_DR)
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
        force = self._long_safety_force(leader, follower=follower, s=_s, Mo=_Mo, Ri=_Ri, DR=_DR)

        # Compute gap_error and velocity_error for authority computation
        _gap_err = self._long_gap_error(leader)   # positive = too close
        _vel_err = self.ego.state.vx - float(
            getattr(leader, 'v', getattr(leader.state, 'vx', PLATOON_TARGET_VELOCITY)))

        # Authority (safety + gap/velocity Swain & Rath) — or fixed if set
        if self._fixed_lambda_long is not None:
            lam = float(self._fixed_lambda_long)
            self._long_lambda = lam   # keep smoothed state in sync for logging
        else:
            lam = self._long_authority(force, gap_error=_gap_err, vel_error=_vel_err)

        # Reference trajectories
        R1 = self._long_sys_ref(leader, LONG_NASH_NP, LONG_NASH_CONTROL_DT)
        R2 = self._long_hum_ref(leader, LONG_NASH_NP, LONG_NASH_CONTROL_DT)

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

        # Advance neuromuscular IIR history with real ego heading (runs at 20 Hz)
        _psi_now = float(self.ego.state.psi)
        self._neuro_psi_state[1] = self._neuro_psi_state[0]
        self._neuro_psi_state[0] = _psi_now

        ego     = self.ego
        x0      = ego.get_lat_state_vector()   # [y, vy, psi, psi_dot]
        target_y = PLATOON_LANE_Y

        # Safety force (DSF)
        force = self._lat_safety_force(platoon_vehicles)

        # Authority — or fixed if set
        y_err = ego.state.y - target_y
        if self._fixed_lambda_lat is not None:
            lam = float(self._fixed_lambda_lat)
            self._lat_lambda = lam
        else:
            lam = self._lat_authority(force, y_err)

        # Reference trajectories
        R1 = self._lat_sys_ref(target_y, LAT_NASH_NP, LAT_NASH_CONTROL_DT)
        R2 = self._lat_hum_ref(LAT_NASH_NP, LAT_NASH_CONTROL_DT)

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
                # R2 is now managed adaptively in _long_authority / _lat_authority per-step
                # Rebuild lateral Nash matrices once at cruise speed before first MERGE solve
                self._rebuild_lat_nash_once()
                # Reset neuromuscular IIR to current heading (clean entry into merge)
                _psi_entry = float(self.ego.state.psi)
                self._neuro_psi_state[:] = _psi_entry
                # Lock MERGE entry state for time-based trajectory
                self._merge_entry_time = sim_time
                self._merge_entry_y = float(self.ego.state.y)
                self._merge_T_lc_sys = None
                self._merge_T_lc_hum = None
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
                    self._following_entry_time = sim_time
                    # R2 is now managed adaptively in _long_authority per-step
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
    # Longitudinal safety field  (Li et al. 2019, Wang et al. 2015/2016)
    # Mirrors EllipseLongitudinalSafetyField in longitudinal_safety_field.py
    # =========================================================================

    def _long_gap_error(self, leader) -> float:
        """Signed gap error: positive = too close."""
        ego = self.ego
        gap     = leader.state.x - ego.state.x - VEHICLE_LENGTH
        des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx
        return des_gap - gap

    def _long_position_mult(self, has_leader: bool, has_follower: bool) -> float:
        """Position-based multiplier on s and Mo (mirrors _get_position_multiplier).

        LEADER_MULT  → ego has no leader (platoon head)
        MIDDLE_MULT  → ego has both leader and follower (middle position)
        FOLLOWER_MULT→ ego has leader but no follower (tail)
        """
        if not has_leader:
            return LONG_SAFETY_LEADER_POSITION_MULT
        elif has_follower:
            return LONG_SAFETY_MIDDLE_POSITION_MULT
        else:
            return LONG_SAFETY_FOLLOWER_POSITION_MULT

    def _long_dsf_params(self, ego_vx: float,
                          has_leader: bool, has_follower: bool):
        """Compute dynamic DSF parameters (s, Mo, Ri, DR).

        Mirrors EllipseLongitudinalSafetyField._compute_dynamic_* methods:
          s  = base_radius × pos_mult × vel_factor  (EMA-blended with base)
          Mo = base_mass   × pos_mult               (EMA-blended with base)
          Ri = base_Ri × coherence_factor_if_middle  (EMA-blended with base)
          DR = base_DR                               (EMA-blended; constant in practice)

        Note: the split module never updates the filter state after __init__, so
        the EMA is effectively a fixed blend: alpha*dynamic + (1-alpha)*base.
        """
        pos_mult = self._long_position_mult(has_leader, has_follower)

        # s: velocity-scaled safety radius (mirrors _compute_dynamic_safety_radius)
        s_target     = LONG_SAFETY_BASE_RADIUS * pos_mult
        vel_factor   = 1.0 + LONG_SAFETY_VELOCITY_SCALING * (
            ego_vx / LONG_SAFETY_VELOCITY_REFERENCE - 1.0)
        s_target    *= max(0.5, min(2.0, vel_factor))
        s = (LONG_SAFETY_FILTER_ALPHA * s_target
             + (1.0 - LONG_SAFETY_FILTER_ALPHA) * self._long_s_filt)

        # Mo: position-weighted obstacle mass (mirrors _compute_dynamic_obstacle_mass)
        Mo_target = LONG_SAFETY_OBSTACLE_MASS * pos_mult
        Mo = (LONG_SAFETY_FILTER_ALPHA * Mo_target
              + (1.0 - LONG_SAFETY_FILTER_ALPHA) * self._long_Mo_filt)

        # Ri: platoon coherence factor in middle position (mirrors _compute_dynamic_influence_factor)
        # road_condition always 'dry' → road_factor=1.0; high_risk_situation always False
        Ri_target = LONG_SAFETY_INFLUENCE_FACTOR
        if has_leader and has_follower:
            Ri_target *= LONG_SAFETY_PLATOON_COHERENCE
        Ri = (LONG_SAFETY_FILTER_ALPHA * Ri_target
              + (1.0 - LONG_SAFETY_FILTER_ALPHA) * self._long_Ri_filt)

        # DR: base driver risk (mirrors _compute_dynamic_driving_risk)
        DR = (LONG_SAFETY_FILTER_ALPHA * LONG_SAFETY_DRIVER_RISK
              + (1.0 - LONG_SAFETY_FILTER_ALPHA) * self._long_DR_filt)

        return s, Mo, Ri, DR

    def _long_leader_force_raw(self, leader, s: float, Mo: float,
                                Ri: float, DR: float) -> float:
        """Raw force from the leader [N] using Li et al. (2019) elliptic DSF.

        Force > 0: leader too close (repulsive).
        Force < 0: gap too large (attractive pull — not in split module but needed
                   to pull ego toward platoon when falling behind).
        Uses dynamic parameters (s, Mo, Ri, DR) pre-computed by _long_dsf_params.
        """
        ego     = self.ego
        gap     = leader.state.x - ego.state.x - VEHICLE_LENGTH
        des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx

        # Hard safety zones — MIN_SAFE is tighter (inner), EMERGENCY_BRAKE is outer
        if gap < LONG_SAFETY_MIN_SAFE_DISTANCE:
            return LONG_SAFETY_MAX_REPULSIVE_FORCE
        elif gap < LONG_SAFETY_EMERGENCY_BRAKE_DIST:
            t = 1.0 - (gap - LONG_SAFETY_MIN_SAFE_DISTANCE) / max(
                LONG_SAFETY_EMERGENCY_BRAKE_DIST - LONG_SAFETY_MIN_SAFE_DISTANCE, 1e-3)
            return t * LONG_SAFETY_MAX_REPULSIVE_FORCE

        gap_error = des_gap - gap          # >0 → too close, <0 → too far
        v_rel     = ego.state.vx - leader.state.vx  # >0 when closing

        if gap_error <= 0.0:
            # Gap >= desired: linear attractive pull (bidirectional, capped)
            return float(np.clip(
                gap_error / max(des_gap, 1.0) * LONG_SAFETY_MAX_REPULSIVE_FORCE,
                -0.25 * LONG_SAFETY_MAX_REPULSIVE_FORCE,
                0.0,
            ))
        # Gap < desired: Li et al. (2019) elliptic DSF repulsive force
        r_ell     = gap_error / (s + LONG_SAFETY_EPSILON)
        potential = (Mo * Ri) / (r_ell + LONG_SAFETY_EPSILON) ** 2
        w_dist    = np.exp(-gap_error / LONG_SAFETY_DISTANCE_DECAY)
        w_vel     = np.exp(max(0.0, v_rel) / 5.0)
        w_risk    = 1.0 + DR
        return float(min(potential * w_dist * w_vel * w_risk, LONG_SAFETY_MAX_REPULSIVE_FORCE))

    def _long_follower_force_raw(self, follower, s: float, Mo: float,
                                  Ri: float, DR: float) -> float:
        """Raw repulsive force from the follower [N].

        Force >= 0: follower too close → ego must accelerate (push-forward risk).
        No attractive component — mirrors _compute_force_to_vehicle(is_leader=False).
        Uses dynamic parameters (s, Mo, Ri, DR) pre-computed by _long_dsf_params.
        """
        ego      = self.ego
        gap_rear = ego.state.x - follower.state.x - VEHICLE_LENGTH
        des_gap  = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx

        if gap_rear < LONG_SAFETY_MIN_SAFE_DISTANCE:
            return LONG_SAFETY_MAX_REPULSIVE_FORCE
        elif gap_rear < LONG_SAFETY_EMERGENCY_BRAKE_DIST:
            t = 1.0 - (gap_rear - LONG_SAFETY_MIN_SAFE_DISTANCE) / max(
                LONG_SAFETY_EMERGENCY_BRAKE_DIST - LONG_SAFETY_MIN_SAFE_DISTANCE, 1e-3)
            return t * LONG_SAFETY_MAX_REPULSIVE_FORCE

        gap_error = des_gap - gap_rear   # >0 when follower too close
        if gap_error <= 0.0:
            return 0.0  # follower is safe distance away — no repulsive force

        v_rel     = follower.state.vx - ego.state.vx  # >0 when follower closing
        r_ell     = gap_error / (s + LONG_SAFETY_EPSILON)
        potential = (Mo * Ri) / (r_ell + LONG_SAFETY_EPSILON) ** 2
        w_dist    = np.exp(-gap_error / LONG_SAFETY_DISTANCE_DECAY)
        w_vel     = np.exp(max(0.0, v_rel) / 5.0)
        w_risk    = 1.0 + DR
        return float(min(potential * w_dist * w_vel * w_risk, LONG_SAFETY_MAX_REPULSIVE_FORCE))

    def _long_safety_force(self, leader, follower=None,
                            s: float = None, Mo: float = None,
                            Ri: float = None, DR: float = None) -> float:
        """Compute total longitudinal safety field force [N].

        Mirrors EllipseLongitudinalSafetyField.compute_risk_force_from_platoon
        (longitudinal_safety_field.py lines 368–418):
          F_leader  = _compute_force_to_vehicle(is_leader=True)
          F_follower = follower_weight × _compute_force_to_vehicle(is_leader=False)
          In FOLLOWING phase: apply _apply_soft_transition SEPARATELY to each
            using their own gap errors and FOLLOWING_GAP_ERROR_FACTOR threshold.
          F_total = F_leader + F_follower  (then EMA low-pass filtered)
        """
        # Fall back to base values if params not provided (should not happen in practice)
        if s is None:
            s = LONG_SAFETY_BASE_RADIUS
        if Mo is None:
            Mo = LONG_SAFETY_OBSTACLE_MASS
        if Ri is None:
            Ri = LONG_SAFETY_INFLUENCE_FACTOR
        if DR is None:
            DR = LONG_SAFETY_DRIVER_RISK

        ego     = self.ego
        des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx

        F_leader = self._long_leader_force_raw(leader, s, Mo, Ri, DR) if leader is not None else 0.0
        F_follower_raw = (self._long_follower_force_raw(follower, s, Mo, Ri, DR)
                          if follower is not None else 0.0)
        # Apply follower weight (mirrors _get_follower_weight; always FOLLOWER_WEIGHT
        # since follower_is_joining and follower_decelerating default to False in split module)
        F_follower = LONG_SAFETY_FOLLOWER_WEIGHT * F_follower_raw

        # Quadratic soft transition in FOLLOWING phase (mirrors _apply_soft_transition).
        # Applied SEPARATELY to leader and follower using their own gap errors,
        # with threshold = FOLLOWING_GAP_ERROR_FACTOR × desired_gap (matches split module).
        if self.phase == MergePhase.FOLLOWING:
            threshold = FOLLOWING_GAP_ERROR_FACTOR * max(des_gap, 1.0)
            if threshold > 0 and leader is not None:
                gap      = leader.state.x - ego.state.x - VEHICLE_LENGTH
                gap_err  = abs(des_gap - gap)
                F_leader *= min(1.0, (gap_err / threshold) ** 2)
            if threshold > 0 and follower is not None:
                gap_rear     = ego.state.x - follower.state.x - VEHICLE_LENGTH
                gap_err_rear = abs(des_gap - gap_rear)
                F_follower  *= min(1.0, (gap_err_rear / threshold) ** 2)

        raw = F_leader + F_follower
        self._long_force_filt = (LONG_SAFETY_FILTER_ALPHA * raw
                                 + (1.0 - LONG_SAFETY_FILTER_ALPHA) * self._long_force_filt)
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

        # --- 2b. Adaptive R2: interpolate with merge-progress signal l_n ---
        # l_n≈0 → far from platoon (large errors) → R2=R2_START (system leads).
        # l_n≈1 → integrated (small errors)        → R2=R2_FOLLOW (human leads).
        # Endpoints are instance vars so experiments can scale them via weight_overrides.
        _r2_target    = self._r2_long_start + (self._r2_long_follow - self._r2_long_start) * l_n
        _r2_new       = 0.05 * _r2_target + 0.95 * self._r2_long
        if abs(_r2_new - self._r2_long) > 50.0:   # skip rebuild if change < 50 N·s²/m
            self._r2_long = _r2_new
            self.long_nash.update_r2(self._r2_long)

        # --- 3. Fusion: max urgency ---
        lam_raw = min(max(lam_safety, lam_gap), LONG_AUTHORITY_LAMBDA_MAX)

        # --- 3b. FOLLOWING phase: cap λ so human always retains ≥33% tracking weight ---
        if self.phase == MergePhase.FOLLOWING:
            lam_raw = min(lam_raw, LONG_AUTHORITY_LAMBDA_MAX_FOLLOWING)

        # --- 4. Adaptive EMA ---
        # FOLLOWING uses a faster time constant (~3 s) so λ tracks small perturbations
        # without the slow APPROACH ramp-up delay.
        combined_error = max(abs_gap_err, abs_vel_err)
        if self.phase == MergePhase.FOLLOWING:
            alpha = LONG_AUTHORITY_ALPHA_FOLLOWING
        elif combined_error > 5.0:
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
            lateral_direction = np.tanh(dy / LAT_SAFETY_DIRECTION_SMOOTH_SIGMA)
            total_force += Fr * lateral_direction

        # Road boundary forces (mirrors lateral/config.py SAFETY_* constants)
        dist_right = LAT_SAFETY_ROAD_HALF_WIDTH - ego_y
        if 0 < dist_right < LAT_SAFETY_BOUNDARY_PROXIMITY:
            potential = 1.0 / max(dist_right, LAT_SAFETY_BOUNDARY_EPSILON)
            total_force -= LAT_SAFETY_BOUNDARY_FORCE_GAIN * np.tanh(
                potential / LAT_SAFETY_BOUNDARY_FORCE_SCALE)

        dist_left = LAT_SAFETY_ROAD_HALF_WIDTH + ego_y
        if 0 < dist_left < LAT_SAFETY_BOUNDARY_PROXIMITY:
            potential = 1.0 / max(dist_left, LAT_SAFETY_BOUNDARY_EPSILON)
            total_force += LAT_SAFETY_BOUNDARY_FORCE_GAIN * np.tanh(
                potential / LAT_SAFETY_BOUNDARY_FORCE_SCALE)

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

        # --- 2b. Adaptive R2: interpolate with lane-change progress l_n ---
        # l_n≈0 → far from target lane → R2=R2_START (matched effort, system leads via high λ).
        # l_n≈1 → lane-centred         → R2=R2_FOLLOW (human leads laterally).
        # Endpoints are instance vars so experiments can scale them via weight_overrides.
        _r2_lat_target = self._r2_lat_start + (self._r2_lat_follow - self._r2_lat_start) * l_n
        _r2_lat_new    = 0.05 * _r2_lat_target + 0.95 * self._r2_lat
        if abs(_r2_lat_new - self._r2_lat) > 200.0:   # skip rebuild if change < 200 N·s²/m
            self._r2_lat = _r2_lat_new
            self.lat_nash.update_r2(self._r2_lat)

        m1, m2  = LAT_AUTHORITY_SIGMOID_M1, LAT_AUTHORITY_SIGMOID_M2
        gamma1  = 1.0 / (1.0 + np.exp(m1 * (-l_n + m2)))
        gamma2  = 1.0 - gamma1
        lam_p   = gamma2 / max(gamma1, 1e-6)
        lam_p   = float(np.clip(lam_p, LAT_AUTHORITY_LAMBDA_MIN, LAT_AUTHORITY_LAMBDA_MAX))

        lam_raw = max(lam_s, lam_p)

        # --- 3. Adaptive EMA ---
        e_abs = abs(y_error)
        if e_abs > LAT_AUTHORITY_ALPHA_FAST_THR:
            alpha = LAT_AUTHORITY_ALPHA_FAST
        elif e_abs < LAT_AUTHORITY_ALPHA_BASE_THR:
            alpha = LAT_AUTHORITY_ALPHA_BASE
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
        """R2: Human — phase-aware IDM prediction (mirrors split-module HumanDriver).

        Three-mode behaviour matching the split module's ignore_platoon_before_merge logic:
          APPROACH / GAP_SEARCH : free-road only (leader suppressed) — human accelerates
                                  toward target speed; gap is still large so no collision
                                  risk; u2 > 0 → cooperative with system u1 > 0.
          MERGE                 : IDM with leader, plan_T=1.0 s — gap-aware but less
                                  conservative than T=1.5; smaller IDM-Rajamani conflict.
          FOLLOWING             : IDM with leader, plan_T from driver profile (1.5 s) —
                                  standard shared control.

        Returns (Np*2,) flattened [x0,vx0, x1,vx1, ...].
        """
        dp = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])

        # IDM parameters (from driver profile / longitudinal config defaults)
        a_max      = MOBIL_IDM_A_MAX         # [m/s²]
        v0         = PLATOON_TARGET_VELOCITY + dp.get('velocity_offset', 0.0)  # desired speed
        delta_idm  = MOBIL_IDM_DELTA         # acceleration exponent
        plan_b     = dp.get('plan_decel', 4.0)          # planning deceleration [m/s²]
        s0         = MOBIL_IDM_S0            # minimum spacing [m]
        a_min      = MAX_DECELERATION        # maximum deceleration [m/s²]

        # Phase-dependent time headway
        if self.phase == MergePhase.MERGE:
            plan_T = 1.0   # assertive but gap-aware; T=1.0 < T=1.5 → less IDM-Rajamani conflict
        else:
            plan_T = dp.get('plan_time_headway', 2.0)   # FOLLOWING: driver-profile default (1.5 s)

        x  = float(self.ego.state.x)
        vx = float(self.ego.state.vx)

        # Simulate leader at constant velocity (planning assumption)
        lx = float(leader.state.x) if leader is not None else None
        lv = float(getattr(leader, 'v', getattr(leader.state, 'vx', PLATOON_TARGET_VELOCITY))) if leader is not None else 0.0

        # APPROACH / GAP_SEARCH: suppress leader so human uses free-road only.
        # Mirrors split module ignore_platoon_before_merge=True: human accelerates
        # toward v0 → u2 > 0, cooperative with u1. Safe because gap is still large.
        if self.phase in (MergePhase.APPROACH, MergePhase.GAP_SEARCH):
            lx = None

        ref = np.empty(Np * 2)
        gp_means = self._gp.predict_mean(Np, dt_query=LONG_NASH_CONTROL_DT) if MA_IDM_ENABLED else None
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
            if gp_means is not None:
                a_human = float(np.clip(a_human + gp_means[k], a_min, a_max))

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

        T scales with |dy| via the heading constraint, so the polynomial naturally
        handles all phases without a phase check:
          - MERGE  (large dy): gradual multi-second lane change
          - FOLLOWING (tiny dy): T≈dt → tau=1 at every step → reference = target_y

        Returns (Np*2,) flattened [y0,psi0, y1,psi1, ...].
        """
        y0   = float(self.ego.state.y)
        dy   = target_y - y0
        dp   = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])
        max_heading_rad = np.radians(dp.get('max_heading_deg', 5.0))
        vx   = max(float(self.ego.state.vx), 1.0)

        # T from heading constraint: T = 1.875·|dy| / (vx·tan(ψ_max)).
        # Scales linearly with |dy| — no artificial floor needed:
        #   large dy (MERGE, ~3.5 m)  → T ≈ 3–4 s, gradual lane change
        #   tiny dy  (FOLLOWING, ~mm) → T ≈ dt, tau=1 at every step → target_y immediately
        T = max(dt, 1.875 * abs(dy) / (vx * np.tan(max_heading_rad))) if abs(dy) > 1e-4 else dt

        # 5th-order polynomial coefficients (boundary: y(0)=y0, y(T)=yf, ẏ=0, ÿ=0 at both ends)
        # y(t) = y0 + dy * (10τ³ - 15τ⁴ + 6τ⁵),  τ = t/T
        # Returns (Np*2,) flattened [y0,psi0, y1,psi1, ...].
        ref = np.empty(Np * 2)

        if self.phase == MergePhase.FOLLOWING and self._following_entry_time is not None:
            # Settling trajectory (mirrors lateral module _generate_settling_trajectory):
            # cubic polynomial from current (y, ẏ) → (target_y, 0) over T_remaining.
            # Receding horizon: t_pred starts at 0 so ref[0] always matches current state.
            t_in_following = self._sim_time - self._following_entry_time
            T_settle = float(dp.get('system_settle_time', 20.0))
            T_remaining = max(T_settle - t_in_following, dt)

            vy0 = float(self.ego.state.vy)
            if abs(dy) > 1e-6:
                vy_max = 1.5 * abs(dy) / T_remaining
                vy0 = float(np.clip(vy0, -vy_max, vy_max))
            else:
                vy0 = 0.0

            a0 = y0
            a1 = vy0 * T_remaining
            a2 = 3.0 * dy - 2.0 * vy0 * T_remaining
            a3 = -2.0 * dy + vy0 * T_remaining

            for k in range(Np):
                t_pred = k * dt  # receding: 0, dt, 2*dt, …
                tau = min(t_pred / T_remaining, 1.0)
                if tau < 1.0:
                    y    = a0 + a1*tau + a2*tau**2 + a3*tau**3
                    ydot = (a1 + 2.0*a2*tau + 3.0*a3*tau**2) / T_remaining
                    psi  = float(np.clip(ydot / vx, -0.3, 0.3))
                else:
                    y   = target_y
                    psi = 0.0
                ref[2 * k]     = y
                ref[2 * k + 1] = psi
        elif self.phase == MergePhase.MERGE and self._merge_entry_time is not None:
            # MERGE: time-based 5th-order polynomial locked to entry state.
            # tau advances with wall-clock time so the reference shape is fixed
            # and doesn't collapse as the vehicle approaches the target.
            y_start   = self._merge_entry_y
            dy_merge  = target_y - y_start
            if self._merge_T_lc_sys is None:
                self._merge_T_lc_sys = (
                    max(dt, 1.875 * abs(dy_merge) / (vx * np.tan(max_heading_rad)))
                    if abs(dy_merge) > 1e-4 else dt
                )
            T_lc = self._merge_T_lc_sys
            t_elapsed = self._sim_time - self._merge_entry_time
            for k in range(Np):
                t_total = t_elapsed + (k + 1) * dt
                tau = min(t_total / T_lc, 1.0)
                y = y_start + dy_merge * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
                if tau < 1.0:
                    ydot = (dy_merge / T_lc) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
                    psi  = float(np.clip(ydot / vx, -0.3, 0.3))
                else:
                    psi = 0.0
                ref[2 * k]     = y
                ref[2 * k + 1] = psi
        else:
            # APPROACH / GAP_SEARCH: position-based 5th-order polynomial
            for k in range(Np):
                t    = (k + 1) * dt
                tau  = min(t / T, 1.0)
                y    = y0 + dy * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
                ydot = (dy / max(T, 1e-6)) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
                psi  = float(np.clip(ydot / max(vx, 0.1), -0.3, 0.3))
                ref[2 * k]     = y
                ref[2 * k + 1] = psi
        return ref

    def _lat_hum_ref(self, Np: int, dt: float) -> np.ndarray:
        """R2 (human): three-layer lateral human reference.

        Layer 1 (Gu & Dolan): geometry-aware T_lc = 1.5·|Δy|/(vx·tan(ψ_max)).
        Layer 2 (Swain & Rath 2023): neuromuscular IIR G_d(z) converts ψ_des → ψ_exec.
        Layer 3 (Zhang & Sun 2024): GP models residual ε = ψ_actual − G_d·ψ_des.

        y column: polynomial only (Option B — avoids y/ψ drift that destabilises solver).
        Returns (Np*2,) flattened [y0,psi0, y1,psi1, ...].
        """
        y0    = float(self.ego.state.y)
        psi0  = float(self.ego.state.psi)
        vy0   = float(self.ego.state.vy)
        vx    = max(float(self.ego.state.vx), 1.0)
        T_lc  = max(self._tlc, 1.0)
        target_y = PLATOON_LANE_Y + self._lat_human_y_bias

        gp_lat_means = self._gp_lat.predict_mean(Np, dt_query=dt) if MA_IDM_ENABLED else None

        # Layer 1 params — moved before branch so neuro_J/B/K available in both phases
        dp = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])
        max_heading_rad = np.radians(dp.get('max_heading_deg', 5.0))

        # Layer 2: IIR coefficients (Swain & Rath 2023, Eq. 7, Euler discretisation)
        _neuro_J = float(dp.get('neuro_J', 0.0037))
        _neuro_B = float(dp.get('neuro_B', 0.1363))
        _neuro_K = float(dp.get('neuro_K', 1.1742))
        _nd  = _neuro_J + _neuro_B * dt + _neuro_K * dt * dt
        _na0 = (dt * dt) / _nd
        _na1 = (2.0 * _neuro_J + _neuro_B * dt) / _nd
        _na2 = -_neuro_J / _nd

        ref = np.empty(Np * 2)

        # IIR initial conditions from real-world history — prediction-local, not written back
        _psi_km1 = self._neuro_psi_state[0]
        _psi_km2 = self._neuro_psi_state[1]

        if self.phase == MergePhase.FOLLOWING and self._following_entry_time is not None:
            # FOLLOWING: settling trajectory — cubic polynomial from current (y, ẏ) → (target_y, 0).
            # IIR filters psi_des → psi_exec to model neuromuscular lag during settling.
            delta_y_f = target_y - y0
            t_in_following = self._sim_time - self._following_entry_time
            T_settle_h = float(dp.get('human_settle_time', 20.0))
            T_remaining_h = max(T_settle_h - t_in_following, dt)

            if abs(delta_y_f) > 1e-6:
                vy_max_h = 1.5 * abs(delta_y_f) / T_remaining_h
                vy0_h = float(np.clip(vy0, -vy_max_h, vy_max_h))
            else:
                vy0_h = 0.0

            a0_h = y0
            a1_h = vy0_h * T_remaining_h
            a2_h = 3.0 * delta_y_f - 2.0 * vy0_h * T_remaining_h
            a3_h = -2.0 * delta_y_f + vy0_h * T_remaining_h

            for k in range(Np):
                t_pred = k * dt
                tau_h = min(t_pred / T_remaining_h, 1.0)
                if tau_h < 1.0:
                    y_ref  = a0_h + a1_h*tau_h + a2_h*tau_h**2 + a3_h*tau_h**3
                    ydot_h = (a1_h + 2.0*a2_h*tau_h + 3.0*a3_h*tau_h**2) / T_remaining_h
                    psi_des = float(np.clip(ydot_h / vx, -0.3, 0.3))
                else:
                    y_ref   = target_y
                    psi_des = 0.0
                psi_exec = _na0 * psi_des + _na1 * _psi_km1 + _na2 * _psi_km2
                _psi_km2 = _psi_km1
                _psi_km1 = psi_exec
                ref[2 * k]     = y_ref
                ref[2 * k + 1] = psi_exec

        elif self.phase == MergePhase.MERGE and self._merge_entry_time is not None:
            # MERGE: time-based cubic polynomial + Layer 2 IIR.
            # Locked to entry state — tau advances with wall-clock time.
            delta_y = target_y - self._merge_entry_y
            if self._merge_T_lc_hum is None:
                self._merge_T_lc_hum = (
                    max(dt, 1.5 * abs(delta_y) / (vx * np.tan(max_heading_rad)))
                    if abs(delta_y) > 1e-4 else dt
                )
            T_lc_h  = self._merge_T_lc_hum
            t_elapsed = self._sim_time - self._merge_entry_time

            for k in range(Np):
                t_total = t_elapsed + (k + 1) * dt
                tau = min(t_total / T_lc_h, 1.0)
                if tau < 1.0:
                    y_ref     = self._merge_entry_y + delta_y * (3 * tau**2 - 2 * tau**3)
                    y_dot_ref = delta_y * (6 * tau - 6 * tau**2) / T_lc_h
                    psi_des   = y_dot_ref / vx
                else:
                    y_ref   = target_y
                    psi_des = 0.0
                psi_exec = (_na0 * psi_des + _na1 * _psi_km1 + _na2 * _psi_km2) if tau < 1.0 else 0.0
                _psi_km2 = _psi_km1
                _psi_km1 = psi_exec
                ref[2 * k]     = y_ref
                ref[2 * k + 1] = psi_exec
        else:
            # APPROACH / GAP_SEARCH: human holds current lane, psi decays with T_lc
            for k in range(Np):
                t       = (k + 1) * dt
                psi_des = psi0 * np.exp(-t / max(T_lc, 1e-3))
                psi_exec = _na0 * psi_des + _na1 * _psi_km1 + _na2 * _psi_km2
                _psi_km2 = _psi_km1
                _psi_km1 = psi_exec
                ref[2 * k]     = y0
                ref[2 * k + 1] = psi_exec

        # Layer 3: GP ψ-space feedforward — residual of G_d (Zhang & Sun 2024)
        if gp_lat_means is not None:
            for k in range(Np):
                ref[2 * k]     += vx * gp_lat_means[k] * dt
                ref[2 * k + 1] += gp_lat_means[k]

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
        self._following_entry_time = None
        self._merge_entry_time  = None
        self._merge_entry_y     = 0.0
        self._merge_T_lc_sys    = None
        self._merge_T_lc_hum    = None
        self._long_lambda       = LONG_AUTHORITY_LAMBDA_MIN
        self._lat_lambda        = LAT_AUTHORITY_LAMBDA_MIN
        self._long_force_filt   = 0.0
        self._lat_force_filt    = 0.0
        self.mobil_approved     = False
        self.locked_leader      = None
        self.locked_follower    = None
        self._merge_locked      = False
        _psi_reset = float(self.ego.state.psi)
        self._neuro_psi_state   = np.array([_psi_reset, _psi_reset])
        self.long_nash.reset()
        self.lat_nash.reset()
        for lst in self.data.values():
            lst.clear()
