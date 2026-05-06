"""
unified/config.py — Single source of truth for the unified longitudinal-lateral module.

Design principle:
  This file defines ONLY the parameters needed by the new unified files
  (vehicle_6d, unified_coordinator, unified_simulator). The existing
  Longitudinal/ and lateral/ modules keep their own config.py unchanged.

Conflict resolution:
  Parameters that exist in both modules with different values receive a
  LONG_ or LAT_ prefix. Parameters identical in both are kept as-is.

References:
  Longitudinal/config.py   — longitudinal parameter source
  lateral/config.py        — lateral parameter source
  Belousov et al.          — vehicle EOM (Eq. 3.11a/b/c)
  Li et al. 2019           — Nash GNE framework
  Pustilnik & Borrelli 2025 — non-normalized GNE formulation
"""

import os
import matplotlib
import numpy as np

# =============================================================================
# MATPLOTLIB BACKEND SETUP
# =============================================================================
def _setup_matplotlib():
    global HEADLESS_MODE
    try:
        if os.name == 'nt':
            matplotlib.use('TkAgg')
            HEADLESS_MODE = False
        elif os.environ.get('DISPLAY'):
            matplotlib.use('TkAgg')
            HEADLESS_MODE = False
        else:
            matplotlib.use('Agg')
            HEADLESS_MODE = True
    except Exception:
        matplotlib.use('Agg')
        HEADLESS_MODE = True

_setup_matplotlib()

# =============================================================================
# RESULTS DIRECTORY
# =============================================================================
RESULTS_DIR = "unified_sim_results"

def setup_results_directory():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

setup_results_directory()

# =============================================================================
# SIMULATION TIMING (same in both modules)
# =============================================================================
SIMULATION_DT = 0.01          # 100 Hz — dynamics integration step [s]
NASH_NP       = 20            # Prediction horizon [steps]
NASH_NU       = 10            # Control horizon [steps]
DEFAULT_SIMULATION_TIME = 120.0  # [s]
NOMINAL_VELOCITY        = 120.0 / 3.6   # m/s (~120 km/h) — matches lateral reference

# --- Per-module Nash control rates (CONFLICT: 10 Hz vs 20 Hz) ---
LONG_NASH_CONTROL_DT = 0.1    # Longitudinal Nash: 10 Hz
LAT_NASH_CONTROL_DT  = 0.05   # Lateral Nash:      20 Hz

# --- Derived: simulation steps between each Nash solve ---
LONG_NASH_STEP_INTERVAL = round(LONG_NASH_CONTROL_DT / SIMULATION_DT)  # 10
LAT_NASH_STEP_INTERVAL  = round(LAT_NASH_CONTROL_DT  / SIMULATION_DT)  # 5

# =============================================================================
# VEHICLE PHYSICS (Belousov Eq. 3.11a/b/c)
# Imported from the existing VehicleParameters class (Audi TT 2.0 TFSI)
# to avoid duplicating physical constants.
# =============================================================================
from unified.vehicle.components import VehicleParameters as _VP
_vp = _VP()

# Core vehicle parameters (Belousov Eq. 3.8, 3.11a)
VEHICLE_MASS             = _vp.mass               # [kg]   1305 kg
VEHICLE_WHEEL_INERTIA    = _vp.wheel_inertia       # [kg·m²]
VEHICLE_WHEEL_RADIUS     = _vp.wheel_radius        # [m]
VEHICLE_M_EFF            = _vp.mass + _vp.wheel_inertia / (_vp.wheel_radius ** 2)  # effective mass

# Geometry
VEHICLE_LF = _vp.lf             # Front axle to CoG [m]
VEHICLE_LR = _vp.lr             # Rear axle to CoG [m]
VEHICLE_IZ = _vp.Iz             # Yaw inertia [kg·m²]
VEHICLE_LENGTH = _vp.length     # Vehicle length [m]
VEHICLE_WIDTH  = _vp.width      # Vehicle width  [m]

# Tire cornering stiffness (Rajamani Eq. 13.45)
VEHICLE_CF = _vp.Caf            # Front [N/rad]
VEHICLE_CR = _vp.Car            # Rear  [N/rad]

# Resistances (Belousov Eq. 3.5–3.7)
VEHICLE_CD           = _vp.drag_coefficient
VEHICLE_FRONTAL_AREA = _vp.frontal_area
VEHICLE_AIR_DENSITY  = _vp.air_density
VEHICLE_ROLLING_MU   = _vp.rolling_resistance_coeff
ROAD_GRADE           = _vp.road_grade              # [rad]
ROAD_FRICTION_MU     = 1.0                         # tire-road friction [-]

# Actuator limits
MAX_ACCELERATION     = _vp.max_acceleration        # [m/s²]
MAX_DECELERATION     = _vp.max_deceleration        # [m/s²] (negative)
MAX_STEERING_ANGLE   = np.radians(25.0)            # [rad]  ~25°
MAX_STEERING_RATE    = np.radians(15.0)            # [rad/s] ~15°/s

# =============================================================================
# LANE CONFIGURATION
# =============================================================================
LANE_WIDTH          = 3.5     # [m]
PLATOON_LANE_Y      = 0.0     # Platoon centre lane y-position [m]
HUMAN_INITIAL_LANE_Y = LANE_WIDTH  # Human starts in adjacent lane [m]
LAT_HUMAN_Y_BIAS     = 0.0         # Human steady-state y offset from PLATOON_LANE_Y [m] (0 = no bias)

# =============================================================================
# LONGITUDINAL NASH SOLVER PARAMETERS
# (used to configure ConstrainedLongitudinalNashSolver in the coordinator)
# =============================================================================
LONG_NASH_Q_POS           = 2500.0
LONG_NASH_Q_VEL           = 1500.0
LONG_NASH_R1              = 7500.0   # 30000 * 0.25
LONG_NASH_R2              = 12500.0  # 50000 * 0.25
LONG_NASH_S1              = 10000.0
LONG_NASH_S2              = 10000.0
LONG_NASH_U1_MIN          = MAX_DECELERATION    # [m/s²]
LONG_NASH_U1_MAX          = MAX_ACCELERATION    # [m/s²]
LONG_NASH_U2_MIN          = -4.0
LONG_NASH_U2_MAX          =  3.0
LONG_NASH_DU1_MAX         =  1.5    # jerk [m/s³]
LONG_NASH_DU2_MAX         =  2.0
LONG_NASH_V_MIN           =  0.0
LONG_NASH_V_MAX           = 50.0
LONG_NASH_GAP_MIN         =  7.0    # [m]
LONG_NASH_LAMBDA_LEVELS   = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)

# =============================================================================
# LATERAL NASH SOLVER PARAMETERS
# (used to configure ConstrainedLateralNashSolver in the coordinator)
# =============================================================================
LAT_NASH_Q_Y               =    800.0
LAT_NASH_Q_PSI             =  10000.0
LAT_NASH_Q_Y_TERMINAL_FAC  =     10.0
LAT_NASH_Q_PSI_TERMINAL_FAC=      4.0
LAT_NASH_R1                =  50000.0
LAT_NASH_R2                =  50000.0
LAT_NASH_S1                = 200000.0
LAT_NASH_S2                = 200000.0
LAT_NASH_DELTA_MIN         = -MAX_STEERING_ANGLE   # [rad]
LAT_NASH_DELTA_MAX         =  MAX_STEERING_ANGLE
LAT_NASH_Y_MAX             =  5.0     # [m]
LAT_NASH_PSI_MAX           =  0.5     # [rad]
LAT_NASH_LAMBDA_LEVELS     = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

# Shared OSQP settings
NASH_SOLVER_BACKEND  = 'OSQP'
NASH_WARM_START      = True
NASH_VERBOSE         = False
NASH_POLISH          = False
NASH_REGULARIZATION  = 1e-5
NASH_RISK_GAMMA      = 0.1   # γ in E[J]+γ·Var[J]; risk sensitivity for SE-kernel GP uncertainty
LONG_NASH_MAX_ITER   = 1000
LONG_NASH_EPS_ABS    = 1e-4
LONG_NASH_EPS_REL    = 1e-4
LAT_NASH_MAX_ITER    = 600
LAT_NASH_EPS_ABS     = 1e-3
LAT_NASH_EPS_REL     = 1e-3

# =============================================================================
# LONGITUDINAL SAFETY FIELD (Wang et al. 2015/2016)
# =============================================================================
LONG_SAFETY_MIN_SAFE_DISTANCE       =  7.0    # [m]
LONG_SAFETY_EMERGENCY_BRAKE_DIST    =  8.0    # [m]
LONG_SAFETY_MAX_REPULSIVE_FORCE     = 800.0   # [N]
LONG_SAFETY_FILTER_ALPHA            =  0.2

# =============================================================================
# LATERAL SAFETY FIELD / DSF (Li 2019, Wang 2015/2016)
# =============================================================================
LAT_DSF_TS           = 2.0     # TTC safety margin [s]
LAT_DSF_TAU          = 2.0     # Lateral safety radius [m]
LAT_DSF_A_MIN        = 2.0     # Min longitudinal semi-axis [m]
LAT_DSF_G            = 6e-3    # Field scaling constant
LAT_DSF_K1           = 0.005
LAT_DSF_K2           = 0.005
LAT_DSF_DR           = 0.5     # Driver risk factor
LAT_SAFETY_MAX_FORCE = 2000.0  # [N]
LAT_SAFETY_FILTER_ALPHA = 0.3

# =============================================================================
# AUTHORITY ALLOCATOR — LONGITUDINAL (sigmoid on risk force)
# =============================================================================
LONG_AUTHORITY_LAMBDA_MIN      =   0.1   # Human dominant (safe)
LONG_AUTHORITY_LAMBDA_MAX      = 100.0   # System dominant (emergency)
LONG_AUTHORITY_FORCE_MIDPOINT  = 400.0   # Sigmoid centre [N]
LONG_AUTHORITY_K_STEEPNESS     =   0.015
LONG_AUTHORITY_ALPHA_BASE      =   0.05  # Slow smoothing
LONG_AUTHORITY_ALPHA_FAST      =   0.12  # Fast smoothing

# =============================================================================
# AUTHORITY ALLOCATOR — LATERAL (Swain & Rath 2023, Eq. 15)
# =============================================================================
LAT_AUTHORITY_LAMBDA_MIN      =  0.1
LAT_AUTHORITY_LAMBDA_MAX      = 10.0
LAT_AUTHORITY_FORCE_MIDPOINT  = 200.0   # Sigmoid centre [N]
LAT_AUTHORITY_K_STEEPNESS     =  0.025
LAT_AUTHORITY_ALPHA_BASE      =  0.02
LAT_AUTHORITY_ALPHA_FAST      =  0.08
LAT_AUTHORITY_ALPHA_FAST_THR  =  1.5   # |y_error| > this → use alpha_fast [m]
LAT_AUTHORITY_ALPHA_BASE_THR  =  0.5   # |y_error| < this → use alpha_base [m]
LAT_AUTHORITY_SIGMOID_M1      =  2.0   # Slope (Swain & Rath Eq. 15)
LAT_AUTHORITY_SIGMOID_M2      =  0.5   # Centre shift

# =============================================================================
# PLATOON CONTROL
# =============================================================================
PLATOON_TIME_GAP            = 1.5    # Desired time headway [s]
PLATOON_STANDSTILL_DISTANCE = 2.0           # Minimum gap at standstill [m] — matches IDM s0
PLATOON_TARGET_VELOCITY     = NOMINAL_VELOCITY
PLATOON_VEHICLE_LENGTH      = _vp.length  # [m]

# ── Linearization rebuild thresholds (physics-derived) ───────────────────────
# Longitudinal: dominant nonlinear term is aerodynamic drag → A_c[1,1] = -Cd_aero·vx
#   Rebuild when velocity prediction error over horizon exceeds ε_v = 0.2 m/s:
#   Δvx_thr = ε_v / (Cd_aero · dt_long · Np · V_MAX)
_Cd_aero = VEHICLE_AIR_DENSITY * VEHICLE_CD * VEHICLE_FRONTAL_AREA / VEHICLE_M_EFF
LONG_NASH_REBUILD_DVX = 0.2 / (_Cd_aero * LONG_NASH_CONTROL_DT * NASH_NP * max(LONG_NASH_V_MAX, 1.0))

# Lateral: dominant changing term is Coriolis → A_c[1,3] ≈ -vx → dA_c/dvx ≈ -1
#   Rebuild when lateral velocity prediction error exceeds ε_vy = 0.20 m/s:
#   ψ̇_max ≈ μ·g / V_target (max yaw rate at cruise)
#   Δvx_thr = ε_vy / (dt_lat · Np · ψ̇_max)
_psi_dot_max = _vp.tire_friction_coeff * _vp.gravity / max(PLATOON_TARGET_VELOCITY, 1.0)
LAT_NASH_REBUILD_DVX = 0.20 / (LAT_NASH_CONTROL_DT * NASH_NP * max(_psi_dot_max, 0.01))

# Platoon control dynamics
JERK_LIMIT       = 2.0   # Maximum da/dt [m/s³]
FREE_ROAD_DELTA  = 4     # IDM free-road exponent δ

# Rajamani gains (Longitudinal Ch. 6)
RAJAMANI_H   = 1.5
RAJAMANI_TAU = 0.1
RAJAMANI_K1  = -0.12
RAJAMANI_K5  =  0.1
RAJAMANI_K2  = -RAJAMANI_K1 - RAJAMANI_H * RAJAMANI_K1 * RAJAMANI_K5
RAJAMANI_K3  =  1.0 / RAJAMANI_H - RAJAMANI_K1 * RAJAMANI_K5
RAJAMANI_K4  =  RAJAMANI_K5 / RAJAMANI_H

# =============================================================================
# MOBIL / IDM (Kesting, Treiber & Helbing 2007)
# =============================================================================
MOBIL_IDM_V0    = 33.3
MOBIL_IDM_T     =  1.2
MOBIL_IDM_A_MAX =  1.5
MOBIL_IDM_B     =  2.0
MOBIL_IDM_S0    =  2.0
MOBIL_IDM_DELTA =  4.0
MOBIL_IDM_L     =  4.0
MOBIL_P         =  0.5
MOBIL_B_SAFE    =  4.0
MOBIL_A_TH      =  0.1
MOBIL_A_BIAS    =  0.3   # Right lane bias [m/s²] — matches lateral reference
MOBIL_MIN_GAP   =  8.0  # [m]

# =============================================================================
# UNIFIED COORDINATOR — PHASE THRESHOLDS
# =============================================================================
# APPROACH → MERGE: MOBIL is checked once the ego vehicle is closer than this
APPROACH_MOBIL_CHECK_DISTANCE = 300.0   # [m] to nearest platoon vehicle

# MERGE → FOLLOWING: all three conditions must hold simultaneously
MERGE_COMPLETE_Y_ERROR     = 0.3    # |y - platoon_lane_y| < 0.3 m
MERGE_COMPLETE_PSI_ERROR   = 0.05   # |psi| < ~3 deg [rad]
MERGE_COMPLETE_GAP_RATIO   = 0.20   # |gap - desired_gap| / desired_gap < 20%

# Stability hold time before FOLLOWING transition [s] — matches both reference modules
PHASE_TRANSITION_HOLD_TIME = 5.0

# GAP_SEARCH phase duration before transitioning to MERGE/LANE_CHANGE [s]
GAP_SEARCH_DURATION = 0.5    # matches lateral reference (lateral/config.py)

# Minimum time spent in LANE_CHANGE phase before LANE_KEEPING is allowed [s]
LANE_CHANGE_MIN_TIME = 6.0   # matches lateral reference (lateral/config.py)

# Longitudinal authority: max admissible gap error and velocity error for Swain & Rath sigmoid
AUTHORITY_GAP_ERROR_MAX = 5.0    # [m]  — matches Longitudinal/config.py
AUTHORITY_VEL_ERROR_MAX = 1.5    # [m/s] — matches Longitudinal/config.py

# Longitudinal phase detection: FOLLOWING entry threshold (gap error as fraction of desired gap)
FOLLOWING_GAP_ERROR_FACTOR = 0.15   # matches Longitudinal/config.py

# DSF virtual mass parameters (Wang 2016, Eq. 17): M = m*(speed_coeff*v^speed_exp + speed_offset)
DSF_SPEED_COEFF    = 1.566e-14  # matches lateral/config.py
DSF_SPEED_EXPONENT = 6.687      # matches lateral/config.py
DSF_SPEED_OFFSET   = 0.3345     # matches lateral/config.py
DSF_VEHICLE_MASS   = 1304.0     # [kg] base vehicle mass for virtual mass computation
DSF_EPSILON        = 1e-3       # minimum elliptic distance [m]

# =============================================================================
# DRIVER PROFILES
# =============================================================================
DRIVER_PARAMS = {
    'cautious': {
        'velocity_offset':       -5.0,   # re NOMINAL_VELOCITY [m/s]
        'x_error_offset':        20.0,   # initial X offset from platoon rear [m]
        'tlc':                    6.0,   # lane-change duration [s]
        'max_heading_deg':        3.0,
        'system_tlc_multiplier':  1.5,
        'mobil_p':                0.8,
        'mobil_a_th':             0.2,
        'plan_time_headway':      2.5,   # IDM planning time gap [s] — cautious keeps more distance
        'plan_decel':             3.0,   # IDM comfortable decel [m/s²] — softer braking
    },
    'normal': {
        'velocity_offset':        0.0,
        'x_error_offset':         0.0,
        'tlc':                    4.5,
        'max_heading_deg':        4.0,
        'system_tlc_multiplier':  1.5,
        'mobil_p':                0.5,
        'mobil_a_th':             0.1,
        'plan_time_headway':      2.0,   # IDM planning time gap [s]
        'plan_decel':             4.0,   # IDM comfortable decel [m/s²]
    },
    'aggressive': {
        'velocity_offset':        5.0,
        'x_error_offset':        -5.0,   # was -20: caused ego to start inside/ahead of platoon
        'tlc':                    3.0,
        'max_heading_deg':        6.0,
        'system_tlc_multiplier':  1.5,
        'mobil_p':                0.2,
        'mobil_a_th':             0.05,
        'plan_time_headway':      1.3,   # IDM planning time gap [s] — aggressive follows closely
        'plan_decel':             6.0,   # IDM comfortable decel [m/s²] — accepts harder braking
    },
}

# =============================================================================
# MA-IDM (Memory-Augmented IDM, Zhang & Sun 2024)
# =============================================================================
MA_IDM_ENABLED = True          # Toggle GP noise on/off globally

# Longitudinal GP hyperparameters per driver type — from Table I, hierarchical MA-IDM
# σ_k: noise std [m/s²], ell: memory lengthscale [s]
MA_IDM_PARAMS = {
    'cautious':   {'sigma_k': 0.172, 'ell': 1.55},
    'normal':     {'sigma_k': 0.202, 'ell': 1.435},
    'aggressive': {'sigma_k': 0.240, 'ell': 1.30},
}

# Lateral GP hyperparameters per driver type
# σ_k: steering noise std [rad], ell: memory lengthscale [s]
# (shorter than longitudinal — steering reacts faster)
MA_IDM_LAT_PARAMS = {
    'cautious':   {'sigma_k': 0.012, 'ell': 0.80},
    'normal':     {'sigma_k': 0.018, 'ell': 0.65},
    'aggressive': {'sigma_k': 0.025, 'ell': 0.50},
}

# Future full-GP upgrade hooks (unused by AR(1)):
MA_IDM_WINDOW_SIZE   = 50    # history window = 5s @ 10Hz
MA_IDM_OBS_NOISE     = 0.01  # GP observation noise σ_obs² [(m/s²)²] — longitudinal
MA_IDM_LAT_OBS_NOISE = 1e-5  # GP observation noise σ_obs² [rad²]    — lateral
# Note: lateral sigma_k² ≈ 1.4e-4..6.3e-4 rad²; obs_noise=1e-5 → SNR=14..63
# Previous value (0.01) had SNR<1 → GP sampled i.i.d. at 100 Hz → steering oscillations

# =============================================================================
# LOWER-LEVEL CONTROLLER  (Longitudinal/control/lower_level_controller.py)
# =============================================================================
LOWER_CTRL_KP                  = 0.4     # Proportional gain on accel error [-]
LOWER_CTRL_KI                  = 0.08    # Integral gain on accel error [-]
LOWER_CTRL_INTEGRATOR_LIMIT    = 2.0     # Anti-windup saturation [m/s²·s]
LOWER_CTRL_THROTTLE_SMOOTHING  = 0.08    # Throttle EMA filter coefficient [-]
LOWER_CTRL_BRAKE_SMOOTHING     = 0.10    # Brake EMA filter coefficient [-]
LOWER_CTRL_DEADBAND            = 50.0    # Force deadband [N]
LOWER_CTRL_ACCEL_FILTER_ALPHA  = 0.06    # Accel output filter coefficient [-]
LOWER_CTRL_GEAR_SHIFT_HOLD     = True    # Freeze actuator output during gear shifts
LOWER_CTRL_AIR_DENSITY         = 1.225   # [kg/m³]
LOWER_CTRL_ROLLING_COEFF       = 0.012   # Rolling resistance coefficient [-]
ENGINE_BRAKING_BASE_TORQUE     = 60.0    # Base retarding torque at 2000 RPM [Nm]
ENGINE_BRAKING_MAX_TORQUE      = 120.0   # Maximum retarding torque cap [Nm]

# =============================================================================
# PLOTTING
# =============================================================================
VEHICLE_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
