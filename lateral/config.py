"""
Global configuration for the Lateral Control simulation system.
VERSION 2.1 - Full parameter centralization

Based on: Pustilnik & Borrelli 2025 (GNE lateral merging),
          Stanley 2006 (heading + cross-track error controller),
          Kesting, Treiber & Helbing 2007 (MOBIL lane change model).

All hardcoded parameters from nash_solver, safety_field, authority_allocator,
system_reference_generator, human_reference_generator, platoon_control,
mobil_lane_change, and human_driver are extracted here.

This file is the SINGLE SOURCE OF TRUTH for all lateral parameters.
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# =============================================================================
# MATPLOTLIB BACKEND SETUP
# =============================================================================
def setup_matplotlib():
    """Configure matplotlib backend with fallback options."""
    global HEADLESS_MODE

    try:
        if os.name == 'nt':  # Windows
            matplotlib.use('TkAgg')
            HEADLESS_MODE = False
            print("✅ Using TkAgg backend (interactive mode)")
        elif os.environ.get('DISPLAY'):  # Linux with display
            matplotlib.use('TkAgg')
            HEADLESS_MODE = False
            print("✅ Using TkAgg backend (interactive mode)")
        else:
            matplotlib.use('Agg')
            HEADLESS_MODE = True
            print("⚠️ Using Agg backend (headless mode)")
    except Exception as e:
        matplotlib.use('Agg')
        HEADLESS_MODE = True
        print(f"⚠️ Using Agg backend (fallback): {e}")

setup_matplotlib()

# =============================================================================
# RESULTS & FILE SYSTEM
# =============================================================================
RESULTS_DIR = "lateral_sim_results_v2"

def setup_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

setup_results_directory()

# =============================================================================
# SIMULATION CONSTANTS
# =============================================================================
SIMULATION_DT = 0.01          # 100 Hz - vehicle dynamics integration step [s]
NASH_CONTROL_DT = 0.05        # 20 Hz  - Nash MPC control step [s]
NASH_NP = 20                  # Prediction horizon [steps]
NASH_NU = 10                  # Control horizon [steps]
DEFAULT_SIMULATION_TIME = 120.0  # Default simulation time [s]
NOMINAL_VELOCITY = 20.0       # Nominal longitudinal velocity [m/s]

# =============================================================================
# LANE CONFIGURATION
# =============================================================================
LANE_WIDTH = 3.5              # Lane width [m]
PLATOON_LANE_Y = 0.0          # Platoon travels at y=0 [m]
HUMAN_INITIAL_LANE_Y = 3.5    # Human starts at y=3.5m [m]

# Vehicle colors for plotting
VEHICLE_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# =============================================================================
# STANLEY CONTROLLER PARAMETERS (Stanley 2006)
# Source: control/human_driver.py - StanleyController
# =============================================================================
# Cross-track error gain — higher = more aggressive lane tracking
STANLEY_K_E_CAUTIOUS   = 0.003
STANLEY_K_E_NORMAL     = 0.005
STANLEY_K_E_AGGRESSIVE = 0.008

# Heading error gain — higher = faster heading correction
STANLEY_K_PSI_CAUTIOUS   = 0.3
STANLEY_K_PSI_NORMAL     = 0.5
STANLEY_K_PSI_AGGRESSIVE = 0.7

# Velocity softening term (prevents division by zero at low speed)
STANLEY_K_SOFT = 1.0

# Low-pass filter on steering output
STANLEY_FILTER_ALPHA = 0.2    # 1st-order EMA on steering command [-]

# Per-driver-type parameter dict (additive — flat constants above remain canonical)
STANLEY_DRIVER_PARAMS = {
    'cautious': {
        'k_e':   STANLEY_K_E_CAUTIOUS,
        'k_psi': STANLEY_K_PSI_CAUTIOUS,
    },
    'normal': {
        'k_e':   STANLEY_K_E_NORMAL,
        'k_psi': STANLEY_K_PSI_NORMAL,
    },
    'aggressive': {
        'k_e':   STANLEY_K_E_AGGRESSIVE,
        'k_psi': STANLEY_K_PSI_AGGRESSIVE,
    },
}

# =============================================================================
# NASH SOLVER PARAMETERS (Pustilnik & Borrelli 2025)
# Source: nash_solver/lateral_constrained_nash_solver.py - ConstrainedLateralNashParams
# =============================================================================
# Output tracking weights [y, psi]
# CRITICAL: Q_psi >> Q_y — stabilise heading before lateral position
NASH_Q_Y   = 800.0       # Weight on lateral position error (50× increase for stability)
NASH_Q_PSI = 10000.0     # Weight on heading angle error (HIGH — heading stability first)

# Terminal weight multipliers (applied to stage weights for the last step)
NASH_Q_Y_TERMINAL_FACTOR   = 10.0   # Q_y_terminal = NASH_Q_Y_TERMINAL_FACTOR × NASH_Q_Y
NASH_Q_PSI_TERMINAL_FACTOR  = 4.0   # Q_psi_terminal = NASH_Q_PSI_TERMINAL_FACTOR × NASH_Q_PSI

# Own control effort weights (R) — high values → smooth, gentle control
NASH_R1 = 1000000.0      # System's cost on its own control effort
NASH_R2 = 1000000.0      # Human's cost on its own control effort

# Cross-coupling weights (S) — THE KEY for coupled Nash game
NASH_S1 = 200000.0       # System cares about human effort
NASH_S2 = 200000.0       # Human cares about system effort

# State constraints
NASH_Y_MAX   = 5.0        # Maximum lateral deviation [m]
NASH_PSI_MAX = 0.5        # Maximum heading angle [rad]

# Lambda levels for pre-computation (Pustilnik scaling)
NASH_LAMBDA_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

# Solver settings
NASH_SOLVER_BACKEND = 'OSQP'
NASH_WARM_START     = True
NASH_VERBOSE        = False
NASH_MAX_ITER       = 200
NASH_EPS_ABS        = 1e-3
NASH_EPS_REL        = 1e-3
NASH_POLISH         = False
NASH_REGULARIZATION = 1e-5

# =============================================================================
# LATERAL SAFETY FIELD PARAMETERS
# Source: nash_solver/lateral_safety_field.py - LateralSafetyFieldParams
# =============================================================================
# --- Collision detection thresholds ---
SAFETY_COLLISION_LATERAL_THRESHOLD      = 2.0    # Lateral distance for "danger" [m]
SAFETY_COLLISION_LONGITUDINAL_THRESHOLD = 10.0   # Longitudinal overlap for "danger" [m]
SAFETY_SAFE_LATERAL_DISTANCE            = 3.0    # No force if lateral distance > this [m]

# --- Obstacle force parameters ---
SAFETY_OBSTACLE_FORCE_GAIN  = 150.0    # Obstacle repulsive force gain [N]
SAFETY_OBSTACLE_FORCE_SCALE = 5.0      # tanh saturation scale for obstacle force

# --- Road boundary parameters ---
SAFETY_ROAD_HALF_WIDTH      = 7.0      # Half road width [m] (total 14m = 4 lanes)
SAFETY_BOUNDARY_FORCE_GAIN  = 150.0    # Boundary repulsive force gain [N]
SAFETY_BOUNDARY_FORCE_SCALE = 2.0      # tanh saturation scale for boundary force
SAFETY_BOUNDARY_PROXIMITY   = 3.0      # Activate boundary force when closer than this [m]

# --- Distance decay ---
SAFETY_DISTANCE_DECAY_FACTOR = 20.0    # Exponential decay constant [m]
SAFETY_EPSILON               = 0.1     # Numerical stability offset

# --- Force limits ---
SAFETY_MAX_FORCE = 500.0               # Maximum total lateral safety force [N]

# --- Direction smoothing ---
# tanh direction: at dy=1m: tanh(2)≈0.96 ≈ sign; at dy=0: tanh(0)=0 (no flip)
SAFETY_DIRECTION_SMOOTH_SIGMA = 0.5    # tanh width for smooth direction [m]

# --- Force output filter ---
SAFETY_FILTER_ALPHA = 0.3              # Low-pass EMA on safety force output [-]

# =============================================================================
# AUTHORITY ALLOCATOR PARAMETERS
# Source: nash_solver/lateral_authority_allocator.py - LateralAuthorityAllocator
# Aligned with longitudinal V5.2
# =============================================================================
# Sigmoid parameters for SAFETY (risk-based)
AUTHORITY_LAMBDA_MIN      = 0.1    # Human dominant when safe
AUTHORITY_LAMBDA_MAX      = 10.0   # System dominant when dangerous
AUTHORITY_FORCE_MIDPOINT  = 150.0  # Sigmoid centre point [N]
AUTHORITY_K_STEEPNESS     = 0.025  # Sigmoid slope (sharper: human ~75% at rest vs 64% at k=0.02)

# Smoothing parameters
AUTHORITY_ALPHA_BASE = 0.02        # Slow smoothing (stability)
AUTHORITY_ALPHA_FAST = 0.08        # Fast response (large errors)

# Hysteresis thresholds for PERFORMANCE authority (based on |y_error|)
AUTHORITY_ENTER_THRESHOLD         = 1.0  # Enter performance mode when |y_error| > 1.0 m
AUTHORITY_EXIT_THRESHOLD          = 0.3  # Exit performance mode when |y_error| < 0.3 m
AUTHORITY_LAMBDA_PERFORMANCE_MAX  = 5.0  # Performance authority upper bound

# =============================================================================
# REFERENCE GENERATOR PARAMETERS (Pustilnik & Borrelli 2025)
# Source: nash_solver/system_reference_generator.py, human_reference_generator.py
# =============================================================================
# --- Lane change durations (human driver preference) ---
# T_lc determines how long the planned lane-change trajectory spans.
# Cautious → slower, aggressive → faster.
REFGEN_BASE_TLC = {
    'cautious':  6.0,    # [s]
    'normal':    4.5,    # [s]
    'aggressive': 3.0,   # [s]
}

# --- System T_lc multiplier ---
# System plans a LONGER reference to reduce QP reference rate-of-change (≈ chattering).
# Aggressive gets a higher multiplier because 3.0s × 1.5 = 4.5s was too fast for 20Hz Nash.
REFGEN_SYSTEM_TLC_MULTIPLIER = {
    'cautious':  1.5,    # cautious: 6.0s × 1.5 = 9.0s
    'normal':    1.5,    # normal:   4.5s × 1.5 = 6.75s
    'aggressive': 2.0,   # aggressive: 3.0s × 2.0 = 6.0s (was 4.5s — too fast for QP)
}

# --- System max heading constraint (DynamicTrajectoryParams) ---
REFGEN_SYSTEM_MAX_HEADING_DEG = 8.0    # Max heading angle for system trajectory [deg]

# --- Human max heading angles per driver type ---
REFGEN_HUMAN_MAX_HEADING_DEG = {
    'cautious':  3.0,    # [deg]
    'normal':    4.0,    # [deg]
    'aggressive': 6.0,   # [deg]
}

# --- Settling times for LANE_KEEPING soft transition ---
# When entering LANE_KEEPING, a cubic polynomial settles from current (y, ψ) to (y_target, 0)
# over T_settle seconds. Prevents reference discontinuity = heading saturation.
REFGEN_SYSTEM_SETTLE_TIME = {
    'cautious':  5.0,    # [s] — slow, gentle settling
    'normal':    3.0,    # [s]
    'aggressive': 2.0,   # [s] — quick settling
}

REFGEN_HUMAN_SETTLE_TIME = {
    'cautious':  4.0,    # [s]
    'normal':    3.0,    # [s]
    'aggressive': 2.0,   # [s]
}

# --- Minimum T_lc bounds (from heading constraint derivation) ---
REFGEN_MIN_TLC_QUINTIC = 3.0    # 5th-order polynomial minimum [s] (system)
REFGEN_MIN_TLC_CUBIC   = 2.0    # 3rd-order polynomial minimum [s] (human)
REFGEN_MIN_TLC_FREE_ROAD_SYSTEM = 5.0   # Free-road fallback for system [s]
REFGEN_MIN_TLC_FREE_ROAD_HUMAN  = 8.0   # Free-road fallback for human [s]

# =============================================================================
# PHASE DETECTION PARAMETERS
# Source: nash_solver/lateral_safety_field.py - LateralSafetyField
# =============================================================================
# Entry to FOLLOWING (ALL must be true for PHASE_TRANSITION_TIME seconds)
FOLLOWING_Y_ERROR_FACTOR      = 0.15   # |y_error| < 15% × lane_width
FOLLOWING_PSI_ERROR_THRESHOLD = 0.05   # |psi| < ~3 degrees [rad]
FOLLOWING_Y_DOT_THRESHOLD     = 0.3    # |y_dot| < 0.3 m/s

# Exit from FOLLOWING (ANY triggers return to MERGING — hysteresis)
MERGING_Y_ERROR_FACTOR        = 0.25   # |y_error| > 25% × lane_width
MERGING_PSI_ERROR_THRESHOLD   = 0.10   # |psi| > ~6 degrees [rad]

# Phase transition confirmation time
PHASE_TRANSITION_TIME = 5.0            # [s] stability required for FOLLOWING entry

# =============================================================================
# PHASE DURATION THRESHOLDS
# Source: nash_solver/lateral_safety_field.py - _update_phase()
# =============================================================================
GAP_SEARCH_DURATION         = 0.5    # [s] wait in GAP_SEARCH before LANE_CHANGE
LANE_CHANGE_MIN_TIME        = 6.0    # [s] min time in LANE_CHANGE before LANE_KEEPING allowed
LANE_CHANGE_Y_ERROR_FACTOR  = 0.05   # |y_error| < 5% × lane_width to enter LANE_KEEPING
LANE_CHANGE_Y_DOT_THRESHOLD = 0.10   # |y_dot| < 0.10 m/s — nearly stopped laterally

# =============================================================================
# PLATOON CONTROL PARAMETERS
# Source: control/platoon_control.py - PlatoonParams, PlatoonManager
# =============================================================================
PLATOON_TIME_GAP             = 1.5    # Desired time headway [s]
PLATOON_STANDSTILL_DISTANCE  = 5.0   # Minimum standstill gap [m]
PLATOON_TARGET_VELOCITY      = NOMINAL_VELOCITY  # [m/s]
PLATOON_VEHICLE_LENGTH       = 4.5   # Vehicle length for gap calculation [m]
PLATOON_MIN_MERGE_GAP        = 15.0  # Minimum lateral gap before merge allowed [m]

# Longitudinal control gains (simple PD follower)
PLATOON_LEADER_VEL_GAIN    = 0.5    # Leader: a = gain × (v_target - v) [1/s]
PLATOON_FOLLOWER_DIST_GAIN = 0.3    # Follower: distance error gain [1/s²]
PLATOON_FOLLOWER_VEL_GAIN  = 0.8    # Follower: velocity error gain [1/s]
PLATOON_ACCEL_MAX          =  2.0   # Platoon max acceleration [m/s²]
PLATOON_ACCEL_MIN          = -3.0   # Platoon max deceleration [m/s²]

# =============================================================================
# MOBIL / IDM PARAMETERS (Kesting, Treiber & Helbing 2007)
# Source: control/mobil_lane_change.py - IDMParams, MOBILParams
# =============================================================================
# --- IDM (Intelligent Driver Model) base parameters ---
MOBIL_IDM_V0    = 33.3   # Desired velocity [m/s] (120 km/h)
MOBIL_IDM_T     = 1.2    # Safe time headway [s]
MOBIL_IDM_A_MAX = 1.5    # Maximum acceleration [m/s²]
MOBIL_IDM_B     = 2.0    # Comfortable deceleration [m/s²]
MOBIL_IDM_S0    = 2.0    # Minimum spacing [m]
MOBIL_IDM_DELTA = 4.0    # Acceleration exponent
MOBIL_IDM_L     = 4.0    # Vehicle length for IDM gap calculation [m]

# --- MOBIL model base parameters ---
MOBIL_P        = 0.5    # Politeness factor [0=egoistic, 1=altruistic]
MOBIL_B_SAFE   = 4.0    # Maximum safe deceleration for new follower [m/s²]
MOBIL_A_TH     = 0.1    # Lane change incentive threshold [m/s²]
MOBIL_A_BIAS   = 0.3    # Right lane bias [m/s²]
MOBIL_MIN_GAP  = 8.0    # Minimum front/rear gap for middle merge [m]

# --- Driver-type MOBIL overrides ---
MOBIL_CAUTIOUS_P     = 0.8    # Very polite — considers others
MOBIL_CAUTIOUS_A_TH  = 0.2    # Conservative threshold
MOBIL_AGGRESSIVE_P   = 0.2    # Less polite — more egoistic
MOBIL_AGGRESSIVE_A_TH = 0.05  # Easier to trigger lane change

# =============================================================================
# EXPORT
# =============================================================================
__all__ = [
    # System
    'HEADLESS_MODE', 'RESULTS_DIR', 'SIMULATION_DT', 'NASH_CONTROL_DT',
    'NASH_NP', 'NASH_NU', 'DEFAULT_SIMULATION_TIME', 'NOMINAL_VELOCITY',

    # Lane / plotting
    'LANE_WIDTH', 'PLATOON_LANE_Y', 'HUMAN_INITIAL_LANE_Y', 'VEHICLE_COLORS',

    # Stanley controller
    'STANLEY_K_E_CAUTIOUS', 'STANLEY_K_E_NORMAL', 'STANLEY_K_E_AGGRESSIVE',
    'STANLEY_K_PSI_CAUTIOUS', 'STANLEY_K_PSI_NORMAL', 'STANLEY_K_PSI_AGGRESSIVE',
    'STANLEY_K_SOFT', 'STANLEY_FILTER_ALPHA', 'STANLEY_DRIVER_PARAMS',

    # Nash solver
    'NASH_Q_Y', 'NASH_Q_PSI',
    'NASH_Q_Y_TERMINAL_FACTOR', 'NASH_Q_PSI_TERMINAL_FACTOR',
    'NASH_R1', 'NASH_R2', 'NASH_S1', 'NASH_S2',
    'NASH_Y_MAX', 'NASH_PSI_MAX', 'NASH_LAMBDA_LEVELS',
    'NASH_SOLVER_BACKEND', 'NASH_WARM_START', 'NASH_VERBOSE',
    'NASH_MAX_ITER', 'NASH_EPS_ABS', 'NASH_EPS_REL',
    'NASH_POLISH', 'NASH_REGULARIZATION',

    # Lateral safety field
    'SAFETY_COLLISION_LATERAL_THRESHOLD', 'SAFETY_COLLISION_LONGITUDINAL_THRESHOLD',
    'SAFETY_SAFE_LATERAL_DISTANCE',
    'SAFETY_OBSTACLE_FORCE_GAIN', 'SAFETY_OBSTACLE_FORCE_SCALE',
    'SAFETY_ROAD_HALF_WIDTH', 'SAFETY_BOUNDARY_FORCE_GAIN', 'SAFETY_BOUNDARY_FORCE_SCALE',
    'SAFETY_BOUNDARY_PROXIMITY', 'SAFETY_DISTANCE_DECAY_FACTOR', 'SAFETY_EPSILON',
    'SAFETY_MAX_FORCE', 'SAFETY_DIRECTION_SMOOTH_SIGMA', 'SAFETY_FILTER_ALPHA',

    # Authority allocator
    'AUTHORITY_LAMBDA_MIN', 'AUTHORITY_LAMBDA_MAX',
    'AUTHORITY_FORCE_MIDPOINT', 'AUTHORITY_K_STEEPNESS',
    'AUTHORITY_ALPHA_BASE', 'AUTHORITY_ALPHA_FAST',
    'AUTHORITY_ENTER_THRESHOLD', 'AUTHORITY_EXIT_THRESHOLD',
    'AUTHORITY_LAMBDA_PERFORMANCE_MAX',

    # Reference generators
    'REFGEN_BASE_TLC', 'REFGEN_SYSTEM_TLC_MULTIPLIER',
    'REFGEN_SYSTEM_MAX_HEADING_DEG', 'REFGEN_HUMAN_MAX_HEADING_DEG',
    'REFGEN_SYSTEM_SETTLE_TIME', 'REFGEN_HUMAN_SETTLE_TIME',
    'REFGEN_MIN_TLC_QUINTIC', 'REFGEN_MIN_TLC_CUBIC',
    'REFGEN_MIN_TLC_FREE_ROAD_SYSTEM', 'REFGEN_MIN_TLC_FREE_ROAD_HUMAN',

    # Phase detection
    'FOLLOWING_Y_ERROR_FACTOR', 'FOLLOWING_PSI_ERROR_THRESHOLD', 'FOLLOWING_Y_DOT_THRESHOLD',
    'MERGING_Y_ERROR_FACTOR', 'MERGING_PSI_ERROR_THRESHOLD',
    'PHASE_TRANSITION_TIME',

    # Phase duration thresholds
    'GAP_SEARCH_DURATION', 'LANE_CHANGE_MIN_TIME',
    'LANE_CHANGE_Y_ERROR_FACTOR', 'LANE_CHANGE_Y_DOT_THRESHOLD',

    # Platoon control
    'PLATOON_TIME_GAP', 'PLATOON_STANDSTILL_DISTANCE', 'PLATOON_TARGET_VELOCITY',
    'PLATOON_VEHICLE_LENGTH', 'PLATOON_MIN_MERGE_GAP',
    'PLATOON_LEADER_VEL_GAIN', 'PLATOON_FOLLOWER_DIST_GAIN', 'PLATOON_FOLLOWER_VEL_GAIN',
    'PLATOON_ACCEL_MAX', 'PLATOON_ACCEL_MIN',

    # MOBIL / IDM
    'MOBIL_IDM_V0', 'MOBIL_IDM_T', 'MOBIL_IDM_A_MAX', 'MOBIL_IDM_B',
    'MOBIL_IDM_S0', 'MOBIL_IDM_DELTA', 'MOBIL_IDM_L',
    'MOBIL_P', 'MOBIL_B_SAFE', 'MOBIL_A_TH', 'MOBIL_A_BIAS', 'MOBIL_MIN_GAP',
    'MOBIL_CAUTIOUS_P', 'MOBIL_CAUTIOUS_A_TH',
    'MOBIL_AGGRESSIVE_P', 'MOBIL_AGGRESSIVE_A_TH',
]
