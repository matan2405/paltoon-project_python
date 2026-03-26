"""
Global configuration for the Lateral Control simulation system.

Research basis and scope:
- Pustilnik and Borrelli (2025): non-normalized GNE/Nash parameters.
- Stanley controller line: heading and cross-track steering gains.
- MOBIL/IDM literature: lane-change incentive and safety parameters.

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
NOMINAL_VELOCITY = 33.0       # Nominal longitudinal velocity [m/s]

# =============================================================================
# DRIVER TYPE PARAMETERS — single source of truth (mirrors Longitudinal/config.py)
# Inspired by NASH_DRIVER_PARAMS in the longitudinal system.
# Every per-driver constant below is DERIVED from this dict.
# =============================================================================
DRIVER_PARAMS = {
    'cautious': {
        # Longitudinal velocity
        'velocity_offset':        -5.0,   # [m/s] re. NOMINAL_VELOCITY  → 28 m/s
        'x_error_offset':       20,    # [m] positive → more conservative (larger initial error)
        # Stanley controller (Stanley 2006)
        'stanley_k_e':            0.003,  # Cross-track error gain
        'stanley_k_psi':          0.3,    # Heading error gain
        # Human reference trajectory
        'tlc':                    6.0,    # Lane-change duration [s]
        'max_heading_deg':        3.0,    # Max heading angle [deg]
        'human_settle_time':      20.0,   # LANE_KEEPING settling time [s]
        # System reference trajectory
        'system_tlc_multiplier':  1.5,    # System T_lc = tlc × multiplier → 9.0 s
        'system_settle_time':     20.0,   # System settling time [s]
        # MOBIL lane-change model
        'mobil_p':                0.8,    # Politeness factor (very polite)
        'mobil_a_th':             0.2,    # Lane-change incentive threshold [m/s²]
    },
    'normal': {
        'velocity_offset':        0.0,    # → 33 m/s (matches platoon)
        'x_error_offset':       0.0,    # [m] positive → more conservative (larger initial error)
        'stanley_k_e':            0.005,
        'stanley_k_psi':          0.5,
        'tlc':                    4.5,
        'max_heading_deg':        4.0,
        'human_settle_time':      20.0,
        'system_tlc_multiplier':  1.5,    # → 6.75 s
        'system_settle_time':     20.0,
        'mobil_p':                0.5,
        'mobil_a_th':             0.1,
    },
    'aggressive': {
        'velocity_offset':        5.0,    # → 38 m/s
        'x_error_offset':       -20.0,    # [m] positive → more conservative (larger initial error)
        'stanley_k_e':            0.008,
        'stanley_k_psi':          0.7,
        'tlc':                    3.0,
        'max_heading_deg':        6.0,
        'human_settle_time':      15.0,
        'system_tlc_multiplier':  1.5,    # → 4.5 s
        'system_settle_time':     15.0,
        'mobil_p':                0.2,    # Less polite
        'mobil_a_th':             0.05,
    },
}

# =============================================================================
# VEHICLE DYNAMICS PARAMETERS
# =============================================================================
# Road friction coefficient μ (Swain & Rath 2023, Eq. 1 — scales cornering stiffness)
# μ multiplies Cf and Cr in ALL linear model matrices (A_body_c, B_c)
# and is used in the nonlinear tire saturation model (_tire_force, Rajamani Eq. 13.45).
# Reference values: 1.0 = dry asphalt, 0.7 = wet road, 0.4 = slippery (Swain & Rath)
ROAD_FRICTION_MU = 1.0        # [-] tire-road friction coefficient

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
# Per-driver values live in DRIVER_PARAMS — access via DRIVER_PARAMS[driver_type].
# =============================================================================
# Velocity softening term (prevents division by zero at low speed)
STANLEY_K_SOFT = 1.0

# Low-pass filter on steering output
STANLEY_FILTER_ALPHA = 0.2    # 1st-order EMA on steering command [-]

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

# Own control effort weights (R) — tuned for Q_y/R ratio ≈ 0.016 (stable tracking)
NASH_R1 = 50000.0      # System's cost on its own control effort (lowered from 1e6 to enable tracking)
NASH_R2 = 50000.0      # Human's cost on its own control effort

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
# LATERAL SAFETY FIELD PARAMETERS — Driving Safety Field (DSF)
# Source: Li et al. (2019), Wang et al. (2015, 2016)
# Source file: nash_solver/lateral_safety_field.py - LateralSafetyFieldParams
# =============================================================================
# --- Elliptic distance parameters (Li 2019, Eq. 12-13) ---
# r* = sqrt(((x-xo)/a)² + ((y-yo)/b)²)
# a = max(|ve - vo| * DSF_TS, DSF_A_MIN)  [longitudinal semi-axis]
# b = DSF_TAU = 2 m                        [lateral safety circle radius]
DSF_TS    = 2.0    # TTC safety margin for longitudinal semi-axis [s] (Li 2019 Eq. 13)
DSF_TAU   = 2.0    # Lateral safety circle radius b = τ [m] (Li 2019 Table 1)
DSF_A_MIN = 2.0    # Min longitudinal semi-axis (= τ, for equal-speed case) [m]

# --- Field scaling and kinetic parameters (Li 2019) ---
# Fr = G * M_obs / |r*| * exp(k1*v_obs*cosθ) * M_ego * exp(-k2*v_ego*cosθ) * (1+DR)
# Calibrated to AUTHORITY_FORCE_MIDPOINT=200N (from simulation at DSF_G=6e-3):
#   join_after  (dx≈4.5m, dy=3.5m): measured Fr≈27N  → scale 10× → Fr≈270N > 200N ✓
#   join_middle (dx≈9m,   dy=3.5m): measured Fr≈16N  → scale 10× → Fr≈160N < 200N ✓
#   join_before (dx≥40m,  dy=3.5m): measured Fr≈8N   → scale 10× → Fr≈80N  < 200N ✓
DSF_G    = 6e-3    # Field scaling constant [calibrated to sigmoid midpoint=200N]
DSF_K1   = 0.005   # Kinetic field scaling weight (Li 2019 Table 1, Eq. 18)
DSF_K2   = 0.005   # Field force scaling weight   (Li 2019 Table 1, Eq. 21)
DSF_DR   = 0.5     # Driver risk factor DR        (Li 2019 Table 1)
DSF_VEHICLE_MASS = 1304.0  # Base vehicle mass for virtual mass computation [kg]

# Virtual mass speed coefficients (Wang 2016, Eq. 17):
# M = m * (1.566e-14 * v^6.687 + 0.3345)  — at 20 m/s, speed term ≈ 0, M ≈ 0.3345 * m
DSF_SPEED_COEFF    = 1.566e-14  # Speed-dependent mass coefficient
DSF_SPEED_EXPONENT = 6.687      # Speed exponent
DSF_SPEED_OFFSET   = 0.3345     # Speed-independent mass offset
DSF_EPSILON        = 1e-3       # Minimum elliptic distance (numerical safety) [m]

# --- Road boundary parameters (unchanged) ---
SAFETY_ROAD_HALF_WIDTH      = 7.0      # Half road width [m] (total 14m = 4 lanes)
SAFETY_BOUNDARY_FORCE_GAIN  = 150.0    # Boundary repulsive force gain [N]
SAFETY_BOUNDARY_FORCE_SCALE = 2.0      # tanh saturation scale for boundary force
SAFETY_BOUNDARY_PROXIMITY   = 3.0      # Activate boundary force when closer than this [m]
SAFETY_EPSILON               = 0.1     # Boundary numerical stability offset

# --- Force limits ---
SAFETY_MAX_FORCE = 2000.0               # Maximum total lateral DSF force [N]

# --- Direction smoothing ---
# tanh direction: at dy=1m: tanh(2)≈0.96 ≈ sign; at dy=0: tanh(0)=0 (no flip)
SAFETY_DIRECTION_SMOOTH_SIGMA = 0.5    # tanh width for smooth direction [m]

# --- Force output filter ---
SAFETY_FILTER_ALPHA = 0.3              # Low-pass EMA on safety force output [-]

# =============================================================================
# AUTHORITY ALLOCATOR PARAMETERS
# Source: nash_solver/lateral_authority_allocator.py - LateralAuthorityAllocator
# Aligned with the current longitudinal authority architecture (embedded-authority Nash pipeline)
# =============================================================================
# Sigmoid parameters for SAFETY (risk-based)
AUTHORITY_LAMBDA_MIN      = 0.1    # Human dominant when safe
AUTHORITY_LAMBDA_MAX      = 10.0   # System dominant when dangerous
AUTHORITY_FORCE_MIDPOINT  = 200.0   # Sigmoid centre [N] calibrated from empirical proximity-force range
AUTHORITY_K_STEEPNESS     = 0.025  # Sigmoid slope (sharper: human ~75% at rest vs 64% at k=0.02)

# Smoothing parameters
AUTHORITY_ALPHA_BASE = 0.02        # Slow smoothing (stability)
AUTHORITY_ALPHA_FAST = 0.08        # Fast response (large errors)

# Sigmoid parameters for LATERAL-OFFSET authority (Swain & Rath 2023, Eq. 15)
# γ1(human) = 1/(1+exp(m1·(-l_n+m2))), γ2=1-γ1, λ_sigmoid=γ2/γ1
# l_n = (l_o_max - |y_error|) / l_o_max  (1=lane centre, 0=lane boundary)
# Reference values: m1=2, m2=0.5 → λ range [0.37 at centre, 2.72 at boundary]
AUTHORITY_SIGMOID_M1 = 2.0   # Slope steepness (Swain & Rath: m1=2)
AUTHORITY_SIGMOID_M2 = 0.5   # Centre shift    (Swain & Rath: m2=0.5)

# =============================================================================
# REFERENCE GENERATOR PARAMETERS (Pustilnik & Borrelli 2025)
# Source: nash_solver/system_reference_generator.py, human_reference_generator.py
# =============================================================================
# --- Per-driver values live in DRIVER_PARAMS — access via DRIVER_PARAMS[driver_type]. ---
# --- System max heading constraint (shared, not per-driver) ---
REFGEN_SYSTEM_MAX_HEADING_DEG = 8.0    # Max heading angle for system trajectory [deg]

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
LANE_CHANGE_Y_ERROR_FACTOR  = 0.35   # |y_error| < 35% × lane_width to enter LANE_KEEPING
LANE_CHANGE_Y_DOT_THRESHOLD = 0.25   # |y_dot| < 0.25 m/s — nearly stopped laterally

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

# --- Per-driver MOBIL overrides live in DRIVER_PARAMS — access via DRIVER_PARAMS[driver_type]. ---

# =============================================================================
# COMFORT EVALUATION THRESHOLDS
# Source: lateral/metrics/comfort.py
# References: ISO 2631-1 (ay thresholds), Toledo & Zohar 2007 (duration),
#             Lee et al. 2004 (naturalistic data), empirical ADI thresholds
# =============================================================================
# Metric 1: Lane-change duration [s]
COMFORT_LC_DURATION_GOOD       =  6.0   # <= 6 s  natural (Toledo & Zohar 2007)
COMFORT_LC_DURATION_ACCEPTABLE = 10.0   # <= 10 s slow but acceptable

# Metric 2: Peak lateral acceleration [m/s²]
COMFORT_AY_PEAK_GOOD           = 0.5    # ISO 2631-1: "not uncomfortable"
COMFORT_AY_PEAK_ACCEPTABLE     = 1.0    # ISO 2631-1: "a little uncomfortable"

# Metric 3: RMS lateral acceleration [m/s²]
COMFORT_AY_RMS_GOOD            = 0.315  # ISO 2631-1 §6.2 lower boundary
COMFORT_AY_RMS_ACCEPTABLE      = 0.63   # ISO 2631-1 §6.2 upper boundary

# Metric 4: Peak lateral jerk [m/s³]
COMFORT_JERK_GOOD              = 0.5    # smooth transition
COMFORT_JERK_ACCEPTABLE        = 2.0    # noticeable but not harsh

# Metric 5: Peak body-frame lateral velocity [m/s]
# For sinusoidal LC profile: vy_peak = pi*delta_y/(2*T_lc)
# With delta_y=3.5m: T=6s -> 0.92 m/s (comfortable), T=4s -> 1.37 m/s (natural)
COMFORT_LATERAL_VEL_GOOD       = 1.0    # reachable only for T_lc >= 5.5s (smooth)
COMFORT_LATERAL_VEL_ACCEPTABLE = 1.5    # reachable for T_lc >= 3.7s (natural highway)

# Metric 6: Authority Disruption Index (ADI) — Nash-specific
# ADI = mean(|d(lambda)/dt|) / AUTHORITY_LAMBDA_MAX
# Measures rate of authority switches; 0 = smooth handover
COMFORT_ADI_GOOD               = 0.05
COMFORT_ADI_ACCEPTABLE         = 0.15

# =============================================================================
# EXPORT
# =============================================================================
__all__ = [
    # System
    'HEADLESS_MODE', 'RESULTS_DIR', 'SIMULATION_DT', 'NASH_CONTROL_DT',
    'NASH_NP', 'NASH_NU', 'DEFAULT_SIMULATION_TIME', 'NOMINAL_VELOCITY',
    'DRIVER_PARAMS',

    # Vehicle dynamics
    'ROAD_FRICTION_MU',

    # Lane / plotting
    'LANE_WIDTH', 'PLATOON_LANE_Y', 'HUMAN_INITIAL_LANE_Y', 'VEHICLE_COLORS',

    # Stanley controller
    'STANLEY_K_SOFT', 'STANLEY_FILTER_ALPHA',

    # Nash solver
    'NASH_Q_Y', 'NASH_Q_PSI',
    'NASH_Q_Y_TERMINAL_FACTOR', 'NASH_Q_PSI_TERMINAL_FACTOR',
    'NASH_R1', 'NASH_R2', 'NASH_S1', 'NASH_S2',
    'NASH_Y_MAX', 'NASH_PSI_MAX', 'NASH_LAMBDA_LEVELS',
    'NASH_SOLVER_BACKEND', 'NASH_WARM_START', 'NASH_VERBOSE',
    'NASH_MAX_ITER', 'NASH_EPS_ABS', 'NASH_EPS_REL',
    'NASH_POLISH', 'NASH_REGULARIZATION',

    # Lateral safety field — DSF parameters (Li 2019, Wang 2015/2016)
    'DSF_G', 'DSF_TS', 'DSF_TAU', 'DSF_A_MIN',
    'DSF_K1', 'DSF_K2', 'DSF_DR', 'DSF_VEHICLE_MASS',
    'DSF_SPEED_COEFF', 'DSF_SPEED_EXPONENT', 'DSF_SPEED_OFFSET', 'DSF_EPSILON',
    # Boundary and output parameters
    'SAFETY_ROAD_HALF_WIDTH', 'SAFETY_BOUNDARY_FORCE_GAIN', 'SAFETY_BOUNDARY_FORCE_SCALE',
    'SAFETY_BOUNDARY_PROXIMITY', 'SAFETY_EPSILON',
    'SAFETY_MAX_FORCE', 'SAFETY_DIRECTION_SMOOTH_SIGMA', 'SAFETY_FILTER_ALPHA',

    # Authority allocator
    'AUTHORITY_LAMBDA_MIN', 'AUTHORITY_LAMBDA_MAX',
    'AUTHORITY_FORCE_MIDPOINT', 'AUTHORITY_K_STEEPNESS',
    'AUTHORITY_ALPHA_BASE', 'AUTHORITY_ALPHA_FAST',
    'AUTHORITY_SIGMOID_M1', 'AUTHORITY_SIGMOID_M2',

    # Reference generators
    'REFGEN_SYSTEM_MAX_HEADING_DEG',
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

    # Comfort evaluation (ISO 2631-1 + Nash ADI)
    'COMFORT_LC_DURATION_GOOD', 'COMFORT_LC_DURATION_ACCEPTABLE',
    'COMFORT_AY_PEAK_GOOD', 'COMFORT_AY_PEAK_ACCEPTABLE',
    'COMFORT_AY_RMS_GOOD', 'COMFORT_AY_RMS_ACCEPTABLE',
    'COMFORT_JERK_GOOD', 'COMFORT_JERK_ACCEPTABLE',
    'COMFORT_LATERAL_VEL_GOOD', 'COMFORT_LATERAL_VEL_ACCEPTABLE',
    'COMFORT_ADI_GOOD', 'COMFORT_ADI_ACCEPTABLE',
]
