"""
Global configuration for the Longitudinal Control simulation system.

Research basis and scope:
- Centralizes parameters used by longitudinal safety field, authority
    allocation, reference generation, platoon control, and Nash solver.
- Aligns with the same architecture used in the lateral stack for easier
    thesis traceability and cross-domain comparisons.

Based on the lateral system's config architecture (lateral/config.py).
All hardcoded parameters from nash_solver, safety_field, authority_allocator,
system_reference_generator, human_driver, and platoon_control are extracted here.

This file is the SINGLE SOURCE OF TRUTH for all longitudinal parameters.
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
        matplotlib.use('Qt5Agg', force=True)
        import matplotlib.pyplot as plt
        plt.ion()
        print("✅ Using Qt5Agg backend (interactive mode enabled)")
        HEADLESS_MODE = False
    except ImportError:
        try:
            matplotlib.use('TkAgg', force=True)
            import matplotlib.pyplot as plt
            plt.ion()
            print("✅ Using TkAgg backend (interactive mode enabled)")
            HEADLESS_MODE = False
        except ImportError:
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            print("⚠️ Using Agg backend (headless mode - plots will be saved only)")
            HEADLESS_MODE = True

    plt.ioff()

setup_matplotlib()

# =============================================================================
# RESULTS & FILE SYSTEM
# =============================================================================
RESULTS_DIR = "platoon_sim_kinematic_results"

def setup_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"📁 Created results directory: {RESULTS_DIR}")
    else:
        print(f"📁 Using existing results directory: {RESULTS_DIR}")

setup_results_directory()

# =============================================================================
# SIMULATION CONSTANTS
# =============================================================================
SIMULATION_DT = 0.01              # 100 Hz - vehicle dynamics integration step [s]
NASH_CONTROL_DT = 0.1             # 10 Hz  - Nash MPC control step [s]
NASH_NP = 20                      # Prediction horizon [steps]
NASH_NU = 10                      # Control horizon [steps]
DEFAULT_SIMULATION_TIME = 60 * 2  # Default simulation time [s]

# =============================================================================
# VEHICLE PARAMETERS (single source of truth)
# Import AFTER SIMULATION_DT so vehicle.py can resolve its top-level import.
# =============================================================================
from vehicle.components import VehicleParameters
_vp = VehicleParameters()

# Road grade — Belousov Eq. 3.11a grade resistance: Rg = m·g·sin(θ)
ROAD_GRADE = _vp.road_grade   # [rad] positive = uphill

# Vehicle colors for plotting
VEHICLE_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
GAP_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# =============================================================================
# PLATOON PARAMETERS
# =============================================================================
PLATOON_TARGET_VELOCITY = 120.0 / 3.6   # Target platoon velocity [m/s] (120 km/h)
PLATOON_MAX_VELOCITY = _vp.max_velocity * 3.6  # Max velocity limit [km/h] (from VehicleParameters)
PLATOON_MAX_ACCELERATION = _vp.max_acceleration  # Max acceleration limit [m/s²] (from VehicleParameters)

# Rajamani controller gains (Chapter 6.4)
RAJAMANI_H = 1.5           # Desired time headway [s]
RAJAMANI_TAU = 0.1         # Time lag [s]
RAJAMANI_K1 = -0.12        # Constraint: k1 < -tau/h
RAJAMANI_K5 = 0.1          # Constraint: k5 > 0
# Derived gains (from Rajamani Eq. 6.15-6.16)
RAJAMANI_K2 = -RAJAMANI_K1 - RAJAMANI_H * RAJAMANI_K1 * RAJAMANI_K5
RAJAMANI_K3 = 1.0 / RAJAMANI_H - RAJAMANI_K1 * RAJAMANI_K5
RAJAMANI_K4 = RAJAMANI_K5 / RAJAMANI_H

# Jerk limiting (ISO 15622 ACC comfort standard)
JERK_LIMIT = 2.0           # Maximum da/dt [m/s³]

# Free road acceleration IDM exponent
FREE_ROAD_DELTA = 4

# =============================================================================
# NASH SOLVER PARAMETERS (Li et al. 2019)
# Source: longitudinal_constrained_nash_solver.py - ConstrainedNashParams
# =============================================================================
# Cost weights - Output tracking
NASH_Q_POS = 2500.0               # Weight on position tracking error
NASH_Q_VEL = 1500.0               # Weight on velocity tracking error

# Control effort weights (R) - BASE VALUES
# S = R for cooperation (Nash equilibrium insight)
NASH_R1 = 30000.0*0.25                  # System's cost on its own control effort
NASH_R2 = 50000.0*0.25                  # Human's cost on its own control effort

# Cross-coupling weights (S) - THE KEY FOR COUPLED NASH GAME
NASH_S1 = 10000.0                   # System penalty on human control effort
NASH_S2 = 10000.0                   # Human penalty on system control effort

# Input constraints [m/s²] — derived from VehicleParameters
NASH_U1_MIN = _vp.max_deceleration        # System min acceleration (from vehicle specs)
NASH_U1_MAX = _vp.max_acceleration         # System max acceleration (from vehicle specs)
NASH_U2_MIN = -4.0                # Human min acceleration (more aggressive than vehicle limit)
NASH_U2_MAX = 3.0                 # Human max acceleration (more aggressive than vehicle limit)

# Rate constraints (jerk limits) [m/s³]
NASH_DU1_MAX = 1.5                # System max jerk
NASH_DU2_MAX = 2.0                # Human max jerk

# State constraints
NASH_V_MIN = 0.0                  # Min velocity [m/s]
NASH_V_MAX = 50.0                 # Max velocity [m/s]
NASH_GAP_MIN = 7.0                # Min safe gap [m]

# Lambda levels for pre-computation (authority ratio discrete values)
# Covers range from Li et al. lookup table: λ = 0.014 (safe) to λ = 106.6 (emergency)
NASH_LAMBDA_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)

# Solver settings
NASH_SOLVER_BACKEND = 'OSQP'
NASH_WARM_START = True
NASH_VERBOSE = False
NASH_MAX_ITER = 1000
NASH_EPS_ABS = 1e-4
NASH_EPS_REL = 1e-4
NASH_REGULARIZATION = 1e-5

# =============================================================================
# DRIVER TYPE - NASH PARAMETER MODIFIERS (Li et al. 2019)
# =============================================================================
# Aggressive driver: lower R2 (more control effort), lower S2 (less cooperative)
# Cautious driver: higher R2 (gentle control), higher S2 (more cooperative)
NASH_DRIVER_PARAMS = {
    'cautious': {
        'max_acceleration': 1.8,             # [m/s²]
        'comfortable_deceleration': 1.5,     # [m/s²]
        'desired_time_headway': 1.8,         # [s]
        'min_spacing': 3.0,                  # [m]
        'delta_IDM': 4.0,
        'plan_time_headway': 1.2,            # [s]
        'plan_decel': 3.0,                   # [m/s²]
        'target_speed_offset': -10.0,        # [km/h]
        'max_deceleration': -3.0,            # [m/s²]
        'initial_x_offset': 40.0,             # [m] Start with larger gap to leader
        'initial_speed': 0.0,                # [km/h] Target speed for the driver
    },
    'normal': {
        'max_acceleration': 2.5,
        'comfortable_deceleration': 2.0,
        'desired_time_headway': 1.2,
        'min_spacing': 2.0,
        'delta_IDM': 4.0,
        'plan_time_headway': 0.8,
        'plan_decel': 4.0,
        'target_speed_offset': 0.0,
        'max_deceleration': -4.0,
        'initial_x_offset': 0.0,              # [m] Start with normal gap to leader
        'initial_speed': 0.0,                # [km/h] Target speed for the driver
    },
    'aggressive': {
        'max_acceleration': 3.5,
        'comfortable_deceleration': 3.0,
        'desired_time_headway': 0.8,
        'min_spacing': 1.5,
        'delta_IDM': 4.0,
        'plan_time_headway': 0.5,
        'plan_decel': 5.0,
        'target_speed_offset': +10.0,
        'max_deceleration': -5.0,
        'initial_x_offset': -20.0,             # [m] Start with smaller gap to leader
        'initial_speed': 0.0,                # [km/h] Target speed for the driver
    }
}

# =============================================================================
# SAFETY FIELD PARAMETERS (Wang et al. 2015/2016)
# Source: longitudinal_safety_field.py - EllipseLongitudinalParams
# =============================================================================
# --- Hard Constraints ---
SAFETY_MIN_SAFE_DISTANCE = 7.0          # [m]
SAFETY_EMERGENCY_BRAKE_DISTANCE = 8.0   # [m]
SAFETY_MAX_DECEL = abs(_vp.max_deceleration)   # [m/s²] (from vehicle specs)

# --- Base Ellipse Parameters ---
SAFETY_BASE_SAFETY_RADIUS = 10.0
SAFETY_BASE_OBSTACLE_MASS = 400.0
SAFETY_BASE_INFLUENCE_FACTOR = 1.0
SAFETY_BASE_DRIVER_RISK = 0.5
SAFETY_EPSILON = 0.1

# --- Position Multipliers ---
SAFETY_LEADER_POSITION_MULTIPLIER = 0.8
SAFETY_MIDDLE_POSITION_MULTIPLIER = 1.2
SAFETY_FOLLOWER_POSITION_MULTIPLIER = 1.0

# --- Risk Multipliers ---
SAFETY_NORMAL_RISK_MULTIPLIER = 1.0
SAFETY_MODERATE_RISK_MULTIPLIER = 1.5
SAFETY_HIGH_RISK_MULTIPLIER = 2.0
SAFETY_EMERGENCY_RISK_MULTIPLIER = 2.5

# --- Velocity Scaling ---
SAFETY_VELOCITY_REFERENCE = 120.0       # [km/h]
SAFETY_VELOCITY_SCALING_FACTOR = 0.8

# --- Distance Decay ---
SAFETY_DISTANCE_DECAY_FACTOR = 15.0

# --- Force Limits ---
SAFETY_MAX_REPULSIVE_FORCE = 800.0
SAFETY_MIN_ATTRACTIVE_FORCE = -500.0

# --- Follower Weights ---
SAFETY_LEADER_WEIGHT = 1.0
SAFETY_FOLLOWER_WEIGHT = 0.5
SAFETY_JOINING_FOLLOWER_WEIGHT = 0.5  # increased from 0.2 to push human away from approaching P3
SAFETY_DECELERATING_FOLLOWER_WEIGHT = 0.4

# --- Platoon Factors ---
SAFETY_PLATOON_COHERENCE_FACTOR = 1.5
SAFETY_MERGING_MULTIPLIER = 1.2

# --- Low-Pass Filter ---
SAFETY_FILTER_ALPHA = 0.2

# --- Hard Constraint Jerk Limit ---
SAFETY_HARD_CONSTRAINT_MAX_JERK = 2.0   # [m/s³]
SAFETY_HARD_CONSTRAINT_DT = 0.02        # [s]
SAFETY_HARD_CONSTRAINT_U_MIN = _vp.max_deceleration    # [m/s²] (from vehicle specs)
SAFETY_HARD_CONSTRAINT_U_MAX = _vp.max_acceleration     # [m/s²] (from vehicle specs)

# --- Road Conditions ---
SAFETY_ROAD_CONDITION_FACTORS = {
    "dry": 1.0,
    "wet": 0.7,
    "icy": 0.5,
}

# =============================================================================
# LOWER-LEVEL CONTROLLER PARAMETERS (Rajamani Ch.6 / Belousov et al.)
# Hierarchical control: Upper (double integrator) → Lower (force-level)
# =============================================================================
# PI feedback gains for acceleration tracking
LOWER_CTRL_KP = 0.4                    # Proportional gain on accel error [-]
LOWER_CTRL_KI = 0.08                   # Integral gain on accel error [-]
LOWER_CTRL_INTEGRATOR_LIMIT = 2.0      # Anti-windup saturation [m/s²·s]

# Actuator smoothing (1st-order filter: 1.0 = no smoothing, 0.0 = full hold)
LOWER_CTRL_THROTTLE_SMOOTHING = 0.08   # Throttle filter coefficient [-]
LOWER_CTRL_BRAKE_SMOOTHING = 0.10      # Brake filter coefficient [-]

# Dead-band for throttle/brake switching [N]
LOWER_CTRL_DEADBAND = 50.0

# Acceleration output filter (1st-order EMA on a_actual fed back to upper controller)
# Smooths engine/gear-shift noise before Rajamani/Nash see it.
# α = dt/(τ+dt): with dt=0.01s, τ=0.15s → α≈0.062
LOWER_CTRL_ACCEL_FILTER_ALPHA = 0.06   # Accel output filter coefficient [-]

# Engine braking (pumping losses + friction when throttle = 0, brake = 0)
# Retarding torque scales linearly with RPM: T_brake = base * (RPM / 2000)
ENGINE_BRAKING_BASE_TORQUE = 60.0      # Base retarding torque at 2000 RPM [Nm]
ENGINE_BRAKING_MAX_TORQUE = 120.0      # Maximum retarding torque cap [Nm]

# Gear-shift hold: freeze PI output while transmission is shifting
# to avoid chasing transient force discontinuities
LOWER_CTRL_GEAR_SHIFT_HOLD = True      # Freeze actuator output during gear shifts

# Physical constants (air density, rolling coeff) come from VehicleParameters.
# Use _vp.air_density and _vp.rolling_resistance_coeff instead of separate config entries.

# =============================================================================
# PHASE DETECTION PARAMETERS (MERGING ↔ FOLLOWING)
# Source: longitudinal_safety_field.py
# =============================================================================
# Entry to FOLLOWING thresholds (ALL must be satisfied for transition_time)
FOLLOWING_GAP_ERROR_FACTOR = 0.15             # |gap_error| < 15% of desired_gap
FOLLOWING_VELOCITY_ERROR_FACTOR = 0.05        # |rel_velocity| < 5% of target_velocity
FOLLOWING_ACCELERATION_THRESHOLD = 0.5        # |acceleration| < 0.5 m/s²

# Exit from FOLLOWING thresholds (ANY triggers return to MERGING - hysteresis)
MERGING_GAP_ERROR_FACTOR = 0.25               # |gap_error| > 25% of desired_gap
MERGING_VELOCITY_ERROR_FACTOR = 0.10          # |rel_velocity| > 10% of target_velocity

# Phase transition confirmation time
PHASE_TRANSITION_TIME = 5.0                   # [s] stability required

# =============================================================================
# AUTHORITY ALLOCATOR PARAMETERS
# Source: longitudinal_authority_allocator.py
# =============================================================================
# Sigmoid parameters for SAFETY (risk-based)
AUTHORITY_LAMBDA_MIN = 0.1          # Human dominant when safe
AUTHORITY_LAMBDA_MAX = 100.0        # High λ -> low α: emergency regime where system overrides
AUTHORITY_FORCE_MIDPOINT = 400.0    # Sigmoid center point [N]
AUTHORITY_K_STEEPNESS = 0.015       # Sigmoid slope

# Smoothing parameters
AUTHORITY_ALPHA_BASE = 0.05         # Base smoothing (extremely smooth)
AUTHORITY_ALPHA_FAST = 0.12         # Fast response (still smooth)

# Gap-error sigmoid parameters (Swain & Rath 2023, Eq. 15 — adapted for longitudinal)
AUTHORITY_SIGMOID_M1 = 2.0          # Slope steepness (Swain & Rath: m1=2)
AUTHORITY_SIGMOID_M2 = 0.5          # Centre shift    (Swain & Rath: m2=0.5)
AUTHORITY_GAP_ERROR_MAX = 5.0       # Max gap error for full authority [m]
                                    # (analogous to LANE_WIDTH/2 in lateral)
AUTHORITY_VEL_ERROR_MAX = 1.5       # Max velocity error for full authority [m/s]
                                    # (~10.8 km/h): sigmoid saturates at this speed deviation

# =============================================================================
# SYSTEM REFERENCE GENERATOR PARAMETERS (Rajamani Ch. 6.7)
# Source: system_reference_generator.py
# =============================================================================
# Detection range
REFGEN_DETECTION_RANGE = 150.0      # [m]

# CTG policy (Rajamani Eq. 6.5)
REFGEN_TIME_HEADWAY = 1.5          # [s]
REFGEN_STANDSTILL_DISTANCE = 2.0   # [m] balanced: 0 overshoots below, 5 overshoots above desired gap

# Comfortable deceleration (Rajamani Section 6.7.2, typical 0.1g-0.2g)
REFGEN_A_COMFORT = 1.5             # [m/s²] (gentler than vehicle comfortable_deceleration for smooth refs)

# Velocity tracking gain (Rajamani Section 6.7.2, typical 0.3-0.5)
REFGEN_K_V = 0.4                   # [1/s]

# CTG Controller gains (Rajamani Eq. 6.15-6.16)
REFGEN_K1 = 0.23                   # [1/s²] position gain (typical 0.2-0.4)
REFGEN_K2 = 0.6                    # [1/s]  velocity gain (typical 0.6-1.0)

# Comfort limits
REFGEN_MAX_ACCEL = _vp.max_acceleration     # [m/s²] (from vehicle specs)
REFGEN_MAX_DECEL = _vp.max_deceleration     # [m/s²] (from vehicle specs)

# Cruise catch-up factor
REFGEN_CATCHUP_FACTOR = 1.15

# Free-road IDM exponent for RefGen CRUISE mode
# Higher delta → stronger push near target speed (join_before steady-state velocity)
# delta=4: a(99.5%→100%) = 0.05 m/s²  | delta=6: 0.074 m/s²  | delta=8: 0.098 m/s²
REFGEN_FREE_ROAD_DELTA = 6

# CTG blending zone (gap error threshold for CTG/parabola blend)
REFGEN_CTG_BLEND_ZONE = 5.0        # [m] - blend in CTG for |gap_error| < 5m
REFGEN_CTG_BLEND_POWER = 2.0       # Quadratic blending

# Leader acceleration feedforward gain
REFGEN_LEADER_FF_GAIN = 0.5

# =============================================================================
# HUMAN DRIVER PARAMETERS (IDM-based)
# Source: human_driver.py
# =============================================================================
# IDM execution parameters — derived from VehicleParameters where applicable
HUMAN_MAX_ACCELERATION = _vp.max_acceleration   # [m/s²] (from vehicle specs)
HUMAN_DESIRED_TIME_HEADWAY = 1.2   # [s]
HUMAN_MIN_SPACING = 2.0            # [m]
HUMAN_DELTA_IDM = 4.0              # IDM exponent
HUMAN_COMFORTABLE_DECEL = _vp.comfortable_deceleration  # [m/s²] (from vehicle specs)

# IDM planning parameters (more tolerant for Nash prediction)
HUMAN_PLAN_TIME_HEADWAY = 0.8      # [s] shorter headway for planning
HUMAN_PLAN_DECEL = 4.0             # [m/s²] harder braking allowed in planning

# Execution constraints — derived from VehicleParameters
HUMAN_ACCEL_MIN = -4.0             # [m/s²] (human can brake harder than vehicle limit)
HUMAN_ACCEL_MAX = _vp.max_acceleration  # [m/s²] (from vehicle specs)

# Lane change rate
HUMAN_LANE_CHANGE_RATE = 0.25      # Progress per second
LANE_WIDTH = 3.5                     # [m]
# =============================================================================
# EXPORT
# =============================================================================
__all__ = [
    # System
    'HEADLESS_MODE', 'RESULTS_DIR', 'SIMULATION_DT', 'NASH_CONTROL_DT',
    'NASH_NP', 'NASH_NU', 'DEFAULT_SIMULATION_TIME',
    'VEHICLE_COLORS', 'GAP_COLORS','LANE_WIDTH',

    # Platoon
    'PLATOON_TARGET_VELOCITY', 'PLATOON_MAX_VELOCITY', 'PLATOON_MAX_ACCELERATION',
    'RAJAMANI_H', 'RAJAMANI_TAU', 'RAJAMANI_K1', 'RAJAMANI_K2', 'RAJAMANI_K3',
    'RAJAMANI_K4', 'RAJAMANI_K5', 'FREE_ROAD_DELTA',

    # Nash solver
    'NASH_Q_POS', 'NASH_Q_VEL', 'NASH_R1', 'NASH_R2', 'NASH_S1', 'NASH_S2',
    'NASH_U1_MIN', 'NASH_U1_MAX', 'NASH_U2_MIN', 'NASH_U2_MAX',
    'NASH_DU1_MAX', 'NASH_DU2_MAX', 'NASH_V_MIN', 'NASH_V_MAX', 'NASH_GAP_MIN',
    'NASH_LAMBDA_LEVELS', 'NASH_SOLVER_BACKEND', 'NASH_WARM_START',
    'NASH_VERBOSE', 'NASH_MAX_ITER', 'NASH_EPS_ABS', 'NASH_EPS_REL',
    'NASH_REGULARIZATION', 'NASH_DRIVER_PARAMS',

    # Safety field
    'SAFETY_MIN_SAFE_DISTANCE', 'SAFETY_EMERGENCY_BRAKE_DISTANCE', 'SAFETY_MAX_DECEL',
    'SAFETY_BASE_SAFETY_RADIUS', 'SAFETY_BASE_OBSTACLE_MASS',
    'SAFETY_BASE_INFLUENCE_FACTOR', 'SAFETY_BASE_DRIVER_RISK', 'SAFETY_EPSILON',
    'SAFETY_LEADER_POSITION_MULTIPLIER', 'SAFETY_MIDDLE_POSITION_MULTIPLIER',
    'SAFETY_FOLLOWER_POSITION_MULTIPLIER',
    'SAFETY_NORMAL_RISK_MULTIPLIER', 'SAFETY_MODERATE_RISK_MULTIPLIER',
    'SAFETY_HIGH_RISK_MULTIPLIER', 'SAFETY_EMERGENCY_RISK_MULTIPLIER',
    'SAFETY_VELOCITY_REFERENCE', 'SAFETY_VELOCITY_SCALING_FACTOR',
    'SAFETY_DISTANCE_DECAY_FACTOR',
    'SAFETY_MAX_REPULSIVE_FORCE', 'SAFETY_MIN_ATTRACTIVE_FORCE',
    'SAFETY_LEADER_WEIGHT', 'SAFETY_FOLLOWER_WEIGHT',
    'SAFETY_JOINING_FOLLOWER_WEIGHT', 'SAFETY_DECELERATING_FOLLOWER_WEIGHT',
    'SAFETY_PLATOON_COHERENCE_FACTOR', 'SAFETY_MERGING_MULTIPLIER',
    'SAFETY_FILTER_ALPHA',
    'SAFETY_HARD_CONSTRAINT_MAX_JERK', 'SAFETY_HARD_CONSTRAINT_DT',
    'SAFETY_HARD_CONSTRAINT_U_MIN', 'SAFETY_HARD_CONSTRAINT_U_MAX',
    'SAFETY_ROAD_CONDITION_FACTORS',

    # Phase detection
    'FOLLOWING_GAP_ERROR_FACTOR', 'FOLLOWING_VELOCITY_ERROR_FACTOR',
    'FOLLOWING_ACCELERATION_THRESHOLD',
    'MERGING_GAP_ERROR_FACTOR', 'MERGING_VELOCITY_ERROR_FACTOR',
    'PHASE_TRANSITION_TIME',

    # Authority allocator
    'AUTHORITY_LAMBDA_MIN', 'AUTHORITY_LAMBDA_MAX',
    'AUTHORITY_FORCE_MIDPOINT', 'AUTHORITY_K_STEEPNESS',
    'AUTHORITY_ALPHA_BASE', 'AUTHORITY_ALPHA_FAST',
    'AUTHORITY_SIGMOID_M1', 'AUTHORITY_SIGMOID_M2',
    'AUTHORITY_GAP_ERROR_MAX',

    # System reference generator
    'REFGEN_DETECTION_RANGE', 'REFGEN_TIME_HEADWAY', 'REFGEN_STANDSTILL_DISTANCE',
    'REFGEN_A_COMFORT', 'REFGEN_K_V', 'REFGEN_K1', 'REFGEN_K2',
    'REFGEN_MAX_ACCEL', 'REFGEN_MAX_DECEL', 'REFGEN_CATCHUP_FACTOR',
    'REFGEN_FREE_ROAD_DELTA',
    'REFGEN_CTG_BLEND_ZONE', 'REFGEN_CTG_BLEND_POWER', 'REFGEN_LEADER_FF_GAIN',

    # Human driver
    'HUMAN_MAX_ACCELERATION', 'HUMAN_DESIRED_TIME_HEADWAY', 'HUMAN_MIN_SPACING',
    'HUMAN_DELTA_IDM', 'HUMAN_COMFORTABLE_DECEL',
    'HUMAN_PLAN_TIME_HEADWAY', 'HUMAN_PLAN_DECEL',
    'HUMAN_ACCEL_MIN', 'HUMAN_ACCEL_MAX',
    'HUMAN_LANE_CHANGE_RATE',

    # Lower-level controller
    'LOWER_CTRL_KP', 'LOWER_CTRL_KI', 'LOWER_CTRL_INTEGRATOR_LIMIT',
    'LOWER_CTRL_THROTTLE_SMOOTHING', 'LOWER_CTRL_BRAKE_SMOOTHING',
    'LOWER_CTRL_DEADBAND',
    'LOWER_CTRL_ACCEL_FILTER_ALPHA', 'LOWER_CTRL_GEAR_SHIFT_HOLD',
    'ENGINE_BRAKING_BASE_TORQUE', 'ENGINE_BRAKING_MAX_TORQUE',

    # Jerk limit
    'JERK_LIMIT',
]