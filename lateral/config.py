"""
Global configuration for the Lateral Control simulation system.
VERSION 2.0 - Based on successful longitudinal controller architecture
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# Global matplotlib configuration
def setup_matplotlib():
    global HEADLESS_MODE
    import matplotlib
    
    # Try to use interactive backend on Windows/Linux with display
    try:
        # Check if we have a display
        import os
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

RESULTS_DIR = "lateral_sim_results_v2"

def setup_results_directory():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

setup_results_directory()

# =============================================================================
# SIMULATION CONSTANTS
# =============================================================================
SIMULATION_DT = 0.01          # 100 Hz - vehicle dynamics
NASH_CONTROL_DT = 0.05        # 20 Hz - Nash MPC control step
NASH_NP = 20     # Prediction horizon
NASH_NU = 10# Control horizon
DEFAULT_SIMULATION_TIME = 120.0  # seconds
NOMINAL_VELOCITY = 20.0       # m/s - nominal longitudinal velocity

# Lane configuration
LANE_WIDTH = 3.5              # meters
PLATOON_LANE_Y = 0.0          # Platoon travels at y=0
HUMAN_INITIAL_LANE_Y = 3.5    # Human starts at y=3.5m

# Vehicle colors for plotting
VEHICLE_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# =============================================================================
# STANLEY CONTROLLER GAINS - Based on stable PD controller
# k_y=0.005, k_psi=0.5 work well
# =============================================================================
STANLEY_K_E_CAUTIOUS = 0.003
STANLEY_K_E_NORMAL = 0.005
STANLEY_K_E_AGGRESSIVE = 0.008

STANLEY_K_PSI_CAUTIOUS = 0.3
STANLEY_K_PSI_NORMAL = 0.5
STANLEY_K_PSI_AGGRESSIVE = 0.7

# =============================================================================
# NASH SOLVER PARAMETERS - Matching longitudinal structure (Li et al. 2019)
# =============================================================================
# Output weights [y, psi]
# CRITICAL: Q_psi >> Q_y for stability (stabilize heading first!)
# REDUCED Q_y for gentler, more gradual lane change
NASH_Q_Y = 10.0         # Weight on lateral position error (was 50 - too aggressive)
NASH_Q_PSI = 10000.0     # Weight on heading angle error (HIGH!)

# Own control effort weights (R) - BASE VALUES
# VERY HIGH for smooth, gentle control
NASH_R1 = 1000000.0     # System's cost on its own control (was 500000)
NASH_R2 = 1000000.0     # Human's cost on its own control (BASE - will be modified by driver type)

# Cross-coupling weights (S) - THE KEY FOR COUPLED NASH GAME
NASH_S1 = 200000.0      # System cares about human effort (was 100000)
NASH_S2 = 200000.0      # Human cares about system effort

# =============================================================================
# AUTHORITY ALLOCATOR PARAMETERS - Aligned with longitudinal V5.2
# =============================================================================
# Sigmoid parameters for SAFETY (risk-based)
AUTHORITY_LAMBDA_MIN = 0.1          # Human dominant when safe
AUTHORITY_LAMBDA_MAX = 10.0         # System dominant when dangerous
AUTHORITY_FORCE_MIDPOINT = 150.0    # Sigmoid center point [N]
AUTHORITY_K_STEEPNESS = 0.02        # Sigmoid slope

# Smoothing parameters
AUTHORITY_ALPHA_BASE = 0.02         # Slow smoothing (stability)
AUTHORITY_ALPHA_FAST = 0.08         # Fast response (large errors)

# Hysteresis thresholds for PERFORMANCE authority (based on y_error)
AUTHORITY_ENTER_THRESHOLD = 1.0     # Enter performance mode when |y_error| > 1.0m
AUTHORITY_EXIT_THRESHOLD = 0.3      # Exit performance mode when |y_error| < 0.3m
AUTHORITY_LAMBDA_PERFORMANCE_MAX = 5.0  # Performance authority upper bound

# =============================================================================
# PHASE DETECTION PARAMETERS - From longitudinal controller
# =============================================================================
# Entry to FOLLOWING (all must be true for 5 seconds)
FOLLOWING_Y_ERROR_FACTOR = 0.15      # |y_error| < 15% * lane_width
FOLLOWING_PSI_ERROR_THRESHOLD = 0.05  # |psi| < ~3 degrees
FOLLOWING_Y_DOT_THRESHOLD = 0.3       # |y_dot| < 0.3 m/s

# Exit from FOLLOWING (any triggers exit)
MERGING_Y_ERROR_FACTOR = 0.25        # |y_error| > 25% * lane_width
MERGING_PSI_ERROR_THRESHOLD = 0.10   # |psi| > ~6 degrees

# Phase transition time
PHASE_TRANSITION_TIME = 5.0          # seconds of stability required for FOLLOWING entry

# Phase duration thresholds
GAP_SEARCH_DURATION = 0.5            # seconds to wait in GAP_SEARCH before LANE_CHANGE
LANE_CHANGE_MIN_TIME = 3.0           # min seconds in LANE_CHANGE before LANE_KEEPING
LANE_CHANGE_Y_ERROR_FACTOR = 0.3     # |y_error| < 30% * lane_width to enter LANE_KEEPING

__all__ = [
    'HEADLESS_MODE', 'RESULTS_DIR', 'SIMULATION_DT', 'NASH_CONTROL_DT', 'NASH_NP', 'NASH_NU',
    'DEFAULT_SIMULATION_TIME', 'NOMINAL_VELOCITY',
    'LANE_WIDTH', 'PLATOON_LANE_Y', 'HUMAN_INITIAL_LANE_Y', 'VEHICLE_COLORS',
    'STANLEY_K_E_CAUTIOUS', 'STANLEY_K_E_NORMAL', 'STANLEY_K_E_AGGRESSIVE',
    'STANLEY_K_PSI_CAUTIOUS', 'STANLEY_K_PSI_NORMAL', 'STANLEY_K_PSI_AGGRESSIVE',
    'NASH_Q_Y', 'NASH_Q_PSI', 'NASH_R1', 'NASH_R2', 'NASH_S1', 'NASH_S2',
    'AUTHORITY_LAMBDA_MIN', 'AUTHORITY_LAMBDA_MAX', 'AUTHORITY_FORCE_MIDPOINT',
    'AUTHORITY_K_STEEPNESS', 'AUTHORITY_ALPHA_BASE', 'AUTHORITY_ALPHA_FAST',
    'AUTHORITY_ENTER_THRESHOLD', 'AUTHORITY_EXIT_THRESHOLD', 'AUTHORITY_LAMBDA_PERFORMANCE_MAX',
    'FOLLOWING_Y_ERROR_FACTOR', 'FOLLOWING_PSI_ERROR_THRESHOLD',
    'FOLLOWING_Y_DOT_THRESHOLD', 'MERGING_Y_ERROR_FACTOR', 'MERGING_PSI_ERROR_THRESHOLD',
    'PHASE_TRANSITION_TIME', 'GAP_SEARCH_DURATION', 'LANE_CHANGE_MIN_TIME', 'LANE_CHANGE_Y_ERROR_FACTOR',
]
