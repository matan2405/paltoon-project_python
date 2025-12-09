"""
Global configuration and setup for the platoon simulation system.
Handles matplotlib backend selection and global constants.
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# Global matplotlib configuration
def setup_matplotlib():
    """Configure matplotlib backend with fallback options."""
    global HEADLESS_MODE
    
    try:
        # Force interactive mode - don't fall back to headless unless absolutely necessary
        matplotlib.use('Qt5Agg', force=True)
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode
        print("✅ Using Qt5Agg backend (interactive mode enabled)")
        HEADLESS_MODE = False
    except ImportError as e:
        try:
            # Try TkAgg if Qt5Agg not available
            matplotlib.use('TkAgg', force=True)
            import matplotlib.pyplot as plt
            plt.ion()  # Turn on interactive mode
            print("✅ Using TkAgg backend (interactive mode enabled)")
            HEADLESS_MODE = False
        except ImportError:
            # Only use Agg as last resort
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            print("⚠️ Using Agg backend (headless mode - plots will be saved only)")
            HEADLESS_MODE = True

    # Enable interactive mode for better display
    matplotlib.pyplot.ion()
    
    # Disable interactive mode to prevent threading issues
    plt.ioff()

# Initialize matplotlib
setup_matplotlib()

# Results directory configuration
RESULTS_DIR = "platoon_sim_kinematic_results"

def setup_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"📁 Created results directory: {RESULTS_DIR}")
    else:
        print(f"📁 Using existing results directory: {RESULTS_DIR}")

# Initialize results directory
setup_results_directory()

# Global constants
SIMULATION_DT = 0.02  # Default timestep
DEFAULT_SIMULATION_TIME = 60 * 2  # Default simulation time in seconds

# Vehicle colors for plotting
VEHICLE_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
GAP_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Export important variables
__all__ = [
    'HEADLESS_MODE',
    'RESULTS_DIR', 
    'SIMULATION_DT',
    'DEFAULT_SIMULATION_TIME',
    'VEHICLE_COLORS',
    'GAP_COLORS',
    'setup_matplotlib',
    'setup_results_directory'
]