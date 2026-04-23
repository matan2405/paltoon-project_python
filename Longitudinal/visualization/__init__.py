"""
Visualization module - Contains plotting and animation functionality.

This module provides:
- create_platoon_animation: Animated visualization
- create_comprehensive_plots: Static analysis plots  
- create_detailed_scenario_summary: Scenario reporting
- create_hierarchical_control_plots: Lower-level controller metrics
"""

from .animation import create_platoon_animation
from .plots import (create_comprehensive_plots, create_detailed_scenario_summary,
                    create_hierarchical_control_plots)

__all__ = [
    'create_platoon_animation',
    'create_comprehensive_plots', 
    'create_detailed_scenario_summary',
    'create_hierarchical_control_plots'
]