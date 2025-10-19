"""
Visualization module - Contains plotting and animation functionality.

This module provides:
- create_platoon_animation: Animated visualization
- create_comprehensive_plots: Static analysis plots  
- create_detailed_scenario_summary: Scenario reporting
"""

from .animation import create_platoon_animation
from .plots import create_comprehensive_plots, create_detailed_scenario_summary

__all__ = [
    'create_platoon_animation',
    'create_comprehensive_plots', 
    'create_detailed_scenario_summary'
]