"""
Control module - Contains control algorithms and drivers.

This module provides:
- PlatoonManager: Manages autonomous platoon behavior
- HumanDriver: Human driver simulation model
- LowerLevelController: Converts desired acceleration to throttle/brake
- rajamani: Rajamani control algorithm
- free_road_acc: Free road acceleration function
"""

from .platoon_control import PlatoonManager, rajamani, free_road_acc
from .human_driver import HumanDriver
from .lower_level_controller import LowerLevelController

__all__ = [
    'PlatoonManager',
    'rajamani', 
    'free_road_acc',
    'HumanDriver',
    'LowerLevelController'
]