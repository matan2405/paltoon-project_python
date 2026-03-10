"""
Vehicle module - Contains vehicle physics, components and dynamics.

This module provides:
- VehicleParameters: Physical vehicle parameters
- VehicleState: Vehicle state representation
- Vehicle: Main vehicle class with dynamics

Engine and Transmission have moved to control/lower_level_controller.py.
"""

from .components import VehicleParameters, VehicleState
from .vehicle import Vehicle

__all__ = [
    'VehicleParameters',
    'VehicleState',
    'Vehicle'
]