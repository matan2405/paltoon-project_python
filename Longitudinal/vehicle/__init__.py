"""
Vehicle module - Contains vehicle physics, components and dynamics.

This module provides:
- VehicleParameters: Physical vehicle parameters 
- Engine: Engine simulation model
- Transmission: Transmission model
- VehicleState: Vehicle state representation
- Vehicle: Main vehicle class with dynamics
"""

from .components import VehicleParameters, Engine, Transmission, VehicleState
from .vehicle import Vehicle

__all__ = [
    'VehicleParameters',
    'Engine', 
    'Transmission',
    'VehicleState',
    'Vehicle'
]