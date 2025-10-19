"""
Simulation module - Contains simulation engine and execution logic.

This module provides:
- PlatoonSimulation: Main simulation class
- run_simulation: Basic simulation runner
"""

from .simulator import PlatoonSimulation, run_simulation

__all__ = [
    'PlatoonSimulation',
    'run_simulation'
]