"""Nash solver package for lateral shared control with embedded-authority GNE formulation."""
from .lateral_constrained_nash_solver import (
    ConstrainedLateralNashSolver, 
    ConstrainedLateralNashParams
)
from .lateral_safety_field import LateralSafetyField, LateralSafetyFieldParams, ControlPhase
from .lateral_authority_allocator import LateralAuthorityAllocator
from .system_reference_generator import SystemReferenceGenerator, TrajectoryPhase, DynamicTrajectoryParams
from .human_reference_generator import HumanReferenceGenerator, HumanTrajectoryPhase, HumanDynamicParams

__all__ = [
    # Nash Solver (constrained MPC)
    'ConstrainedLateralNashSolver', 'ConstrainedLateralNashParams',
    # Safety Field
    'LateralSafetyField', 'LateralSafetyFieldParams', 'ControlPhase',
    # Authority Allocator
    'LateralAuthorityAllocator',
    # Reference Generators
    'SystemReferenceGenerator', 'TrajectoryPhase', 'DynamicTrajectoryParams',
    'HumanReferenceGenerator', 'HumanTrajectoryPhase', 'HumanDynamicParams'
]
