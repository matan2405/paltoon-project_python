"""
Lateral Authority Allocator.
VERSION 3.0 - Aligned with Longitudinal V5.2

Changes from V2.2:
- Replaced rate-limiting + fixed smoothing with HYSTERESIS + adaptive smoothing
- Added PERFORMANCE authority based on y_error (mirrors longitudinal gap_error logic)
- All parameters moved to config.py (no hardcoded values)
- Logic mirrors longitudinal_authority_allocator.py V5.2
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUTHORITY_LAMBDA_MIN, AUTHORITY_LAMBDA_MAX,
    AUTHORITY_FORCE_MIDPOINT, AUTHORITY_K_STEEPNESS,
    AUTHORITY_ALPHA_BASE, AUTHORITY_ALPHA_FAST,
    AUTHORITY_ENTER_THRESHOLD, AUTHORITY_EXIT_THRESHOLD,
    AUTHORITY_LAMBDA_PERFORMANCE_MAX,
)


class LateralAuthorityAllocator:
    """
    Dynamic authority allocation with HYSTERESIS to prevent oscillations.

    VERSION 3.0 - Aligned with Longitudinal V5.2:

    Two authority sources (same structure as longitudinal):
    1. SAFETY authority  — sigmoid on risk_force magnitude
    2. PERFORMANCE authority — hysteresis on |y_error| (mirrors gap_error in longitudinal)

    Combined: target_lambda = max(lambda_safety, lambda_performance)

    Adaptive smoothing:
    - alpha_fast when |y_error| > 2× ENTER_THRESHOLD (need quick response)
    - alpha_base otherwise (need stability)
    """

    def __init__(self):
        # Sigmoid parameters for SAFETY (risk-based)
        self.lambda_min = AUTHORITY_LAMBDA_MIN
        self.lambda_max = AUTHORITY_LAMBDA_MAX
        self.force_midpoint = AUTHORITY_FORCE_MIDPOINT
        self.k_steepness = AUTHORITY_K_STEEPNESS

        self.prev_lambda = self.lambda_min

        # Smoothing parameters
        self.alpha_base = AUTHORITY_ALPHA_BASE
        self.alpha_fast = AUTHORITY_ALPHA_FAST

        # Hysteresis state for PERFORMANCE authority
        self.in_performance_mode = False
        self.ENTER_THRESHOLD = AUTHORITY_ENTER_THRESHOLD
        self.EXIT_THRESHOLD = AUTHORITY_EXIT_THRESHOLD
        self.lambda_performance_max = AUTHORITY_LAMBDA_PERFORMANCE_MAX

        # Logging
        self.last_lambda_safety = 0.0
        self.last_lambda_performance = 0.0

        print(f"🎯 Authority Allocator V3.0 (Hysteresis) Initialized")
        print(f"   λ range: {self.lambda_min}-{self.lambda_max}")
        print(f"   Smoothing: alpha_base={self.alpha_base}, alpha_fast={self.alpha_fast}")
        print(f"   Hysteresis: enter={self.ENTER_THRESHOLD}m, exit={self.EXIT_THRESHOLD}m")

    def compute_authority_ratio(self, risk_force: float,
                                 lateral_error: float = 0.0,
                                 heading_error: float = 0.0) -> float:
        """
        Compute authority ratio λ(k) with hysteresis.

        Args:
            risk_force:    Safety field force magnitude [N]
            lateral_error: y_error = ego_y - target_y [m]  (mirrors gap_error in longitudinal)
            heading_error: psi_error [rad] (unused in authority computation)

        Returns:
            lambda_k: Authority ratio (higher = more system authority)
        """
        force_mag = abs(risk_force)
        abs_y_error = abs(lateral_error)

        # --- 1. SAFETY Authority (sigmoid on force) ---
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. PERFORMANCE Authority with HYSTERESIS ---
        # Enter: |y_error| > ENTER_THRESHOLD (active lane change)
        # Exit:  |y_error| < EXIT_THRESHOLD  (close to target)
        if not self.in_performance_mode:
            if abs_y_error > self.ENTER_THRESHOLD:
                self.in_performance_mode = True
        else:
            if abs_y_error < self.EXIT_THRESHOLD:
                self.in_performance_mode = False

        lambda_performance = self.lambda_min  # Default: human in control

        if self.in_performance_mode:
            # Progressive authority based on distance from target
            # Normalized relative to enter/exit band
            normalized_error = (abs_y_error - self.EXIT_THRESHOLD) / (self.ENTER_THRESHOLD - self.EXIT_THRESHOLD)
            normalized_error = max(0.0, normalized_error)

            lambda_performance = 1.0 + normalized_error * 0.3 + (abs_y_error - self.EXIT_THRESHOLD) * 0.25
            lambda_performance = min(lambda_performance, self.lambda_performance_max)

        # --- 3. Fusion: highest urgency wins ---
        target_lambda = max(lambda_safety, lambda_performance)

        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_performance = lambda_performance

        # --- 4. Adaptive smoothing ---
        if abs_y_error > 2.0 * self.ENTER_THRESHOLD:
            alpha = self.alpha_fast
        elif abs_y_error > self.ENTER_THRESHOLD:
            alpha = self.alpha_base + (self.alpha_fast - self.alpha_base) * (
                (abs_y_error - self.ENTER_THRESHOLD) / self.ENTER_THRESHOLD
            )
        else:
            alpha = self.alpha_base

        lambda_k = alpha * target_lambda + (1 - alpha) * self.prev_lambda
        self.prev_lambda = lambda_k

        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        """Convert λ to (weight_human, weight_system)."""
        total = 1.0 + lambda_k
        return 1.0 / total, lambda_k / total

    def reset(self):
        """Reset state for new simulation."""
        self.prev_lambda = self.lambda_min
        self.in_performance_mode = False


__all__ = ['LateralAuthorityAllocator']
